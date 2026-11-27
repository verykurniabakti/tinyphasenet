// Uji ekuivalensi di komputer: kernel TRANSPOSE_CONV pengganti (pola gather, src/transpose_conv_act_fix.cc) terhadap kernel acuan
// runtime (reference_ops and reference_integer_ops from esp-tflite-micro) on random tensors, for Float32, INT8, and 16x8.
// The replacement kernel is included directly from its source file (#include) so that the tested function is exactly the one that runs on the board.
// Build and run: see teste_kernel_host/jalankan.sh
#include "../src/transpose_conv_act_fix.cc"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

using namespace tflite;

// Sensitivity check of this test itself: with -DMUTASI the padding of the replacement kernel is deliberately shifted by one, so the test MUST report a difference.
#ifdef MUTASI
#define PERTURB(p) (p).padding_values.width += 1
#else
#define PERTURB(p) ((void)0)
#endif

static std::mt19937_64 rng(20261110);
static int ri(int lo, int hi) { return std::uniform_int_distribution<int>(lo, hi)(rng); }
static int64_t rl(int64_t lo, int64_t hi) { return std::uniform_int_distribution<int64_t>(lo, hi)(rng); }

struct Case { int in_w, cin, cout, fw, stride, pad, out_w; };

static Case random_case(bool paper_like) {
  static const int CIN[] = {1, 2, 3, 4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256};
  static const int COUT[] = {1, 2, 3, 4, 8, 16, 24, 32, 64};
  Case c;
  if (paper_like) { c.stride = 4; c.fw = 7; c.pad = ri(1, 3); c.in_w = ri(1, 48); c.out_w = c.in_w * 4; }
  else {
    static const int FW[] = {1, 2, 3, 4, 5, 6, 7, 9}, ST[] = {1, 2, 3, 4, 8};
    c.fw = FW[ri(0, 7)]; c.stride = ST[ri(0, 4)]; c.pad = ri(0, c.fw - 1); c.in_w = ri(1, 40);
    c.out_w = std::max(1, c.in_w * c.stride + ri(-c.fw, c.fw));
  }
  c.cin = CIN[ri(0, 13)]; c.cout = COUT[ri(0, 8)];
  return c;
}

static ConvParams make_params(const Case& c) {
  ConvParams p{}; p.stride_width = c.stride; p.stride_height = 1; p.padding_values.width = c.pad; p.padding_values.height = 0;
  p.dilation_width_factor = 1; p.dilation_height_factor = 1; return p;
}

int main() {
  const int N_RAND = 20000, N_PAPER = 5000, N_EXTREME = 600;
  long tested[3] = {0, 0, 0}, differ[3] = {0, 0, 0}; double maxdiff_f = 0;
  for (int type = 0; type < 3; ++type) {      // 0 = Float32, 1 = INT8, 2 = 16x8
    for (int k = 0; k < N_RAND + N_PAPER + N_EXTREME; ++k) {
      const bool paper = k >= N_RAND && k < N_RAND + N_PAPER, extreme = k >= N_RAND + N_PAPER;
      Case c = random_case(paper);
      if (extreme) { c.cin = 256; c.fw = 7; c.stride = 4; c.pad = ri(1, 3); c.in_w = ri(2, 12); c.out_w = c.in_w * 4; c.cout = ri(1, 8); }
      ConvParams p = make_params(c);
      RuntimeShape ish(4, std::vector<int>{1, 1, c.in_w, c.cin}.data()), fsh(4, std::vector<int>{c.cout, 1, c.fw, c.cin}.data());
      RuntimeShape osh(4, std::vector<int>{1, 1, c.out_w, c.cout}.data()), bsh(1, std::vector<int>{c.cout}.data()), nsh(4, std::vector<int>{0, 0, 0, 0}.data());
      const int nin = c.in_w * c.cin, nfl = c.cout * c.fw * c.cin, nout = c.out_w * c.cout;
      tested[type]++;
      if (type == 0) {
        std::vector<float> in(nin), fl(nfl), bias(c.cout), r(nout), f(nout);
        std::uniform_real_distribution<float> U(-2.f, 2.f);
        for (auto& x : in) x = U(rng); for (auto& x : fl) x = U(rng); for (auto& x : bias) x = U(rng);
        p.float_activation_min = -std::numeric_limits<float>::infinity(); p.float_activation_max = std::numeric_limits<float>::infinity();
        if (k % 3 == 0) { p.float_activation_min = 0.f; }
        reference_ops::TransposeConv(p, ish, in.data(), fsh, fl.data(), bsh, bias.data(), osh, r.data(), nsh, nullptr);
        PERTURB(p);
        TransposeConvFastF32(p, c.in_w, c.cin, in.data(), c.fw, c.cout, fl.data(), bias.data(), c.out_w, f.data());
        for (int i = 0; i < nout; ++i) maxdiff_f = std::max(maxdiff_f, (double)std::fabs(r[i] - f[i]));
        if (std::memcmp(r.data(), f.data(), nout * sizeof(float)) != 0) differ[0]++;
      } else {
        std::vector<int32_t> mult(c.cout), shift(c.cout);
        for (int i = 0; i < c.cout; ++i) { mult[i] = (int32_t)rl(1 << 30, (1LL << 31) - 1); shift[i] = ri(-12, 0); }
        const int lim = type == 1 ? 127 : 32767;
        std::vector<int8_t> fl(nfl);
        for (auto& x : fl) x = extreme ? (ri(0, 1) ? 127 : -127) : (int8_t)ri(-127, 127);
        if (type == 1) {
          std::vector<int8_t> in(nin), r(nout), f(nout); std::vector<int32_t> bias(c.cout), scratch(nout);
          for (auto& x : in) x = extreme ? (ri(0, 1) ? 127 : -128) : (int8_t)ri(-128, 127);
          for (auto& x : bias) x = (int32_t)rl(-(1 << 20), 1 << 20);
          p.input_offset = ri(-128, 127); p.output_offset = ri(-128, 127); p.quantized_activation_min = k % 4 == 0 ? -100 : -128; p.quantized_activation_max = k % 5 == 0 ? 90 : 127;
          reference_integer_ops::TransposeConv(p, mult.data(), shift.data(), ish, in.data(), fsh, fl.data(), bsh, bias.data(), osh, r.data(), nsh, nullptr, scratch.data());
        PERTURB(p);
          TransposeConvFast<int8_t, int32_t>(p, mult.data(), shift.data(), c.in_w, c.cin, in.data(), c.fw, c.cout, fl.data(), bias.data(), c.out_w, f.data(), p.input_offset, p.output_offset, true);
          if (std::memcmp(r.data(), f.data(), nout) != 0) differ[1]++;
        } else {
          std::vector<int16_t> in(nin), r(nout), f(nout); std::vector<int64_t> bias(c.cout), scratch(nout);
          for (auto& x : in) x = extreme ? (ri(0, 1) ? 32767 : -32768) : (int16_t)ri(-lim, lim);
          for (auto& x : bias) x = rl(-(1LL << 30), 1LL << 30);
          p.input_offset = 0; p.output_offset = 0; p.quantized_activation_min = k % 4 == 0 ? -20000 : -32768; p.quantized_activation_max = k % 5 == 0 ? 20000 : 32767;
          reference_integer_ops::TransposeConv<int64_t>(p, mult.data(), shift.data(), ish, in.data(), fsh, fl.data(), bsh, bias.data(), osh, r.data(), nsh, nullptr, scratch.data());
        PERTURB(p);
          TransposeConvFast<int16_t, int64_t>(p, mult.data(), shift.data(), c.in_w, c.cin, in.data(), c.fw, c.cout, fl.data(), bias.data(), c.out_w, f.data(), 0, 0, false);
          if (std::memcmp(r.data(), f.data(), nout * sizeof(int16_t)) != 0) differ[2]++;
        }
      }
    }
  }
  const char* nm[3] = {"Float32", "INT8", "16x8"};
  for (int t = 0; t < 3; ++t) std::printf("%-8s kasus: %ld  keluaran berbeda: %ld\n", nm[t], tested[t], differ[t]);
  std::printf("Float32 selisih mutlak maksimum: %.3g\n", maxdiff_f);
  return (differ[0] + differ[1] + differ[2]) ? 1 : 0;
}
