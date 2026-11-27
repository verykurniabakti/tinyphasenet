"""
Original PhaseNet (Zhu & Beroza 2019; code: the original PhaseNet repository, checkpoint 190703-214543) in Keras, for the three-channel, three-step comparison.
The structure follows the original UNet: depths=5, filters_root=8 (widths 8, 16, 32, 64, 128), kernel 7, pool/stride 4, input convolution 3->8 (with bias) + BN + ReLU; at each level: convolution 7 (no bias) + BN + ReLU
(kept as the skip connection), then a stride-4 convolution 7 (no bias) + BN + ReLU except at the deepest level; up path: stride-4 transposed convolution + BN + ReLU, concatenation [skip, up], convolution 7 + BN + ReLU;
output 1x1 convolution (3 classes) + softmax. The standard form has 268,445 trainable parameters (the same as the official checkpoint).
conv='separable': the plain convolutions 7 (input, lower level, up path) are replaced by SeparableConv1D; the transposed convolution and the 1x1 output stay (the same rule as TinyPhaseNet, model.py).
Output class order: the official model is [noise, P, S]; the models trained here follow our pipeline [P, S, noise] (like TinyPhaseNet); choose with output_order.
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

DEPTHS, ROOT, K, STRIDE = 5, 8, 7, 4


def create_phasenet_keras(input_length=3072, input_channels=3, conv='standard', name=None):
    Conv = layers.SeparableConv1D if conv == 'separable' else layers.Conv1D
    kw = dict(padding='same')
    inp = layers.Input(shape=(input_length, input_channels), name='seismic_input_enz')

    def bn_relu(x, nm):
        x = layers.BatchNormalization(name=nm + '_bn')(x)
        return layers.Activation('relu', name=nm + '_relu')(x)

    if conv == 'separable':
        x = Conv(ROOT, K, use_bias=True, name='input_conv', **kw)(inp)
    else:
        x = layers.Conv1D(ROOT, K, use_bias=True, name='input_conv', **kw)(inp)
    x = bn_relu(x, 'input')
    skips = [None] * DEPTHS
    for d in range(DEPTHS):
        f = ROOT * 2 ** d
        x = Conv(f, K, use_bias=False, name=f'down_conv1_{d + 1}', **kw)(x)
        x = bn_relu(x, f'down1_{d + 1}')
        skips[d] = x
        if d < DEPTHS - 1:
            x = Conv(f, K, strides=STRIDE, use_bias=False, name=f'down_conv3_{d + 1}', **kw)(x)
            x = bn_relu(x, f'down3_{d + 1}')
    for d in range(DEPTHS - 2, -1, -1):
        f = ROOT * 2 ** d
        x = layers.Conv1DTranspose(f, K, strides=STRIDE, use_bias=False, name=f'up_conv0_{d + 1}', **kw)(x)
        x = bn_relu(x, f'up0_{d + 1}')
        x = layers.Concatenate(name=f'concat_{d + 1}')([skips[d], x])
        x = Conv(f, K, use_bias=False, name=f'up_conv1_{d + 1}', **kw)(x)
        x = bn_relu(x, f'up1_{d + 1}')
    out = layers.Conv1D(3, 1, name='prob_output', activation='softmax')(x)
    return models.Model(inp, out, name=name or ('PhaseNet-DW' if conv == 'separable' else 'PhaseNet'))


def load_official_weights(model, ckpt=os.environ.get('PHASENET_CKPT', 'phasenet/model/190703-214543/model_95.ckpt')):
    """Transfer the weights of the official checkpoint (TF1) to the standard Keras model. The output of the transferred model is ordered [noise, P, S] (the official order)."""
    r = tf.train.load_checkpoint(ckpt)

    def g(n): return r.get_tensor(n)

    def bn(layer, scope):
        layer.set_weights([g(f'{scope}/gamma'), g(f'{scope}/beta'), g(f'{scope}/moving_mean'), g(f'{scope}/moving_variance')])

    model.get_layer('input_conv').set_weights([g('Input/input_conv/kernel')[:, 0], g('Input/input_conv/bias')]); bn(model.get_layer('input_bn'), 'Input/input_bn')
    for d in range(DEPTHS):
        model.get_layer(f'down_conv1_{d + 1}').set_weights([g(f'DownConv_{d}/down_conv1_{d + 1}/kernel')[:, 0]]); bn(model.get_layer(f'down1_{d + 1}_bn'), f'DownConv_{d}/down_bn1_{d + 1}')
        if d < DEPTHS - 1:
            model.get_layer(f'down_conv3_{d + 1}').set_weights([g(f'DownConv_{d}/down_conv3_{d + 1}/kernel')[:, 0]]); bn(model.get_layer(f'down3_{d + 1}_bn'), f'DownConv_{d}/down_bn3_{d + 1}')
    for d in range(DEPTHS - 2, -1, -1):
        model.get_layer(f'up_conv0_{d + 1}').set_weights([g(f'UpConv_{d}/up_conv0_{d + 1}/kernel')[:, 0]]); bn(model.get_layer(f'up0_{d + 1}_bn'), f'UpConv_{d}/up_bn0_{d + 1}')
        model.get_layer(f'up_conv1_{d + 1}').set_weights([g(f'UpConv_{d}/up_conv1_{d + 1}/kernel')[:, 0]]); bn(model.get_layer(f'up1_{d + 1}_bn'), f'UpConv_{d}/up_bn1_{d + 1}')
    model.get_layer('prob_output').set_weights([g('Output/output_conv/kernel')[:, 0], g('Output/output_conv/bias')])
    return model
