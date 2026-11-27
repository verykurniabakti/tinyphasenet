"""
Step 3: post-training quantization, two variants of the same Float32 model:
- full INT8  (int8 activations, int8 weights)  -> models/tinyphasenet_int8.tflite
- 16x8       (int16 activations, int8 weights) -> models/tinyphasenet_16x8.tflite
Calibration uses random windows from the 'train' split (earthquake + noise), not from val/test.
"""
import os
import numpy as np
import tensorflow as tf
import config as C
from data import load_stead, zscore, random_start

N_CALIB = 300
VARIANTS = {
    'int8': ([tf.lite.OpsSet.TFLITE_BUILTINS_INT8], tf.int8),
    '16x8': ([tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8], tf.int16),
}

if __name__ == '__main__':
    meta, z = load_stead()
    train = meta[meta.split == 'train'].sample(n=N_CALIB, random_state=C.SEED)
    model = tf.keras.models.load_model(os.path.join(C.MODEL_DIR, 'tinyphasenet_best.keras'))
    run_model = tf.function(lambda x: model(x))
    concrete = run_model.get_concrete_function(tf.TensorSpec([1, C.WINDOW, 1], tf.float32))

    for name, (ops, io_type) in VARIANTS.items():
        rng = np.random.default_rng(C.SEED)   # identical calibration windows for both variants

        def representative_data_gen():
            for i, row in train.iterrows():
                st = random_start(rng, row.label, row.p_arrival_sample, C.STEAD_NPTS)
                yield [zscore(z[i, st:st + C.WINDOW]).reshape(1, C.WINDOW, 1)]

        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete], model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_data_gen
        converter.target_spec.supported_ops = ops
        converter.inference_input_type = io_type
        converter.inference_output_type = io_type
        tflite_model = converter.convert()

        path = os.path.join(C.MODEL_DIR, f'tinyphasenet_{name}.tflite')
        with open(path, 'wb') as f:
            f.write(tflite_model)
        print(f"[{name}] {path}: {len(tflite_model)} byte ({len(tflite_model) / 1024:.2f} KB, 1 KB = 1024 B)")
    print(f"Model parameters: {model.count_params():,}")
