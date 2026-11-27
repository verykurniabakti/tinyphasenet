"""
TinyPhaseNet architecture: identical to eksperimen_tinyphasenet/tinyphasenet_1channel.py
(1-channel Z U-Net, SeparableConv1D, 4 stages of stride 4, softmax output [P, S, Noise]).

Parameterized for two ablations (the default behaviour does NOT change; input_channels=1 &
conv='separable' produces a model identical to the original version):
- input_channels=3: ablation "is the Z channel alone enough" (uses data3ch.py).
- conv='standard': replace SeparableConv1D with plain Conv1D, for the parameter/MAC ablation.
"""
from tensorflow.keras import layers, models
import config as C


def create_tinyphasenet(input_length=C.WINDOW, num_classes=3, base_filters=C.BASE_FILTERS,
                      input_channels=1, conv='separable', model_name='TinyPhaseNet'):
    ConvLayer = layers.SeparableConv1D if conv == 'separable' else layers.Conv1D
    in_name = 'seismic_input_z' if input_channels == 1 else 'seismic_input_enz'
    inputs = layers.Input(shape=(input_length, input_channels), name=in_name)

    def encoder_block(x, filters, kernel_size=7, stride=4):
        x = ConvLayer(filters, kernel_size, strides=stride, padding='same', activation='relu')(x)
        return layers.BatchNormalization()(x)

    def decoder_block(x, skip, filters, kernel_size=7, stride=4):
        x = layers.Conv1DTranspose(filters, kernel_size, strides=stride, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Concatenate()([x, skip])
        return ConvLayer(filters, kernel_size=3, padding='same', activation='relu')(x)

    e1 = encoder_block(inputs, base_filters)          # (768, 8)
    e2 = encoder_block(e1, base_filters * 2)          # (192, 16)
    e3 = encoder_block(e2, base_filters * 4)          # (48, 32)
    e4 = encoder_block(e3, base_filters * 8)          # (12, 64)

    b = ConvLayer(base_filters * 8, 7, padding='same', activation='relu')(e4)
    b = layers.BatchNormalization()(b)

    d3 = decoder_block(b, e3, base_filters * 4)
    d2 = decoder_block(d3, e2, base_filters * 2)
    d1 = decoder_block(d2, e1, base_filters)

    d0 = layers.Conv1DTranspose(base_filters, 7, strides=4, padding='same', activation='relu')(d1)
    d0 = layers.BatchNormalization()(d0)              # (3072, 8)
    outputs = layers.Conv1D(num_classes, 1, activation='softmax', name='prob_output')(d0)
    return models.Model(inputs=inputs, outputs=outputs, name=model_name)
