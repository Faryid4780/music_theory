import tensorflow as tf
from keras import Sequential
from keras.src.layers import Bidirectional, Reshape, Conv1D, MaxPooling1D, MultiHeadAttention, GlobalAveragePooling1D, \
    Conv2D, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout
import music_theory2 as music_theory
import os
import numpy as np

file_chord_root = None
file_chord_feature = None
filepath_1 = "networks/chord_root_model.h5"
filepath_2 = "E:/PycharmProjects/Dikaer/AudioAnalytic/Veras/networks/chord_feature_model.h5"
filepath_2_best = "E:/PycharmProjects/Dikaer/AudioAnalytic/Veras/networks/chord_feature_model_best.h5"

limit = 0.85



def translate_to_music_theory(input_) -> tuple:
    tones = set()
    tones_o, lowest_o = input_
    for k, v in zip(music_theory.phonic * 2, tones_o):  # 24 tones
        if v > limit:
            tones.add(k)
    return tuple(tones), music_theory.phonic[np.argmax(lowest_o)]


# 输入层
inputs = tf.keras.Input(shape=(88,))

# 隐藏层
hidden_layer = tf.keras.layers.Dense(128, activation='relu')(inputs)

hidden_layer2 = tf.keras.layers.Dense(128, activation='relu')(hidden_layer)

# 根音输出层
root_output = tf.keras.layers.Dense(12, activation='softmax', name='root_output')(hidden_layer2)

# 模型构建
model_chord_root_type = tf.keras.Model(inputs=inputs, outputs=[root_output])

# 编译模型
model_chord_root_type.compile(optimizer='adam',
                              loss={'root_output': 'sparse_categorical_crossentropy'},
                              metrics=['accuracy'])

###

# 输入层，88键FFT
inputs = Input(shape=(None, 88))  # (batch, timesteps, 88)

# 调整形状为 (batch, timesteps, 88, 1) → 适配 Conv2D
x = Reshape((-1, 88, 1))(inputs)  # 形状变为 (batch, timesteps, 88, 1)

# Conv2D 提取时频特征（频域不降维）
conv_filters = 64
x = Conv2D(
    filters=conv_filters,
    kernel_size=(4, 5),  # 时间维度卷积核=4（捕捉长时依赖），频域卷积核=5（相邻5个半音），大三度
    activation="relu",
    padding="same"         # 保持时间、频域维度不变
)(x)  # 输出形状: (batch, timesteps, 88, 64)

# MaxPooling2D 仅压缩时域（频域维度保持88键）
x = MaxPooling2D(pool_size=(2, 1))(x)  # 输出形状: (batch, timesteps//2, 88, 64)

# 调整形状适配 Transformer
x = Reshape((-1, 88 * conv_filters))(x)  # 合并频域和通道 → (batch, timesteps//2, 88*64)
x = Dense(128, activation="relu")(x)  # 降维到适配注意力层

# Transformer 建模时间依赖（频域信息完整保留）
x = MultiHeadAttention(num_heads=4, key_dim=32)(x, x)  # 输出形状: (batch, timesteps//2, 128)
x = GlobalAveragePooling1D()(x)  # 聚合时间维度 → (batch, 128)

tones_outputs = Dense(24, activation='sigmoid', name='tones_output')(x)
lowest_outputs = Dense(12, activation='softmax', name='lowest_output')(x)

# 模型构建
model_chord_feature = Model(inputs=inputs, outputs=[tones_outputs, lowest_outputs])

# 编译模型
model_chord_feature.compile(optimizer='adam',
                            loss={'tones_output': tf.keras.losses.binary_crossentropy,
                                  'lowest_output': tf.keras.losses.binary_crossentropy},
                            metrics=["accuracy"])

model_chord_feature_best = None

# 加载或创建模型
if os.path.exists(filepath_1):
    print("Loading Old...")
    model_chord_root_type = tf.keras.models.load_model(filepath_1)
else:
    print("Created New")
    model_chord_root_type.save(filepath_1)

if os.path.exists(filepath_2):
    print("Loading Old...")
    model_chord_feature = tf.keras.models.load_model(filepath_2)
else:
    print("Created New")
    model_chord_feature.save(filepath_2)

if os.path.exists(filepath_2_best):
    print("Loading Best...")
    model_chord_feature_best = tf.keras.models.load_model(filepath_2_best)

from logging import warning


def to_chord(tones_output: np.array, lowest_output: np.array, output_warnings: bool = True):
    tones_output, lowest = translate_to_music_theory((tones_output, lowest_output))
    result = music_theory.Chord.from_tones(tones_output)
    if result is None:
        return result
    chord_slash = music_theory.ChordSlash(result[0].root,
                                          result[0].type,
                                          *result[0].args,
                                          **result[0].kwargs,
                                          lowest=lowest)

    result = (chord_slash, result[1])

    chord, appendix = result
    is_warned = False
    if appendix[0].__len__() > 0 or appendix[1].__len__() > 0:
        is_warned = True
        if output_warnings:
            warning(f"Unexpected tones: add {appendix[0]} lost {appendix[1]}")
    return chord, is_warned


if __name__ == '__main__':
    pass
