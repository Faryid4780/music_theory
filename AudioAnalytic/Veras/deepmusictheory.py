import tensorflow as tf
import music_theory2 as music_theory
import os
import numpy as np

file_chord_root = None
file_chord_feature = None
filepath_1 = "networks/chord_root_model.h5"
filepath_2 = "C:/Users/Administrator/PycharmProjects/Dikaer/AudioAnalytic/Veras/networks/chord_feature_model.h5"
filepath_2_best = "C:/Users/Administrator/PycharmProjects/Dikaer/AudioAnalytic/Veras/networks/chord_feature_model_best.h5"

limit = 0.999


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


# 输入层
input_stft = tf.keras.Input(shape=(88,))

# 隐藏层
hidden_layer = tf.keras.layers.Dense(128, activation='relu')(input_stft)

# # 隐藏层
hidden_layer2 = tf.keras.layers.Dense(256, activation='relu')(hidden_layer)

# 输出层
tones_outputs = tf.keras.layers.Dense(24, activation='sigmoid', name='tones_output')(hidden_layer2)
lowest_outputs = tf.keras.layers.Dense(12, activation='softmax', name='lowest_output')(hidden_layer2)

# 模型构建
model_chord_feature = tf.keras.Model(inputs=[input_stft], outputs=[tones_outputs, lowest_outputs])

# 编译模型
# 二元化的tones，因为它非1即0，loss函数使用binary_crossentropy
model_chord_feature.compile(optimizer='adam',
                            loss={'tones_output': tf.keras.losses.binary_crossentropy,
                                  'lowest_output': tf.keras.losses.binary_crossentropy}, )
# metrics=[tf.keras.metrics.categorical_accuracy])
# fft: [13, 0, 0, 0, 0, 0, 34, 12, 0, 0, ...] (88 items)
# chord: [0 0 1 0 0 1 0 0 0 1 0 ...] (24 items)
# lowest: [0 0 0 0 0 1 0 0 0 0 0] (12 items)

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
