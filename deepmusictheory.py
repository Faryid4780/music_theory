import tensorflow as tf
import music_theory2 as music_theory
import pickle,os
import numpy as np

file_chord_root = None
file_chord_feature = None
filepath_1 = "networks\\chord_root_model.h5"
filepath_2 = "networks\\chord_feature_model.h5"

limit = 0.99

def translate_to_music_theory(input_) -> tuple:
    tones = []
    for k,v in zip(music_theory.phonic,input_):
        if v>limit:
            tones.append(k)
    return tuple(tones)

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
hidden_layer2 = tf.keras.layers.Dense(128, activation='relu')(hidden_layer)

# 输出层
tones_outputs = tf.keras.layers.Dense(12, activation='sigmoid', name='tones_output')(hidden_layer2)

# 模型构建
model_chord_feature = tf.keras.Model(inputs=[input_stft], outputs=[tones_outputs])

# 编译模型
# 二元化的tones，因为它非1即0，loss函数使用binary_crossentropy
model_chord_feature.compile(optimizer='adam',
                            loss={'tones_output': tf.keras.losses.binary_crossentropy},)
                             # metrics=[tf.keras.metrics.categorical_accuracy])
# fft: [13, 0, 0, 0, 0, 0, 34, 12, 0, 0, ...] (88 items)
# chord: [0 0 1 0 0 1 0 0 0 1 0] (12 items)


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


from logging import warning
def to_chord(tones_output:np.array, output_warnings:bool=True) -> music_theory.Chord:
    tones_output = translate_to_music_theory(tones_output)
    result = music_theory.Chord.from_tones(tones_output, None)
    if result == None:
        return result
    
    chord, appendix = result
    is_warned = False
    if appendix[0].__len__() > 0 or appendix[1].__len__() > 0:
        is_warned = True
        if output_warnings:
            warning(f"Unexpected tones: add {appendix[0]} lost {appendix[1]}")
    return chord,is_warned





