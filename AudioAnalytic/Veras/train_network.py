import pickle
from AudioAnalytic.music_theory2 import phonic
import numpy as np
import tensorflow as tf
import deepmusictheory

#
filepath_1 = deepmusictheory.filepath_1
filepath_2 = deepmusictheory.filepath_2

# assert isinstance(stft_data,dict)
# 对deepmusictheory.py中的神经网络的对应训练库进行标记

# 定义回调函数
checkpoint_root = tf.keras.callbacks.ModelCheckpoint('networks\\chord_root_model_best.h5', monitor='accuracy',
                                                     save_best_only=True)
early_stopping_root = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=30)

checkpoint_feature = tf.keras.callbacks.ModelCheckpoint('networks/chord_feature_model_best.h5', monitor='loss',
                                                        save_best_only=True)
early_stopping_feature = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=50)


class TrainingLibImproved:
    def __init__(self, path, override: bool = True):
        if override:
            self.f = open(path, 'wb')
            self.data = None
        else:
            self.f = open(path, 'rb')
            self.data = pickle.load(self.f)

    def write_and_save(self, fft_chord_tupl: tuple, youtube_id: str, youtube_name: str, time_to_chord):
        save = {'youtube_id': youtube_id, 'youtube_name': youtube_name, 'data': fft_chord_tupl, 'ttc': time_to_chord}
        pickle.dump(save, self.f)
        self.f.close()

    def read_from(self) -> tuple:
        return self.data['data']

    def read_id(self) -> str:
        return self.data['youtube_id']

    def read_name(self) -> str:
        return self.data['youtube_name']

    def read_time_to_chord(self) -> str:
        return self.data['ttc']


def standardize_array(arr):
    """
  将数组标准化为均值为0，标准差为1的分布。

  参数：
  arr : numpy array
      要标准化的数组。

  返回：
  normalized_arr : numpy array
      标准化后的数组。
  """
    # 计算均值和标准差
    mean = np.mean(arr)
    std_dev = np.std(arr)

    # 标准化数组
    normalized_arr = (arr - mean) / std_dev

    return normalized_arr


## 训练用
def train2(training_libs_improved: list):
    global filepath_2

    tones_probabilities_array = []
    lowest_probabilities_array = []
    stft_inputs = []

    for tli in training_libs_improved:
        assert isinstance(tli, TrainingLibImproved)

        stft_data, chord_tuple = tli.read_from()
        assert len(stft_data) == len(chord_tuple)

        for x in range(len(stft_data)):
            tones_label = np.zeros(24)
            lowest_label = np.zeros(12)

            chord = chord_tuple[x]
            if chord is None:
                continue

            stft_inputs.append(stft_data[x])
            lowest_label[phonic.index(chord.lowest)] = 1

            for t in chord.getChord(4):
                tp = t[:-1]  # 去尾

                if t.endswith('4'):
                    tones_label[phonic.index(tp)] = 1
                elif t.endswith('5'):
                    tones_label[phonic.index(tp) + 12] = 1
                else:
                    raise Exception('BUG!!!!!!!!!!!!!!!')

            tones_probabilities_array.append(tones_label)
            lowest_probabilities_array.append(lowest_label)

    stft_inputs = np.array(stft_inputs)
    tones_probabilities_array = np.array(tones_probabilities_array)
    lowest_probabilities_array = np.array(lowest_probabilities_array)

    deepmusictheory.model_chord_feature.fit(
        x=(stft_inputs,),
        y=(tones_probabilities_array, lowest_probabilities_array),
        batch_size=32,
        epochs=25,
        callbacks=[checkpoint_feature, early_stopping_feature]
    )

    deepmusictheory.model_chord_feature.save(filepath_2)


import os


def get_all_libs() -> list:
    folder_path = 'E:\\PycharmProjects\\VideoGet\\stft_data_modified'  # 替换为你的文件夹路径
    dat_files = []

    for filename in os.listdir(folder_path):
        if filename.endswith('.dat'):
            print(filename)
            dat_files.append(os.path.join(folder_path, filename))

    return [TrainingLibImproved(path, False) for path in dat_files]


if __name__ == '__main__':
    train2(get_all_libs())
