import pickle
from AudioAnalytic.music_theory2 import Chord, phonic
import numpy as np
import tensorflow as tf
import deepmusictheory
from os.path import exists
#
filepath_1 = deepmusictheory.filepath_1
filepath_2 = deepmusictheory.filepath_2

# assert isinstance(stft_data,dict)
# 对deepmusictheory.py中的神经网络的对应训练库进行标记

# 定义回调函数
checkpoint_root = tf.keras.callbacks.ModelCheckpoint('networks\\chord_root_model_best.h5', monitor='accuracy', save_best_only=True)
early_stopping_root = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=30)

checkpoint_feature = tf.keras.callbacks.ModelCheckpoint('networks/chord_feature_model_best.h5', monitor='loss', save_best_only=True)
early_stopping_feature = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=50)


class TrainingLib:
    def __init__(self, stft_filepath: str, override: bool = True):
        self.stft_filename = stft_filepath
        stft_file = open(stft_filepath, 'rb')
        self.train_filename = stft_filepath.replace('.dat', '') + "_train_data.dat"
        self.train_data = {}

        if exists(self.train_filename) and not override:
            self.train_file = open(self.train_filename, 'rb')
            self.train_data = pickle.load(self.train_file)
            self.train_file.close()

        self.train_file = open(self.train_filename, 'wb')
        self.stft_data = pickle.load(stft_file)
        assert isinstance(self.stft_data, dict)
        stft_file.close()

        self.train_pickler = pickle.Pickler(self.train_file)
        self.ks = np.array(tuple(self.stft_data.keys()))
        # self.iset = set()

    def add(self, time: float, chord: Chord):
        i = np.argmin(np.abs(self.ks - time))
        # if i in self.iset:
        #     raise Exception("Repeat")
        # self.iset.add(i)
        self.train_data[self.ks[i]] = chord

    def save(self):
        self.train_pickler.dump(self.train_data)
        self.train_file.close()

# tl = TrainingLib("stft_data\\【東北きりたん】管理【nori（初投稿）】.wav.dat")
# tl3 = TrainingLib("stft_data\\c418.wav.dat")
# tl4 = TrainingLib("stft_data\\【UTAU7人】沙丁鱼从地里钻了出来.wav.dat")
# # tl5 = TrainingLib("C:\\Users\\Administrator\\PycharmProjects\\VideoGet\\stft_data\\50ABOddeI3c.wav.dat", False)
# # tl6 = TrainingLib("C:\\Users\\Administrator\\PycharmProjects\\VideoGet\\stft_data\\7e8lNH6CRRY.wav.dat", False)
# # tl7 = TrainingLib("C:\\Users\\Administrator\\PycharmProjects\\VideoGet\\stft_data\\_1fCpoMXApE.wav.dat", False)
# # tl8 = TrainingLib("C:\\Users\\Administrator\\PycharmProjects\\VideoGet\\stft_data\\Z2zVg2xKCQQ.wav.dat", False)
# # tl9 = TrainingLib("C:\\Users\\Administrator\\PycharmProjects\\VideoGet\\stft_data\\_I92YDvuw_Q.wav.dat", False)
# # tl10 = TrainingLib("C:\\Users\\Administrator\\PycharmProjects\\VideoGet\\stft_data\\-dXgfyppGAY.wav.dat", False)
# # tl11 = TrainingLib("C:\\Users\\Administrator\\PycharmProjects\\VideoGet\\stft_data\\kngTMwZ1GDM.wav.dat", False)
# # tl12 = TrainingLib("C:\\Users\\Administrator\\PycharmProjects\\VideoGet\\stft_data\\7liS6sVDDVE.wav.dat", False)
# # tl13 = TrainingLib("C:\\Users\\Administrator\\PycharmProjects\\VideoGet\\stft_data\\sGNOLLOfFwM.wav.dat", False)
# # tl14 = TrainingLib("C:\\Users\\Administrator\\PycharmProjects\\VideoGet\\stft_data\\010nHWXwdYw.wav.dat", False)
# # tl15 = TrainingLib("C:\\Users\\Administrator\\PycharmProjects\\VideoGet\\venv\\stft_data\\N8nGig78lNs.wav.dat", False)
# # tl16 = TrainingLib("C:\\Users\\Administrator\\PycharmProjects\\VideoGet\\venv\\stft_data\\H87GqJujcOk.wav.dat", False)
#
#
# tl.add(16.383, Chord('C', MINOR))
# tl.add(17.095, Chord('Bb', MAJOR))
# tl.add(17.570, Chord('A', AUGMENTED,7))
# tl.add(18.198, Chord('Ab', MAJOR, 7))
# tl.add(18.990, Chord('G', MINOR, 7, others=Interval(13,-1)))
# tl.add(19.740, Chord('F', MINOR, 7))
# tl.add(20.106, Chord('F', MINOR, 7))
# tl.add(19.9, Chord('F', MINOR, 7))
# tl.add(20.716, Chord('Bb', MAJOR))
# tl.add(21.132, Chord('Bb', AUGMENTED, 7))
# tl.add(21.39, Chord('C', MINOR))
# tl.add(22.435, Chord('E', DIMISHED, 7))
# tl.add(22.67, Chord('E', DIMISHED, 7))
# tl.add(23.447, Chord('Ab', MAJOR, 7))
# tl.add(24.103, Chord('G', MAJOR, 7, others=Interval(9,-1)))
# tl.add(24.558, Chord('G', MAJOR, 7, others=Interval(9,-1)))
# tl.add(25.139, Chord('G', MAJOR, 7))
# tl.add(25.953, Chord('C', MINOR, add=11))
# tl.add(26.309, Chord('C', MINOR))
# tl.add(26.848, Chord('F', MINOR, 7))
# tl.add(27.669, Chord('Bb', AUGMENTED, 7))
# tl.add(28.725, Chord('C', MINOR, add=11))
# tl.add(29.718, Chord('Eb', MAJOR, 7))
# tl.add(30.19, Chord('Ab', MINOR, 7))
# tl.add(31.240, Chord('Db', MINOR, 7))
# tl.add(32.060, Chord('G', MINOR, 7))
# tl.add(32.959, Chord('Bb', MINOR, 7))
# tl.add(33.858, Chord('E', MAJOR, 7, sus=4))
# tl.add(35.349, Chord('Gb', MINOR, 7))
# tl.add(35.899, Chord('F', AUGMENTED))
# tl.add(36.233, Chord('Ab', AUGMENTED))
# tl.add(36.540, Chord('G', AUGMENTED))
# tl.add(36.905, Chord('G', MAJOR, 7))
# tl.add(37.424, Chord('C', MINOR))
# tl.add(37.9, Chord('C', MINOR))
# tl.add(38.3, Chord('C', MINOR))
# tl.add(38.74, Chord('C', MINOR))
# tl.add(39.128, Chord('C', MINOR))
# tl.add(39.885, Chord('C', MINOR))
# tl.add(40.539, Chord('C', MINOR))
# tl.add(40.942, Chord('F', MINOR, 7))
# tl.add(41.744, Chord('G', MINOR, 7))
# tl.add(42.685, Chord('C', MINOR))
# tl.add(43.107, Chord('B', AUGMENTED))
# tl.add(43.467, Chord('Eb', MAJOR))
# tl.add(44.003, Chord('G', MINOR))
# tl.add(44.354, Chord('Ab', MAJOR, 7))
# tl.add(44.672, Chord('Ab', MAJOR, 7))
# tl.add(45.334, Chord('G', MINOR, 7))
# tl.add(45.620, Chord('Gb', MAJOR, 7, others=Interval(5,-1))) # very short
# tl.add(46.004, Chord('F', MINOR, 7))
# tl.add(46.640, Chord('Bb', MAJOR, 7))
# tl.add(47.083, Chord('C', MINOR, 7))
# tl.add(47.537, Chord('C', MINOR))
# tl.add(48.120, Chord('F', MINOR, 7))
# tl.add(48.820, Chord('G', AUGMENTED, 7, add=9))
# tl.add(49.807, Chord('C', MAJOR, sus=4))
# tl.add(50.731, Chord('C', MINOR))
# tl.add(51.1, Chord('C', MINOR))
# tl.add(51.519, Chord('C', MINOR))
# tl.add(52.1, Chord('C', MINOR))
# tl.add(52.7, Chord('C', MINOR))
# tl.add(53.232, Chord('C', MINOR))
# tl.add(53.9, Chord('C', MINOR))
# tl.add(54.245, Chord('C', MINOR))
# tl.add(55.018, Chord('F', MINOR, 7))
# tl.add(55.884, Chord('G', MINOR, 7))
# tl.add(56.634, Chord('C', MINOR))
# tl.add(57.21, Chord('B', AUGMENTED))
# tl.add(57.689, Chord('Eb', MAJOR))
# tl.add(58.174, Chord('G', MINOR))
# tl.add(58.585, Chord('Ab', MAJOR, 7))
# tl.add(59.425, Chord('G', MINOR, 7))
# tl.add(59.787, Chord('Gb', MAJOR, 7, others=Interval(5,-1))) # very short
# tl.add(60.146, Chord('F', MINOR, 7))
# tl.add(60.794, Chord('Bb', MAJOR, 7))
# tl.add(61.297, Chord('C', MINOR, 7))
# tl.add(61.803, Chord('C', MINOR))
# tl.add(62.251, Chord('F', MINOR, 7))
# tl.add(63.020, Chord('G', AUGMENTED, 7, add=9))
# tl.add(63.809, Chord('C', MAJOR, sus=4))
# tl.add(64.901, Chord('C', MINOR))
#
# tl3.add(2.822, Chord('Ab',MAJOR,7))
# tl3.add(3.277, Chord('Ab',MAJOR,7))
# tl3.add(5.830, Chord('Db',MAJOR,7,sus=2))
# tl3.add(6.582, Chord('Db',MAJOR,7,sus=2))
# tl3.add(8.970, Chord('Ab',MAJOR,7,sus=2))
# tl3.add(9.5, Chord('Ab',MAJOR,7))
# tl3.add(11.561, Chord('Db',MAJOR,7,sus=2))
# tl3.add(12.3, Chord('Db',MAJOR,7,sus=2))
# tl3.add(15.340, Chord('Ab',MAJOR,7))
# tl3.add(17.886, Chord('Db',MAJOR,7,sus=2))
# tl3.add(18.5, Chord('Db',MAJOR,7,sus=2))
# tl3.add(21.727, Chord('Ab',MAJOR,7,sus=2))
# tl3.add(23.78, Chord('Db',MAJOR,sus=2))
#
# # 【UTAU7人】沙丁鱼从地里钻了出来
# tl4.add(7.7, Chord('D',MINOR,7))
# tl4.add(8.55, Chord('A',MINOR,7))
# tl4.add(9.403, Chord('D',MINOR,7))
# tl4.add(10.296, Chord('A',MINOR,7))
# tl4.add(11.16, Chord('D',MINOR,7))
# tl4.add(12.022, Chord('A',MINOR,7))
# tl4.add(12.498, Chord('D',MINOR,7))
# tl4.add(13.17, Chord('A',MAJOR,7))
# tl4.add(13.58, Chord('D',MINOR,7))
# tl4.add(13.94, Chord('Eb',MINOR,7))
# tl4.add(14.459, Chord('D',MINOR,7))
# tl4.add(15.270, Chord('A',MINOR,7))
# tl4.add(15.666, Chord('A',MINOR,7))
# tl4.add(16.172, Chord('D',MINOR,7))
# tl4.add(16.557, Chord('D',MINOR,7))
# tl4.add(17.132, Chord('A',MINOR,7))
# tl4.add(17.505, Chord('D',MINOR,7))
# tl4.add(17.99, Chord('G',MINOR,7))
# tl4.add(18.373, Chord('G',MINOR,7))
# tl4.add(18.917, Chord('C',DOMINANT,7))
# tl4.add(19.227, Chord('C',DOMINANT,7))
# tl4.add(19.76, Chord('F',MAJOR,7))
# tl4.add(20.15, Chord('F',MAJOR,7))
# tl4.add(20.716, Chord('F',DOMINANT,7))
# tl4.add(21.096, Chord('F',DOMINANT,7))
# tl4.add(21.6, Chord('E',MINOR,7,others=Interval(5,-1)))
# tl4.add(22.04, Chord('E',MINOR,7,others=Interval(5,-1)))
# tl4.add(22.472, Chord('A',DOMINANT,7))
# tl4.add(22.838, Chord('A',DOMINANT,7))
# tl4.add(23.375, Chord('D',MINOR,7))
# tl4.add(23.781, Chord('D',MINOR,7))
# tl4.add(24.193, Chord('C',MAJOR))
# tl4.add(24.659, Chord('C',MAJOR))
# tl4.add(25.234, Chord('Bb',MAJOR,7))
# tl4.add(25.405, Chord('Bb',MAJOR,7))
# tl4.add(25.905, Chord('A',MINOR,7))
# tl4.add(26.319, Chord('A',MINOR,7))
# tl4.add(26.682, Chord('D',MINOR,7))
# tl4.add(27.021, Chord('Db',MAJOR,7))
# tl4.add(27.371, Chord('C',MAJOR,7))
# tl4.add(27.683, Chord('B',MAJOR,7))
# tl4.add(28.079, Chord('A',DOMINANT,7))
# tl4.add(28.690, Chord('D',MINOR,7))
# tl4.add(29.016, Chord('D',MINOR,7))
# tl4.add(29.621, Chord('C',MAJOR))
# tl4.add(29.963, Chord('C',MAJOR))
# tl4.add(30.471, Chord('Bb',MAJOR,7))
# tl4.add(30.836, Chord('Bb',MAJOR,7))
# tl4.add(31.503, Chord('F',MAJOR,7))
# tl4.add(32.133, Chord('G',MINOR,7))
# tl4.add(32.663, Chord('G',MINOR,7))
# tl4.add(33.2, Chord('C',DOMINANT,7))
# tl4.add(33.576, Chord('C',DOMINANT,7))
# tl4.add(34.020, Chord('F',MAJOR,7))
# tl4.add(34.369, Chord('F',MAJOR,7))
# tl4.add(34.872, Chord('F',DOMINANT,7))
# tl4.add(35.272, Chord('F',DOMINANT,7))
# tl4.add(35.799, Chord('Bb',MAJOR,7))
# tl4.add(36.104, Chord('Bb',MAJOR,7))
# tl4.add(36.671, Chord('A',MINOR,7))
# tl4.add(37.048, Chord('A',MINOR,7))
# tl4.add(37.583, Chord('D',MINOR,7))
# tl4.add(37.96, Chord('D',MINOR,7))
# tl4.add(38.494, Chord('C',MAJOR))
# tl4.add(38.826, Chord('C',MAJOR))
# tl4.add(39.342, Chord('G',MINOR,7))
# tl4.add(39.75, Chord('G',MINOR,7))
# tl4.add(40.215, Chord('C',DOMINANT,7))
# tl4.add(40.605, Chord('C',DOMINANT,7))
# tl4.add(41.216, Chord('E',MINOR,7,others=Interval(5,-1)))
# tl4.add(41.494, Chord('E',MINOR,7,others=Interval(5,-1)))
# tl4.add(41.997, Chord('A',MAJOR))
# tl4.add(42.388, Chord('A',MAJOR))
# tl4.add(42.761, Chord('D',MINOR,7))
# tl4.add(43.325, Chord('D',MINOR,7))
# tl4.add(43.851, Chord('C',MAJOR))
# tl4.add(44.175, Chord('C',MAJOR))
# tl4.add(44.646, Chord('Bb',MAJOR,7))
# tl4.add(44.939, Chord('Bb',MAJOR,7))
# tl4.add(45.248, Chord('F',MAJOR,7))
# tl4.add(45.937, Chord('F',MAJOR,7))
# tl4.add(46.438, Chord('G',MINOR,7))
# tl4.add(46.868, Chord('G',MINOR,7))
# tl4.add(47.384, Chord('C',DOMINANT,7))
# tl4.add(47.768, Chord('C',DOMINANT,7))
# tl4.add(48.186, Chord('F',MAJOR,7))
# tl4.add(48.661, Chord('F',MAJOR,7))
# tl4.add(49.143, Chord('F',DOMINANT,7))
# tl4.add(49.983, Chord('Bb',MAJOR,7))
# tl4.add(50.47, Chord('Bb',MAJOR,7))
# tl4.add(50.885, Chord('A',MINOR,7))
# tl4.add(51.283, Chord('A',MINOR,7))
# tl4.add(51.852, Chord('D',MINOR,7))
# tl4.add(52.214, Chord('D',MINOR,7))
# tl4.add(52.692, Chord('F',DOMINANT,7))
# tl4.add(53.011, Chord('F',DOMINANT,7))
# tl4.add(53.427, Chord('G',MINOR,7))
# tl4.add(53.96, Chord('G',MINOR,7))
# tl4.add(54.485, Chord('C',DOMINANT,7))
# tl4.add(54.886, Chord('C',DOMINANT,7))
# tl4.add(55.363, Chord('F',MAJOR))
# tl4.add(55.712, Chord('F',MAJOR))
# tl4.add(56.388, Chord('C',MAJOR))
# tl4.add(57.402, Chord('Bb',MAJOR,7))
# tl4.add(58.219, Chord('Bb',MAJOR,7))
# tl4.add(58.816, Chord('Bb',MAJOR,7))
# tl4.add(59.1, Chord('Bb',MAJOR,7))
# tl4.add(59.826, Chord('C',DOMINANT,7))
# tl4.add(60.17, Chord('C',DOMINANT,7))
# tl4.add(60.48, Chord('D',MINOR,7))
# tl4.add(61.40, Chord('D',MINOR,7))
# tl4.add(61.84, Chord('D',MINOR,7))
# tl4.add(62.536, Chord('C',MINOR,7))
# tl4.add(62.814, Chord('C',MINOR,7))
# tl4.add(63.436, Chord('F',DOMINANT,7))
# tl4.add(63.75, Chord('F',DOMINANT,7))
# tl4.add(64.494, Chord('Bb',MAJOR,7))
# tl4.add(64.993, Chord('Bb',MAJOR,7))
# tl4.add(65.576, Chord('Bb',MAJOR,7))
# tl4.add(66.097, Chord('Bb',MAJOR,7))
# tl4.add(66.942, Chord('A',DOMINANT,7))
# tl4.add(67.828, Chord('D',MINOR,7))
# tl4.add(68.706, Chord('D',MINOR,7))
# tl4.add(69.603, Chord('C',MINOR,7))
# tl4.add(69.937, Chord('C',MINOR,7))
# tl4.add(70.442, Chord('F',DOMINANT,7))
# tl4.add(70.826, Chord('F',DOMINANT,7))
# tl4.add(71.921, Chord('Bb',MAJOR,7))
# tl4.add(72.995, Chord('Bb',MAJOR,7))
# tl4.add(74.083, Chord('C',DOMINANT,7))
# tl4.add(74.483, Chord('C',DOMINANT,7))
# tl4.add(74.984, Chord('D',MINOR,7))
# tl4.add(75.521, Chord('D',MINOR,7))
# tl4.add(75.981, Chord('D',MINOR,7))
# tl4.add(76.685, Chord('C',MINOR,7))
# tl4.add(77.076, Chord('C',MINOR,7))
# tl4.add(77.687, Chord('F',DOMINANT,7))
# tl4.add(77.993, Chord('F',DOMINANT,7))
# tl4.add(79.286, Chord('Bb',MAJOR,7))
# tl4.add(80.382, Chord('Bb',MAJOR,7))
# tl4.add(81.269, Chord('A',DOMINANT,7))
# tl4.add(81.521, Chord('A',DOMINANT,7))
# tl4.add(82.493, Chord('D',MINOR,7))
# tl4.add(83.103, Chord('D',MINOR,7))
# tl4.add(83.786, Chord('C',MINOR,7))
# tl4.add(84.2, Chord('C',MINOR,7))
# tl4.add(84.714, Chord('F',DOMINANT,7))
# tl4.add(85.55, Chord('D',MINOR,7))
# tl4.add(86.605, Chord('A',MINOR,7))
# tl4.add(87.49, Chord('D',MINOR,7))
# tl4.add(88.486, Chord('A',MINOR,7))
# tl4.add(91.382, Chord('A',DOMINANT,7))
# tl4.add(92.103, Chord('Eb',MINOR,7))

class TrainingLibImproved:
    def __init__(self, path, override: bool = True):
        if override:
            self.f = open(path, 'wb')
        else:
            self.f = open(path, 'rb')

    def write_and_save(self, fft_chord_dict: tuple):
        pickle.dump(fft_chord_dict, self.f)
        self.f.close()

    def read_from(self) -> tuple:
        a = pickle.load(self.f)
        assert isinstance(a, tuple)
        return a

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
def train2(training_libs_improved:list):
    global filepath_2

    tones_probabilities_array = []
    stft_inputs = []

    for tli in training_libs_improved:
        assert isinstance(tli, TrainingLibImproved)

        stft_data, chord_tuple = tli.read_from()
        assert len(stft_data) == len(chord_tuple)

        for x in range(len(stft_data)):
            tones_label = np.zeros(12)
            chord = chord_tuple[x]
            if chord is None:
                continue

            stft_inputs.append(stft_data[x])

            for t in chord.pure_tones:
                tones_label[phonic.index(t)] = 1

            tones_probabilities_array.append(tones_label)
                
    stft_inputs = np.array(stft_inputs)
    tones_probabilities_array = np.array(tones_probabilities_array)

    deepmusictheory.model_chord_feature.fit(
        x=(stft_inputs,),
        y=(tones_probabilities_array,),
        batch_size=len(stft_inputs),
        epochs=1000,
        callbacks=[checkpoint_feature, early_stopping_feature]
    )

    deepmusictheory.model_chord_feature.save(filepath_2)
    
import os

def get_all_libs() -> list:
    folder_path = 'C:\\Users\\Administrator\\PycharmProjects\\VideoGet\\venv\\stft_data'  # 替换为你的文件夹路径
    dat_files = []

    for filename in os.listdir(folder_path):
        if filename.endswith('.dat'):
            dat_files.append(os.path.join(folder_path, filename))

    return [TrainingLibImproved(path, False) for path in dat_files]

if __name__ == '__main__':
    train2(get_all_libs())









