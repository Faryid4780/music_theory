"""
AudioAnalytic Lib for ChordifyGet
"""
import pickle
# AudioAnalytic.py
# encoding: utf-8
import struct
import warnings
import wave
from os import mkdir
from os.path import exists

import librosa
import numpy as np
import pyaudio

import music_theory2 as music_theory

tet = music_theory.twelvetone_equal_temperament
all_phonic_num = music_theory.all_phonic_num[:88]


def find_nearest(var, target: np.ndarray):
    return np.argmin(np.abs(target - var))


# 读取音频文件
class MainProgram:
    def __init__(self, audio_file, _sample_length: int = 2):
        print("Loading wav...")
        # y->音频数据，sr->采样率（22050）
        y, sr = librosa.load(audio_file)

        # 通过 onset_envelope 方法计算音频的开始时间
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)

        # 使用 tempo 方法获取估计的 BPM
        tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)

        window_length_per_beat = 60 / tempo

        # 计算短时傅里叶变换，汉宁窗口长度等于歌曲1/_sample_length拍的时间
        n_fft_min = int(sr / (tet[1] - tet[0]))
        n_fft = int((sr * window_length_per_beat) / _sample_length)

        if n_fft < n_fft_min:
            warnings.warn("每" + str(_sample_length) + "拍汉宁窗口长度" + str(n_fft) + "小于最低要求" + str(n_fft_min) + "，可能会导致不精确")

        self.D = librosa.stft(y, n_fft=n_fft)  # shape[0] -> frequencies, shape[1] -> time
        self.wave_filename = audio_file
        self.wave_file = wave.open(audio_file, 'r')
        self.samplerate = self.wave_file.getframerate()
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

        # 寻找距离tet最近的freqs项，输出到nearest_indices
        # 初始化一个空列表来存储最近邻索引
        nearest_indices = []

        # 对于 tet 中的每个频率值
        for freq in tet:
            # 计算 freqs 中距离 freq 最近的索引
            nearest_index = np.argmin(np.abs(freqs - freq))

            # 将最近的索引添加到列表中
            nearest_indices.append(nearest_index)

        self.piano_D = dict(zip(all_phonic_num, np.abs(self.D[nearest_indices])))
        self.path = "stft_data\\" + audio_file.split("\\")[-1] + ".dat"
        wf = self.wave_file
        self.datasize = wf.getsampwidth() * wf.getnchannels()
        self.sum_tl = wf.getnframes() / self.samplerate

        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=self.p.get_format_from_width(wf.getsampwidth()),
                                  channels=wf.getnchannels(),
                                  rate=wf.getframerate(),
                                  output=True)

    def save_stft_data(self):
        print("Saving stft file...")

        # 时间点对应的stft数据
        time_to_stft = {}

        chord_list = []

        a = np.linspace(0, self.sum_tl, self.D.shape[1])
        for x in range(self.D.shape[1]):
            stft_data = np.array([a[x] for a in self.piano_D.values()])  # stft输入
            # piano_D
            # {'A4':[time1:..., time2:..., time3:..., ...]
            # 'Bb4':[time1:..., time2:..., time3:..., ...]}
            # 训练库stft数据保存, key=time, value=stft_data
            time_to_stft[a[x]] = stft_data

        if not exists('stft_data'):
            mkdir('stft_data')

        save_music_stft = open(self.path, 'wb')
        pickle.dump(time_to_stft, save_music_stft)
        save_music_stft.close()
        self.wave_file.close()
        print("STFT data has saved to", self.path)
        return self.path

    def __play_sound(self, st, et):
        if st >= et:
            return
        sf = round(st * self.samplerate)
        ef = round(et * self.samplerate)
        self.wave_file.setpos(sf)
        data = self.wave_file.readframes(ef - sf)
        self.stream.write(data)


class NewMainProgram:
    def __init__(self, audio_file, start_end_list: tuple):
        self.se_l = start_end_list

        self.wave_file = wave.open(open(audio_file, 'rb'))

        self.framerate = self.wave_file.getframerate()
        self.sampwidth = self.wave_file.getsampwidth()
        self.datasize = self.sampwidth * self.wave_file.getnchannels()

    def read_specific_frames(self, start_frame: int, end_frame: int):
        start_frame -= start_frame % self.datasize
        end_frame -= end_frame % self.datasize
        start_frame = int(start_frame)
        end_frame = int(end_frame)
        data = None
        try:
            self.wave_file.setpos(start_frame)
            data = self.wave_file.readframes(end_frame - start_frame)
        except wave.Error as e:
            print("Errored:", e)

        return data

    def record(self) -> list:
        recordings = []
        for tupl in self.se_l:
            assert isinstance(tupl, tuple)

            wave_data = self.read_specific_frames(tupl[0] * self.framerate,
                                                  tupl[1] * self.framerate)

            if wave_data is None:
                break

            fmt = f'<{len(wave_data) // self.sampwidth}h'  # '<h' means little-endian 16-bit signed integer
            amplitudes = struct.unpack(fmt, wave_data)
            fft_data = np.abs(np.fft.fft(amplitudes))
            fft_freq = np.fft.fftfreq(len(amplitudes))

            positive_freq_indices = np.where(fft_freq >= 0)
            fft_data_positive = fft_data[positive_freq_indices]
            fft_freq_positive = fft_freq[positive_freq_indices] * self.framerate

            # plt.figure(figsize=(10, 6))
            # plt.plot(fft_freq_positive, fft_data_positive)
            # plt.title('Frequency Spectrum')
            # plt.xlabel('Frequency (Hz)')
            # plt.ylabel('Amplitude')
            # plt.grid()
            # plt.show()
            piano_fft = fft_data_positive[[find_nearest(freq, fft_freq_positive) for freq in tet]]
            recordings.append(piano_fft)
        recordings = np.array(recordings)
        return recordings/np.mean(recordings)

    def close(self):
        self.wave_file.close()


if __name__ == '__main__':
    aa = NewMainProgram("youtube_downloaded\\H87GqJujcOk.wav", ((23, 24),))
    aa.record()
    aa.close()
