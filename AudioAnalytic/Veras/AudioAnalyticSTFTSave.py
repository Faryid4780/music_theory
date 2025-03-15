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

        self.D = librosa.cqt(y, sr=sr, hop_length=int(n_fft/4))
        self.wave_filename = audio_file
        self.wave_file = wave.open(audio_file, 'r')
        self.samplerate = self.wave_file.getframerate()


        self.piano_D = dict(zip(all_phonic_num, np.abs(self.D)))
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
        print("Saving cqt file...")

        # 时间点对应的stft数据
        time_to_stft = {}

        a = np.linspace(0, self.sum_tl, self.D.shape[1])
        for x in range(self.D.shape[1]):
            stft_data = np.array([a[x] for a in self.piano_D.values()])  # stft输入
            print(stft_data.shape)
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
        print("CQT data has saved to", self.path)
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
        self.filename = audio_file

        self.framerate = self.wave_file.getframerate()
        self.sampwidth = self.wave_file.getsampwidth()
        self.nframes = self.wave_file.getnframes()
        self.datasize = self.sampwidth * self.wave_file.getnchannels()

    def read_specific_frames(self, start_frame: int, end_frame: int):
        start_frame -= start_frame % self.datasize
        end_frame -= end_frame % self.datasize
        start_frame = max(int(start_frame), 0)
        end_frame = min(int(end_frame), self.nframes)
        data = None
        try:
            self.wave_file.setpos(start_frame)
            data = self.wave_file.readframes(end_frame - start_frame)
        except wave.Error as e:
            print("Errored:", e)

        return data

    def record(self, timesteps=6, neighbors=2) -> list:
        print("Recording...", self.filename)
        recordings = []
        loc = 0
        length = len(self.se_l)
        now = 0

        mul = 50
        print('[' + '=' * mul + ']')
        print('-', end='')

        for tupl in self.se_l:
            assert isinstance(tupl, tuple)

            # 获取音频数据
            wave_data_length = int((tupl[1] - tupl[0]) * self.framerate)
            neighbor_length = wave_data_length // timesteps * neighbors
            wave_data = self.read_specific_frames(tupl[0] * self.framerate - neighbor_length,
                                                  tupl[1] * self.framerate + neighbor_length)

            if wave_data is None:
                break

            fmt = f'<{len(wave_data) // self.sampwidth}h'  # '<h' means little-endian 16-bit signed integer
            amplitudes = struct.unpack(fmt, wave_data)

            # 将获取的幅度数据用于 CQT
            y = np.array(amplitudes, dtype=float)  # amplitudes 是从原始音频数据中解包出来的幅度数据
            # 使用 librosa.cqt 进行常数Q变换（CQT）
            cqt_result = librosa.cqt(y, sr=self.framerate, n_bins=88, bins_per_octave=12,
                                     hop_length=len(amplitudes) // (timesteps - 1)).T
            # 记录下来
            recordings.append(np.abs(cqt_result))

            if loc / length * 50 > now:
                print('=', end='')
                now += 1
            loc += 1
        print('>Done')
        return np.array(recordings) / np.mean(recordings)

    def close(self):
        self.wave_file.close()


if __name__ == '__main__':
    aa = NewMainProgram("youtube_downloaded\\H87GqJujcOk.wav", ((23, 24),(24, 26),(25.7, 27.5), (33.4, 38.7)))
    aa.close()
    
