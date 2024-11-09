# encoding: utf-8
"""
Tone Row
"""
import numpy as np
import sounddevice as sd
from AudioAnalytic import music_theory2 as mt
import matplotlib.pyplot as plt

square_wave_pattern = lambda x: 1 / (x + 1) if x % 2 == 0 else 0
tri_wave_pattern = lambda x: 1 / (x + 1) ** 2 if x % 2 == 0 else 0
sawtooth_wave_pattern = lambda x: (-1) ** x / (x + 1)


def show_part_of_wave(data, iend: int = 1000):
    x = np.linspace(0, 1, iend)
    y = data[:iend]
    plt.plot(x, y, linestyle='-', color='b', label='WAVE')
    plt.title("Wave Display")
    plt.show()


def generate_trapezoid_wave_pattern(D: float = 0.25) -> callable:
    """
    :param D: 占空比
    """
    return lambda x: np.sin((x + 1) * np.pi * D) / (x + 1) ** 2 if x % 2 == 0 else 0


def generate_pulse_wave_pattern(D: float = 0.25) -> callable:
    """
    :param D: 占空比
    """
    return lambda x: np.cos((x + 1) * np.pi * D) / (x + 1) if x % 2 == 0 else 0


def get_tone_row(f0: float, time: float, pattern: callable, velocity: float = 3,
                 t_pattern: callable = lambda x: 1, sr: int = 44100, iter_time: int = 200,
                 mode: str = 'normalized'):
    """
    :param f0: 基频
    :param time: 时长，单位为s
    :param pattern: 泛音因子函数
    :param velocity: 音量，范围在[0,1]
    :param t_pattern: 时间渐变函数
    :param sr: 采样率
    :param iter_time: 泛音叠加次数
    :param mode: 模式，影响pattern，normalized要求pattern定义域为[0,1)，linear要求定义域为[0,iter_time)
    :return: 音频信息
    """
    length = int(time * sr)
    t = np.linspace(0, time, length, endpoint=False)  # 创建时间数组
    tp = np.array(tuple(map(t_pattern, t / time)))
    f = lambda freq: np.sin(2 * np.pi * freq * t) * tp  # 正弦波函数
    tone = np.zeros((length,))

    for n in range(1, iter_time + 1):
        if mode == 'normalized':
            tone += f(f0 * n) * pattern((n - 1) / iter_time)
        elif mode == 'linear':
            tone += f(f0 * n) * pattern(n - 1)
        else:
            raise AttributeError("Illegal argument value of 'mode': " + mode)

    return tone * velocity / iter_time


def get_chord(chord: mt.Chord, height: int = 4, inversion: int = 0) -> tuple:
    return tuple(map(lambda x: mt.twelvetone_equal_temperament[mt.findapn(x)],
                     chord.getChord(height, inversion)))


def get_amplitude_from_db(db_value: float):
    """从分贝获取归一化振幅"""
    return min((10 ** (min(db_value, 1) * 0.05)), 1)


def play_tone(note, duration=1.0):
    iter_time = 200

    r = get_tone_row(mt.twelvetone_equal_temperament[mt.findapn(note)], duration, generate_pulse_wave_pattern(), 2,
                     iter_time=iter_time, mode='linear')

    sd.play(r, 44100)
    sd.wait()


def get_chord_wave(notes_chord: mt.Chord, duration, inversion: int = 0):
    c = get_chord(notes_chord, 3, inversion)
    c = list(c) + [mt.findapn(notes_chord.root + '2')]
    samplerate = 44100
    iter_time = 200

    r = np.zeros((int(duration * samplerate),))
    for freq in c:
        r += get_tone_row(freq, duration, square_wave_pattern, velocity=20, t_pattern=lambda t: (1 - t),
                          sr=samplerate,
                          iter_time=iter_time,
                          mode='linear')
    return r


piano_amplitudes = tuple(map(get_amplitude_from_db, [-15, -25, -30, -35, -40, -42, -41, -46, -51, -55, -68, -65]))


def play_chord(c: list, duration: float = 1.0):
    for n in c:
        sd.play(get_chord_wave(n, duration), 44100)
        sd.wait()


if __name__ == '__main__':
    play_chord([
        mt.Chord('C', mt.MAJOR, 7)
    ], 5)
