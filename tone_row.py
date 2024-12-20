# encoding: utf-8
"""
Tone Row
"""
import numpy as np
import sounddevice as sd
import music_theory2 as mt
import matplotlib.pyplot as plt

square_wave_pattern = lambda x: 1 / (x + 1) if x % 2 == 0 else 0
tri_wave_pattern = lambda x: 1 / (x + 1) ** 2 if x % 2 == 0 else 0
sawtooth_wave_pattern = lambda x: (-1) ** x / (x + 1)


def binary(ndarray: np.ndarray):
    return ndarray / np.max(np.abs(ndarray))


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


def get_tone_row(f0: float, time: float, pattern: callable, velocity: float = 0.1,
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

    return tone * velocity


def get_chord(chord: mt.Chord, height: int = 4, inversion: int = 0) -> tuple:
    return tuple(map(lambda x: mt.twelvetone_equal_temperament[mt.findapn(x)],
                     chord.getChord(height, inversion)))


def get_amplitude_from_db(db_value: float):
    """从分贝获取归一化振幅"""
    return min((10 ** (min(db_value, 1) * 0.05)), 1)


def play_tone(note, duration=1.0):
    iter_time = 27

    r = get_tone_row(mt.twelvetone_equal_temperament[mt.findapn(note)], duration, generate_pulse_wave_pattern(),
                     iter_time=iter_time, mode='linear')

    sd.play(r, 44100)
    sd.wait()


def get_chord_wave(notes_chord: mt.Chord, duration, inversion: int = 0):
    c = get_chord(notes_chord, 3, inversion)
    c = list(c) + [mt.findapn(notes_chord.root + '2')]
    samplerate = 44100
    iter_time = 27

    r = np.zeros((int(duration * samplerate),))
    for freq in c:
        r += get_tone_row(freq, duration, square_wave_pattern, velocity=20, t_pattern=lambda t: (1 - t / 2),
                          sr=samplerate,
                          iter_time=iter_time,
                          mode='linear')
    return binary(r)


piano_amplitudes = tuple(map(get_amplitude_from_db, [-15, -25, -30, -35, -40, -42, -41, -46, -51, -55, -68, -65]))


def play_chord(c: list, duration: float = 1.0):
    for n in c:
        sd.play(get_chord_wave(n, duration), 44100)
        sd.wait()

import time
from threading import Thread

def display_process(pg: mt.ChordProgressionGroup):
    itr = mt.ChordIterator(pg)

    while itr.has_next:
        node = itr.next_node()
        print("-->", node.chord, node.chord.pure_tones if node.chord else None)

        print(f"{node.chord} -> {node.next.chord}: \n属进行性质: {node.get_dominant_data()}, "
              f"\n和弦功能: {node.chord_func},"
              f"调音: {node.tonic}\n")
        time.sleep(node.time(pg.bpm)-0.84/180)

def play_group(progreesion_group: mt.ChordProgressionGroup):
    bpm = progreesion_group.bpm
    wav = np.array([])

    ci = mt.ChordIterator(progreesion_group)
    while ci.has_next:
        node = ci.next_node()
        print(node.chord)
        if node.chord is None:
            wav = np.append(wav, np.zeros((node.time(bpm),)))
        else:
            wav = np.append(wav, get_chord_wave(node.chord, node.time(bpm)))

    # plt.title('Chord Progression Group to play')
    # plt.xticks(np.linspace(0, wav.size, wav.size//22050))
    # plt.plot(wav)
    # plt.show()
    print('---GO!---')
    Thread(target=display_process, args=(progreesion_group,)).start()
    sd.play(wav, 44100)
    sd.wait()


if __name__ == '__main__':
    group = mt.ChordProgressionGroup(bpm=135)
    # MomoneChinoi - Song-01
    intro_p = [(mt.Chord('C'), 4),
               (mt.Chord('A', mt.MINOR, 7), 2),
               (mt.Chord('G', mt.DOMINANT), 2),
               (mt.Chord('F'), 2),
               (mt.Chord('G'), 2),
               (mt.Chord('A', mt.MINOR, 7), 2),
               ]
    intro = intro_p + [(mt.Chord('G'), 1),
                       (mt.Chord('B', mt.DIMISHED), 1)]
    intro2 = intro_p + [(mt.Chord('G', mt.MINOR), 1),
                        (mt.Chord('Bb', mt.MAJOR), 1)]
    intro_p2 = intro_p.copy()
    intro_p2[-1] = (mt.Chord('G', mt.DOMINANT, 6), 2)
    intro3 = intro_p2 + [(mt.Chord('Bb'), 2)]

    intro4 = intro_p + [(mt.Chord('G'), 2)]
    intro5 = intro_p + [(mt.Chord('Bb', mt.MAJOR, 6), 2)]

    passing = [(mt.Chord('F', mt.MAJOR, 7), 2),
               (mt.Chord('E', mt.MINOR, 7), 2),
               (mt.Chord('G', mt.MINOR), 2),
               (mt.Chord('C', mt.DOMINANT, 7), 2),
               (mt.Chord('D', mt.MINOR, 7), 2),
               (mt.Chord('G', mt.DOMINANT, 7, sus=4), 2),
               (mt.Chord('D', mt.DIMISHED, 7), 2),
               (mt.Chord('A', mt.MINOR, 7), 2),
               (mt.Chord('D', mt.MINOR, 7), 2),
               (mt.Chord('E', mt.MAJOR), 2),
               (mt.Chord('A', mt.MINOR, 7), 2),
               (mt.Chord('G', sus=4), 2),
               (mt.Chord('F', mt.MAJOR, 7), 2),
               (mt.Chord('G', sus=4), 2),
               (mt.Chord('G', mt.DIMISHED), 2),
               (mt.Chord('A', mt.MINOR), 2)]

    main_p = [(mt.Chord('C', mt.MAJOR, 7), 2),
              (mt.Chord('B', mt.MINOR, 7), 2),
              (mt.Chord('D', sus=4), 2),
              (mt.Chord('G'), 2),
              (mt.Chord('C', mt.MAJOR, 7), 2),
              (mt.Chord('B', mt.MINOR, 7), 2),
              (mt.Chord('G'), 2),
              ]
    main1 = main_p + [(mt.Chord('F', mt.MAJOR, 6), 2)]
    main2 = main_p + [(mt.Chord('F'), 2)]
    main_p2 = main_p.copy()

    main_p2[2] = (mt.Chord('A', mt.DOMINANT, 7, sus=4), 2)
    main_p2[3] = (mt.Chord('G', mt.DOMINANT, 6), 2)

    main3 = main_p2 + [(mt.Chord('F'), 2)]

    middle_p = [(mt.Chord('C', mt.MAJOR, 7), 2),
                (mt.Chord('B', mt.MINOR, 7), 2), ]
    middle = middle_p + [(mt.Chord('E', mt.MINOR, 7), 2),
                         (mt.Chord('G', mt.DOMINANT, 7), 2)]
    middle2 = middle_p + [(mt.Chord('G'), 2),
                          (mt.Chord('F', mt.MAJOR, 6), 2)]
    middle3 = middle_p + [(mt.Chord('A', mt.DOMINANT, 7, sus=4), 2),
                          (mt.Chord('G', mt.DOMINANT, 6), 2)]
    middle4 = middle_p + [(mt.Chord('G'), 2),
                          (mt.Chord('F'), 2)]

    data = intro + intro2 + intro + intro3 + passing + (main1 + main2) * 2 + \
           middle + middle2 + middle3 + middle4 + \
           intro4 + intro5 + intro + intro3 + \
           passing + \
           (main1 + main2) * 2 + \
           (middle + middle2 + middle3 + middle4) * 4

    for c, v in data:
        group.append(c, beats=v)
    play_group(group)
