# AudioAnalytic.py (src file)
# encoding: utf-8
import wave,librosa,warnings,pyaudio,threading,asyncio,mido
from time import time as tm
import matplotlib.pyplot as plt
import music_theory2 as music_theory
import numpy as np
from scipy import stats
from matplotlib import rcParams
from concurrent.futures import ThreadPoolExecutor
from mido import Message
import tone_row
import sounddevice as sd
from train_network import TrainingLib

# 预设置

# 创建MIDI输出端口
out_port = mido.open_output()


# 播放音符的协程
async def play_note_async(note, duration=1.0):
    msg_on = Message('note_on', note=note, velocity=70)
    msg_off = Message('note_off', note=note, velocity=70)

    # 发送Note On消息
    out_port.send(msg_on)
    await asyncio.sleep(duration)
    # 发送Note Off消息
    out_port.send(msg_off)
# 同时播放多个音符
async def play_multiple_notes_async(notes_chord:music_theory.Chord, duration, available=True):
    if notes_chord is None or not available:
        return
    def p(r, samplerate):
        sd.play(r, samplerate)
        sd.wait()
    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor() as pool:
        await loop.run_in_executor(pool, p, tone_row.get_chord_wave(notes_chord, duration), 44100)


# 设置中文字体（以微软雅黑为例）
rcParams['font.family'] = 'Microsoft YaHei'

tet = music_theory.twelvetone_equal_temperament
all_phonic_num = music_theory.all_phonic_num[:88]

### 函数部分


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

# 读取音频文件与stft预处理

# 读取音频文件
audio_file = r'【ピアノアレンジ】さよなら幽霊無人街 - 玄米ねこまんま(電ǂ鯨).wav'
youtube_area = None
have_bass = True
need_to_save = False
# y->音频数据，sr->采样率（22050）
y, sr = librosa.load(audio_file)

# 通过 onset_envelope 方法计算音频的开始时间
onset_env = librosa.onset.onset_strength(y=y, sr=sr)

# 使用 tempo 方法获取估计的 BPM
tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)

print("Get Tempo:",tempo)

window_length_per_beat = 60/tempo # Time Unit: sec

# 计算短时傅里叶变换，汉宁窗口长度等于歌曲1/_sample_length拍的时间
_sample_length = 0.25
n_fft_min = int(sr/(tet[1]-tet[0]))
# 时长为一拍的窗口，长度为sr*window_length_per_beat
n_fft = round((sr*window_length_per_beat) / _sample_length)

if n_fft<n_fft_min:
    warnings.warn("每"+str(_sample_length)+"拍汉宁窗口长度小于最低要求"+str(n_fft_min)+"，可能会导致不精确")

# n_fft是对于y而言，hop_length默认是1/4倍n_fft
D = librosa.stft(y, n_fft=n_fft) # shape[0] -> frequencies, shape[1] -> time
print(D.shape)
freqs = librosa.fft_frequencies(sr=sr,n_fft=n_fft)
window_interval = D.shape[1]/sr
datasize_avr = n_fft*D.shape[1]/y.size # 这里的值指的是一个单位的音频数据有多少个字节，因为y是模拟信号转化为数值的结果

# stft数据转化为钢琴按键，并估计音频文件的大调

## 寻找距离tet最近的freqs项，输出到nearest_indices
# 初始化一个空列表来存储最近邻索引
nearest_indices = []

# 对于 tet 中的每个频率值
for freq in tet:
    # 计算 freqs 中距离 freq 最近的索引
    nearest_index = np.argmin(np.abs(freqs - freq))

    # 将最近的索引添加到列表中
    nearest_indices.append(nearest_index)

print(nearest_indices)

piano_D = dict(zip(all_phonic_num,np.abs(D[nearest_indices])))
piano_D_sum=dict(zip(piano_D.keys(),np.array([np.sum(n) for n in piano_D.values()])))

# 分析大调
major_possibilities = {}
from music_theory import create_major_num
for major in music_theory.all_phonic:
    natural_major = create_major_num(major)
    if have_bass:
        natural_major = natural_major[12:] # bass range 0~12
    weight = sum(piano_D_sum.get(n, 0) for n in natural_major)
    major_possibilities[major] = weight

# 打印结果
print("大调权重：", major_possibilities)
max_major = max(major_possibilities, key=major_possibilities.get)
print("最可能的大调：", max_major)


# 计算每个键在每个窗口的 Z 分数
z_scores = {}
for key, values in piano_D.items():
    z_scores[key] = stats.zscore(values)

# 根据 Z 分数的阈值判断每个键在每个窗口中是否存在乐音 default=1.4
threshold = 1.3

piano_presence = {}
for key, scores in z_scores.items():
    presence = [abs(score) if abs(score) > threshold else 0 for score in scores]
    piano_presence[key] = presence
piano_presence = dict(zip(piano_presence.keys(),standardize_array(np.array(list(piano_presence.values())))))

# 将 piano_presence 数据转换为二维数组
presence_array = np.array([presence for presence in piano_presence.values().__reversed__()])

# 音频播放和部分与音频信号有关的准备措施

# 加载音频文件，用于PyAudio播放

wf = wave.open(audio_file,'r')
data = wf.readframes(wf.getnframes())
sampwidth = wf.getsampwidth()
samplerate = wf.getframerate()
nframes = wf.getnframes()
time_sum = nframes/samplerate
wf.close()

datasize = sampwidth*wf.getnchannels() # 一个音频信号单位的字节数（采样宽度*通道数）

p = pyaudio.PyAudio()
# 设置一个可输出音频数据流，规定通道数、采样格式、采样率
stream = p.open(format=p.get_format_from_width(sampwidth),
                channels=wf.getnchannels(),
                rate=samplerate,
                output=True)

# 播放音频的控制函数
def play(a,b):
    audio_clip = data[int(a - (a % datasize)):int(b - (b % datasize))]  # 防止audio_clip的音频信号字节错位产生白噪音
    stream.write(audio_clip)
async def async_play(a,b):
    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor() as pool:
        await loop.run_in_executor(pool, play, a, b)

# 可视化、神经网络预测和弦

# 时间点对应的stft数据，会根据need_to_save变量选择是否保存到硬盘，用于神经网络学习
time_to_stft = {}
# 创建动态更新函数
import deepmusictheory
deepmusictheory.limit = 0.99
warned_list = []
time_split = np.linspace(0, time_sum, D.shape[1])
fig = plt.figure()
async def update(frame,chord:music_theory.Chord):
    global actual,predict_interval
    plt.cla()  # 清除当前图形
    start_window = frame  # 计算当前显示窗口范围的起始窗口索引
    end_window = round(min(D.shape[1], frame+(_sample_length*datasize*4)))  # 计算当前显示窗口范围的结束窗口索引 4是n_fft与hop_length之比，默认
    # 显示当前窗口范围内的 presence_array 数据
    content = np.array([presence[start_window:end_window] for presence in presence_array])
    plt.imshow(content, cmap='binary', aspect='auto')
    audio_name = audio_file.split('\\')[-1]
    fig.canvas.manager.set_window_title(f"{audio_name.replace('.wav', '')} Frame [{frame}] Time [{round(time_split[frame]*100)/100}s]")

    sf = time_split[frame] * samplerate * datasize
    ef = time_split[frame + 1] * samplerate * datasize

    # 上标签
    plt.xlabel(f'Frame {round(sf)}->{round(ef)}')
    plt.ylabel('88 Tones')
    chord_text = "("+str(chord)+")" if warned_list[frame] else str(chord)
    plt.title(chord_text)

    # 关闭 x 轴和 y 轴的刻度显示
    plt.xticks([])
    plt.yticks([])

    plt.ion()
    plt.show()
    plt.pause(0.0001)
    plt.ioff()


chord_list = []
warned_times = 0
valid_times = 0


# 使用神经网络预测和弦
a = tm()
stft_data = np.array([np.array([a[x] for a in piano_D.values()]) for x in range(D.shape[1])])
# 用于网络的stft输入
output_2 = deepmusictheory.model_chord_feature.predict(stft_data)
b = tm()
print("Used",b-a,"seconds to predict chords")
a = tm()
for x in range(D.shape[1]):
    chord_and_warned = deepmusictheory.to_chord(output_2[x],False)
    if chord_and_warned is None:
        chord_list.append(None)
        warned_list.append(False)
        continue

    chord,warned = chord_and_warned
    assert isinstance(chord, music_theory.Chord)
    warned_times += 1 if warned else 0
    warned_list.append(warned)
    chord_list.append(chord)
    valid_times += 1

b = tm()
print("Used",b-a,"seconds to analyse chords")
print("Valid:",valid_times,"/",D.shape[1])
print("Valid Percentage:",round(valid_times/D.shape[1]*100*1000)/1000,"%")
print("Warned:",warned_times,"/",valid_times)
print("Warned Percentage:",round(warned_times/valid_times*100*1000)/1000,"%")

# 可视化显示
async def main(x):
    await asyncio.gather(
        update(x, chord_list[x]),
        async_play(sf, ef),
        play_multiple_notes_async(chord_list[x - 1], (ef - sf) / samplerate / datasize),
    )

if __name__ == '__main__':
    for x in range(D.shape[1]-1):
        sf = time_split[x] * samplerate * datasize
        ef = time_split[x + 1] * samplerate * datasize
        asyncio.run(main(x))
    print("Visualization End.")

stream.stop_stream()
p.terminate()
