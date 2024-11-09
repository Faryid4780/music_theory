import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import music_theory2 as mt
import mido
from time import sleep



class CQT_STFT:
    def __init__(self, audio_filename:str):
        y, sr = librosa.load(audio_filename, sr=None)

        # 通过 onset_envelope 方法计算音频的开始时间
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)

        tempo, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)

        # 1. 计算CQT
        # n_bins是频率通道数，默认是84，表示从C1到B7的音符
        print("CQT...")
        cqt_result = np.abs(librosa.cqt(y, sr=sr, hop_length=512, n_bins=88))

        # # 将CQT结果转换为幅度
        # cqt_db = librosa.amplitude_to_db(np.abs(cqt_result), ref=np.max)

        # # 2. 可视化CQT结果
        # plt.figure(figsize=(12, 6))
        # librosa.display.specshow(cqt_db, sr=sr, x_axis='time', y_axis='cqt_note', cmap='coolwarm')
        # plt.colorbar(format='%+2.0f dB')
        # plt.title('CQT (Constant-Q Transform) Spectrogram')
        # plt.show()

        # 3. 计算STFT (可选)
        # n_fft是窗口大小，hop_length是帧移
        print("STFT...")
        self.hop_length = 512
        stft_result = librosa.stft(y, n_fft=2048, hop_length=self.hop_length)
        stft_db = librosa.amplitude_to_db(np.abs(stft_result), ref=np.max)

        # # 4. 可视化STFT结果
        # plt.figure(figsize=(12, 6))
        # librosa.display.specshow(stft_db, sr=sr, x_axis='time', y_axis='log', cmap='coolwarm')
        # plt.colorbar(format='%+2.0f dB')
        # plt.title('STFT Spectrogram')
        # plt.show()

        self.cqt_db = cqt_result
        self.stft_db = stft_db
        self.beat_frames = beat_frames
        self.bpm = round(tempo)
        self.sr = sr
        self.y = y

    def get_combination(self):
        print(self.cqt_db.shape)
        print(self.stft_db.shape)
    def get_note_pattern(self,limit:float=12):
        types = ('on','off')
        note_messages = [list() for x in range(88)]

        diff_cqt = np.diff(self.cqt_db)
        lim = limit * np.mean(np.abs(diff_cqt))
        for freq_i,freq_row in enumerate(diff_cqt):
            pressed = False
            avr = 0
            for time_i,delta in enumerate(freq_row):
                if abs(delta) > lim:
                    if delta < 0 and not pressed:
                        note_messages[freq_i].append((freq_i,time_i*self.hop_length/self.sr,types[0],self.cqt_db[freq_i][time_i]))
                        pressed = True
                    elif delta > 0 and pressed:
                        note_messages[freq_i].append((freq_i,time_i*self.hop_length/self.sr,types[1],avr))
                        pressed = False
                elif pressed:
                    v = self.cqt_db[freq_i][time_i]
                    avr = v if avr == 0 else (avr+v)/2
            if pressed:
                note_messages[freq_i].pop(-1)
        return note_messages

    def frame_dict(self):
        frame_d = {}
        for x in self.get_note_pattern():
            for y in x:
                if not y[1] in frame_d.keys():
                    frame_d[y[1]] = [y]
                else:
                    frame_d[y[1]].append(y)
        return frame_d


    def get_frames_to_time(self):
        return np.linspace(0,len(self.y)/self.sr,len(self.y))

    def play(self):
        # 创建MIDI输出端口
        out_port = mido.open_output()
        d = self.frame_dict()
        interval = len(self.y)/self.sr/self.cqt_db.shape[1]
        for x in range(self.cqt_db.shape[1]):
            if x in d.keys():
                for tone in d[x]:
                    out_port.send(mido.Message('note_'+tone[2],note=tone[0]+21, velocity=int(4*tone[3])))
            sleep(interval)

    def save_to_midi(self,filename,ticks_per_beat:int=960):
        print("Analyse Data...")
        data = []
        for n in self.get_note_pattern():
            data += n
        data.sort(key=lambda item:item[1])

        print("Save to MIDI...")
        midi_file = mido.MidiFile(ticks_per_beat=ticks_per_beat)

        track = mido.MidiTrack()

        midi_file.tracks.append(track)

        track.append(mido.MetaMessage('set_tempo',tempo=mido.bpm2tempo(self.bpm)))

        pressed = [False for _ in range(88)]
        last_time = [0 for _ in range(88)]

        time2ticks = lambda time:int(time/(60/self.bpm)*ticks_per_beat)

        for note in data: 
            tick = time2ticks(note[1])
            if not pressed[note[0]] and note[2] == 'on': # note_on
                last_time[note[0]] = tick
                track.append(mido.Message('note_on', note=note[0] + 21,
                                          velocity=int(64 * note[3]) if 64 * note[3] < 128 else 127,
                                          time=tick))
                pressed[note[0]] = True
            elif pressed[note[0]] and note[2] == 'off': # note_off
                track.append(mido.Message('note_off', note=note[0] + 21,
                                          velocity=int(64 * note[3]) if 64 * note[3] < 128 else 127,
                                          time=tick-last_time[note[0]]))
                pressed[note[0]] = False
            print(f"Tick {tick} Time {int(note[1]*100)/100} pressed: {[1 if n else 0 for n in pressed]}")

        midi_file.save(filename)












if __name__ == '__main__':
    cs = CQT_STFT("wav\\BGM01P.wav")
    cs.get_combination()
    cs.save_to_midi("output.mid")
    cs.play()
