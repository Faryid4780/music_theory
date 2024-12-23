import logging
import pickle
import re
from os import mkdir, remove
from os.path import exists
from time import sleep

import numpy as np
import requests
import yt_dlp as youtube_dl
from bs4 import BeautifulSoup
from pydub import AudioSegment

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options

import music_theory2 as mt
from AudioAnalyticSTFTSave import NewMainProgram as NewALMainProgram
from music_theory2 import Chord, ChordSlash


def fix_string(input_: str):
    """
    :param input_: 原字符串
    :return: 文件名
    """
    replacements = {
        '\\': ';',
        '/': ';',
        '*': ';',
        '<': '[',
        '>': ']',
        '?': ';',
        '"': "'",
        '|': ';',
        ':': ';'
    }

    for old, new in replacements.items():
        input_ = input_.replace(old, new)
    return input_


class TrainingLib:
    """
    训练库
    """

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
        self.title_name = ""

    def add(self, time_: float, chord: Chord):
        """
        :param time_: 时间刻
        :param chord: 插入和弦
        """
        i = np.argmin(np.abs(self.ks - time_))
        # if i in self.iset:
        #     raise Exception("Repeat")
        # self.iset.add(i)
        if not self.ks[i] in self.train_data.keys():
            self.train_data[self.ks[i]] = chord

    def save(self):
        """
        保存训练库
        """
        self.train_pickler.dump(self.train_data)
        self.train_file.close()


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
        return pickle.load(self.f)


proxies = {
    'http': 'http://127.0.0.1:10809',
    'https': 'http://127.0.0.1:10809'
}

YTDL_PATH = "youtube_downloaded"


def _get_chord(root: str, ctype: str, numbers: str, lowest: str = None) -> Chord:
    """3
    :param root: 根音（非music_theory库规范）
    :param ctype: 和弦类型
    :param numbers: 几和弦
    :returns: Chord
    """
    if len(root) == 2 and root[1] == "#":
        root = mt.phonic[(mt.phonic.index(root[0]) + 1) % len(mt.phonic)]
    mt_dict = {
        'maj': mt.MAJOR,
        'min': mt.MINOR,
        'dim': mt.DIMISHED,
        'aug': mt.AUGMENTED,
        '': mt.DOMINANT
    }
    option = None
    if len(ctype) > 3 or not mt_dict.get(ctype):
        option = ctype[-3:]

    lowest = root if not lowest else lowest
    numbers = int(numbers) if numbers else 0

    if not option:
        return ChordSlash(*(root, mt_dict.get(ctype)), numbers, lowest=lowest)
    else:
        if len(ctype) > 3:
            return ChordSlash(*(root, mt_dict.get(ctype[:3])), 0, *{option: numbers, 'lowest': lowest})
        else:
            return ChordSlash(*(root, mt.DOMINANT), 0, *{option: numbers, 'lowest': lowest})

def get_youtube_id_from_api_url(url: str) -> str:
    """
    :param url: 网址
    :return: 获取的youtube id
    """
    # 正则表达式用于匹配YouTube视频ID
    # 使用正则表达式搜索匹配
    match = re.search(r'youtube:([-\w]+)', url)
    return match.group(1)


def to_mt_chord(chord_name: str) -> Chord:
    """
    :param chord_name: 网页捕捉到的和弦名称
    :return: music_theory2的和弦类

    示例：
    "E:maj" -> Chord的Emaj和弦;
    "C#:min" -> Chord的Dbm和弦（mt默认输出降调b,因此需要转化）;
    "D:7" -> Chord的D7和弦;
    """

    # 使用正则表达式分割和弦名称为两部分
    match = re.match(r"([A-Ga-g][#b]?)(.*)", chord_name)
    if match:
        root = match.group(1)  # 根音
        suffix = match.group(2)  # 和弦类型

        # 使用正则表达式分割suffix
        suffix_match = re.search(r"([A-Za-z]+)(\d*)(/[A-Za-z]*)?$", suffix)
        if suffix_match:
            letters = suffix_match.group(1)  # maj
            numbers = suffix_match.group(2)  # 几和弦 (如果存在，否则None)
            lowest = suffix_match.group(3)  # 最低音
            lowest = lowest[1:] if lowest is not None else lowest

            # 返回结果
            return _get_chord(root, letters, numbers, lowest=lowest)
        else:
            suffix_match = re.search(r"(\d*)(/[A-Za-z]*)?$", suffix)
            lowest = suffix_match.group(2)  # 最低音
            lowest = lowest[1:] if lowest is not None else lowest
            # 属n和弦
            return _get_chord(root, '', suffix_match.group(1), lowest=lowest)

    logging.warning("Invalid chord name format: " + chord_name)


windows_versions = (10, 8, 7)


class Main:
    """
    主程序
    """

    def __init__(self, chordify_url: str, proxies=None, stft_sampling_interval=2):
        # self.chrome_path = chromeVersionCheck(proxies)
        # self.chrome_path = None
        self.proxies = proxies
        self.chordify_url = chordify_url
        self.stft_sampling_i = stft_sampling_interval

        # 通过Google Chrome打开Chrodify网址
        # self.driver = None
        self.training_lib = None
        self.wav_path = None
        self.time_to_chord = {}
        self.video_id = ""
        self.title_name = ""
        self.chrome_version = 130
        self.system_xn = "64"
        self.windows_version_i = 0

        # self.actionChains = None

    def __add_version_i(self):
        self.windows_version_i += 1
        self.windows_version_i %= len(windows_versions)

    def get(self):
        # 主部分
        self.time_to_chord = self.get_chordify_requests()
        while not self.wav_path:
            sleep(0.01)
        self.analyze_wav_ano()

    def download_youtube_wav(self):
        """
        :return: 返回wav文件路径
        """
        if not exists(YTDL_PATH):
            mkdir(YTDL_PATH)

        if exists(f'{YTDL_PATH}/{self.video_id}.wav'):
            self.wav_path = YTDL_PATH + '\\%s.wav' % self.video_id
            return

        # 定义下载选项
        ydl_opts = {
            'format': 'bestaudio/best',  # 下载最好的音频格式
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'outtmpl': f'{YTDL_PATH}/{self.video_id}.%(ext)s',  # 输出文件路径和名称模板
        }

        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            ydl.params['proxy'] = self.proxies['http'] if self.proxies else None
            ydl.download(['https://www.youtube.com/watch?v=%s' % self.video_id])

        # 下载后将MP3转换为WAV
        mp3_file = YTDL_PATH + '\\%s.mp3' % self.video_id  # 替换为下载的文件名
        wav_file = YTDL_PATH + '\\%s.wav' % self.video_id  # 要保存的WAV文件名

        # 使用pydub转换
        audio = AudioSegment.from_mp3(mp3_file)
        audio.export(wav_file, format='wav')
        # 移除mp3文件
        remove(mp3_file)

        self.wav_path = wav_file

    def analyze_wav_ano(self):
        """
        Analyse WAV File.
        """
        al = NewALMainProgram(self.wav_path, tuple(self.time_to_chord.keys()))
        tl = TrainingLibImproved("stft_data\\" + self.title_name + ".dat")
        recording = al.record()
        chord_names = self.time_to_chord.values()
        chords = [to_mt_chord(chord) for chord in chord_names]
        tl.write_and_save((recording, chords))

    def __generate_headers(self):
        xn = "64" if self.system_xn == "64" else "86"
        return {
            "User-Agent": f"Mozilla/5.0 (Windows NT {windows_versions[self.windows_version_i]}.0; Win{self.system_xn}; x{xn}) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{self.chrome_version}.0.0.0 Safari/537.36",
            "Referer": self.chordify_url,
            "Accept-Language": "en-US,en;q=0.9",
        }
    def __next_headers(self):
        if self.chrome_version >= 97:
            self.chrome_version -= 1
        else:
            self.chrome_version = 128
            self.__add_version_i()

    def get_chordify_requests(self) -> dict:
        """
        使用requests库在不打开Chrome的情况下爬取数据
        """

        result = {}
        headers = self.__generate_headers()
        chordify = requests.get(self.chordify_url, proxies=proxies, headers=headers)

        options = Options()
        if self.proxies:
            options.add_argument(f"--proxy-server={self.proxies['https']}")

        service = Service(executable_path="chromedriver_130_0_6723_92.exe")

        c = webdriver.Chrome(service=service, options=options)
        if not 200 <= chordify.status_code <= 299:
            print("HTTP request failed, status code: " + str(chordify.status_code) + '\n' +
                            chordify.text)
            print("Use Selenium instead")


            c.get(self.chordify_url)
            chordify = c.page_source

        chordify = chordify.text if not isinstance(chordify, str) else chordify

        soup = BeautifulSoup(chordify, 'html.parser')
        link_tag = soup.find('link', {'rel': 'preload', 'type': 'application/json'})
        self.title_name = fix_string(soup.title.string)

        if link_tag:
            json_url = link_tag['href']
            if not json_url.startswith('http'):  # 如果href是相对路径，补全为绝对路径
                json_url = 'https://chordify.net' + json_url

            self.video_id = get_youtube_id_from_api_url(json_url)
            self.download_youtube_wav()

            print(json_url)

            self.__next_headers()
            headers = self.__generate_headers()

            json_response = requests.get(json_url, proxies=proxies, headers=headers)

            try:
                json_data = json_response.json()  # 使用requests直接解析为Python字典
            except:
                print("Use Selenium to get JSON instead")
                c.get(json_url)
                import json
                json_data = json.JSONDecoder().decode(BeautifulSoup(c.page_source, 'html.parser').find('pre').text)

        else:
            raise Exception("未找到指定的<link>标签。可能是网址有误或者该网址的和弦不支持视频同步")

        c.close()
        assert isinstance(json_data, dict)

        print("[Information]")
        info = json_data.get("chordInfo")
        if not isinstance(info, dict):
            raise Exception("错误的json文本")

        print("BPM:", info.get("derivedBpm"))
        print("Tonality:", info.get("derivedKey"))

        chords = str(json_data.get("chords")).split("\n")
        last_chord = None
        st = None
        et = None
        for chord in chords:
            chord_data = chord.split(";")
            if len(chord_data) != 4:
                continue
            _, chord_type, start_time, end_time = chord_data

            result[(float(start_time), float(end_time))] = chord_type.split('/')[0]  # 忽略转位

        return result


urls = [
    # "https://chordify.net/chords/snail-s-house-songs/hot-milk-chords",
    # "https://chordify.net/chords/snail-s-house-songs/lullaby-chords",
    # "https://chordify.net/chords/dian-nao-mian-mian-mao-tao-qinchinoi-original-tao-qinchinoi-momone-chinoi",
    # "https://chordify.net/chords/tao-qinchinoi-songs/meryimeryironri-chords",
    # "https://chordify.net/chords/tao-yuan-xiangde-jiu-jiwo-chinoi-pepoyo-chinoi",
    # "https://chordify.net/chords/xing-dao-dian-jing-chinoi-chinoi",
    # "https://chordify.net/chords/tao-qinchinoi-songs/momotanfanku-chords",
    # "https://chordify.net/chords/momonechinoi-yin-tou-tao-qinchinoi-original-tao-qinchinoi-momone-chinoi",
    # "https://chordify.net/chords/okusuri-yinnde-qinyou-chu-yinmiku-mochiutsune-mochiutsune",
    # "https://chordify.net/chords/liu-xingrinoaisu-chu-yinmiku-mochiutsune-mochiutsune",
    # "https://chordify.net/chords/feng-lu-rurupurofairu-chu-yinmiku-mochiutsune-mochiutsune",
    # "https://chordify.net/chords/men-fanmooran-chu-yinmiku-mochiutsune-mochiutsune",
    # "https://chordify.net/chords/go-jiao-shikudasai-chu-yinmiku-mochiutsune-mochiutsune",
    # "https://chordify.net/chords/xiagarugaru-ge-aiyuki-mochiutsune-mochiutsune",
    # "https://chordify.net/chords/chu-yinmiku-pinku-orijinaru-maretu",
    # "https://chordify.net/chords/chu-yinmiku-namida-orijinaru-maretu",
    # "https://chordify.net/chords/chu-yinmiku-iyaiyayo-orijinaru-maretu",
    # "https://chordify.net/chords/chu-yinmiku-aishiteitanoni-orijinaru-maretu",
    # "https://chordify.net/chords/ajoura-the-scp-foundation-main-theme-ajoura",
    # "https://chordify.net/chords/c418-songs/sweden-chords",
    # "https://chordify.net/chords/c418-songs/subwoofer-lullaby-chords",
    # "https://chordify.net/chords/c418-songs/moog-city-2-chords",
    # "https://chordify.net/chords/c418-songs/beginning-2-chords",
    # "https://chordify.net/chords/c418-songs/concrete-halls-chords",
    # "https://chordify.net/chords/c418-songs/warmth-chords",
    # "https://chordify.net/chords/c418-songs/wet-hands-chords",
    # "https://chordify.net/chords/cover-bu-an-si-fanshi-intanetto-zong-jiao-c-cyber-milk-p-thaibeatz-narumiya",
    # "https://chordify.net/chords/shinikariti-uta-qin-ye-kui-dian-jing",
    # "https://chordify.net/chords/yoakeland-dian-jing-arrangement-for-piano-drums-fiveninesquared",
    # "https://chordify.net/chords/quanteanatano-suo-weidesu",
    # "https://chordify.net/chords/quanteanatano-suo-weidesu-2",
    # "https://chordify.net/chords/quanteanatano-suo-weidesu-3",
    # "https://chordify.net/chords/k-quanteanatano-suo-weidesu",
    # "https://chordify.net/chords/bach-fantasia-and-fugue-in-g-minor-bwv-542-doeselaar-netherlands-bach-society-netherlands-bach-socie",
    # "https://chordify.net/chords/fa-guo-yin-le-jia-ai-li-ke-sa-di-luo-ti-ge-wu-ji-nuo-pei-di-gymnopedie-1-3-ericalfred-leslie-satie-classical-tunes",
    # "https://chordify.net/chords/rachmaninoff-prelude-in-g-minor-op-23-no-5-kassia",
    # "https://chordify.net/chords/bach-fantasia-and-fugue-in-g-minor-bwv-542-doeselaar-netherlands-bach-society-netherlands-bach-socie",
    # "https://chordify.net/chords/sayonara-you-ling-wu-ren-jie-zhi-zuo-xuan-minekomanma-dian-jing",
    # "https://chordify.net/chords/xue-xiaono-qi-bu-ke-si-yi-tao-qinchinoi-original-tao-qinchinoi-momone-chinoi",
    # "https://chordify.net/chords/zael-songs/moonlight-sonata-beethoven-cover-chords",
    # "https://chordify.net/chords/debussy-clair-de-lune-rousseau"
]

if __name__ == '__main__':
    for url in urls:
        while True:
            try:
                main = Main(url, proxies, 1)
                main.get()
                break
            except Exception as e:
                print("Request Error:", e)
        # try:
        #     main.get()
        # except Exception as e:
        #     print(e)
        #     print("Errored Url:", url)
        #     print("Continue")
        #     continue
