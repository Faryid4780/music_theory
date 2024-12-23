"""
music_theory
"""
# music_theory2.py
# encoding: utf-8
import numpy as np
import warnings
import re

MAJOR = 'maj'
MINOR = 'm'
DIMISHED = 'dim'
AUGMENTED = 'aug'
DOMINANT = ''
MMAJOR = 'mmaj'

degree_num = {1: 'I',
              2: 'II',
              3: 'III',
              4: 'IV',
              5: 'V',
              6: 'VI',
              7: 'VII'}
num_degree = dict(zip(degree_num.values(), degree_num.keys()))

twelvetone_equal_temperament = 27.5 * np.logspace(0, 7.25, 88, endpoint=True, base=2)
phonic_dict = {'C': 3, 'Db': 4, 'D': 5, 'Eb': 6, 'E': 7, 'F': 8, 'Gb': 9, 'G': 10, 'Ab': 11, 'A': 0, 'Bb': 1, 'B': 2}
phonic = list(phonic_dict.keys())
all_phonic = phonic[9:12] + (phonic * 8) + [phonic[0]]
all_phonic_num = []
index = 9
for p in all_phonic:
    location_on_piano = int(index / 12)
    all_phonic_num.append(p + str(location_on_piano))
    index += 1

basic_scale = (0, 2, 4, 5, 7, 9, 11)

# line---------1  2  3  4  5  6  7
ionian_mode = basic_scale
dorian_mode = 0, 2, 3, 5, 7, 9, 10
phygian_mode = 0, 1, 3, 5, 7, 8, 10
lydian_mode = 0, 2, 4, 6, 7, 9, 11
mixolydian_mode = 0, 2, 4, 5, 7, 9, 10
aeolian_mode = 0, 2, 3, 5, 7, 8, 10
locrian_mode = 0, 1, 3, 5, 6, 8, 10
harmonic_minor_mode = 0, 2, 3, 5, 7, 8, 11
melodic_minor_mode = 0, 2, 3, 5, 7, 9, 11

mode_names = {
    ionian_mode: "MajorScale",
    dorian_mode: "DorianScale",
    phygian_mode: "PhygianScale",
    lydian_mode: "LydianScale",
    mixolydian_mode: "MixolydianScale",
    aeolian_mode: "MinorScale",
    locrian_mode: "LocrianScale",
    harmonic_minor_mode: "HarmonicMinorScale",
    melodic_minor_mode: "MelodicMinorScale"
}

# 与maj的音程偏移量，-1是b，1是#，-2是bb
interval_delta = {
    MAJOR: (0, 0, 0, 0),
    DOMINANT: (0, 0, 0, -1),
    MINOR: (0, -1, 0, -1),
    DIMISHED: (0, -1, -1, -2),
    MMAJOR: (0, -1, 0, 0),
    AUGMENTED: (0, 0, 1, 0)
}


def find(target_list: list, target_element):
    result = []
    for i in range(len(target_list)):
        if target_list[i] == target_element:
            result.append(i)
    return result


def findapn(n):
    return all_phonic_num.index(n)


def findph(n):
    return phonic.index(n)


class Tonality:
    def __init__(self, mode):
        """
        :param mode: 顺阶规则
        :type mode: tuple

        可以通过规定的模式来创建不同调式，当然，也可以自定义

        示例:
        ------
        >>> import music_theory2 as mt
        >>> tonality = mt.Tonality(mt.aeolian_mode)
        >>> tonality.chord_types
        ('m', 'm', 'maj', 'm', 'm', 'maj', '')

        你也可以使用预先准备的调式类

        >>> tonality2 = mt.MinorScale()
        >>> print(tonality2.mode)
        (0, 2, 3, 5, 7, 8, 10)
        >>> print(tonality.mode)
        (0, 2, 3, 5, 7, 8, 10)

        """
        self.mode = mode

    def get_level_of(self, root, tone) -> int:
        return self.get_instance_on(root).index(tone) + 1

    def get_rome_level_of(self, root, tone) -> str:
        return degree_num.get(self.get_level_of(root, tone))

    def get_instance_on(self, tone) -> tuple:
        i = phonic.index(tone)
        return tuple([phonic[(i + n) % 12] for n in self.mode])

    def get_chord(self, i: int, with_7=True, root: str = None, instance: tuple = None):
        if root and not instance:
            instance = self.get_instance_on(root)
        elif not instance and not root:
            raise Exception("必须填写root和instance参数的其中一个")

        locations = (0, 2, 4, 6) if with_7 else (0, 2, 4)
        tones = [instance[(i + l) % len(instance)] for l in locations]
        return Chord.from_tones(tones, instance[i])[0]

    @property
    def chord_types(self) -> tuple:
        return tuple([self.get_chord(i, root='C').type for i in range(len(self.mode))])

    @property
    def intervals(self) -> tuple:
        return tuple([Interval.from_location(n, target_var=i) for i, n in enumerate(self.mode)])

    @property
    def has_tritone(self) -> bool:
        """
        :return: 属和弦V是否存在三全音

        返回是否属和弦具有三全音，这决定了该调式中属和弦的不稳定性，即决定了Dominant Motion的强度
        """
        i_s = self.intervals
        return i_s[3].location - i_s[6].location == -6

    @property
    def has_aug(self) -> bool:
        """
        :return: 该调式是否具有增音程

        增音程是和声小调的特征音程（唱起来会较为困难）
        """
        return 3 in np.diff(np.array([i.location for i in self.intervals]))

    @property
    def name(self):
        """
        :return: 调式名称
        """
        class_name = type(self).__name__
        if class_name != Tonality.__name__:
            return class_name
        return mode_names.get(self.mode, class_name)


class MajorScale(Tonality):
    """
    自然大调调式
    """

    def __init__(self):
        super(MajorScale, self).__init__(ionian_mode)  # (0, 2, 4, 5, 7, 9, 11)


class MinorScale(Tonality):
    """
    自然小调调式
    """

    def __init__(self):
        super(MinorScale, self).__init__(aeolian_mode)


class HarmonicMinorScale(Tonality):
    """
    和声小调调式
    """

    def __init__(self):
        super(HarmonicMinorScale, self).__init__(harmonic_minor_mode)


class MelodicMinorScale(Tonality):
    """
    旋律小调调式
    """

    def __init__(self):
        super(MelodicMinorScale, self).__init__(melodic_minor_mode)


class Interval:
    def __init__(self, var: int, prefix: int = 0, tonality: Tonality = MajorScale()):
        """
        :param var: 音程
        :param prefix: 前缀，-2为bb, -1表示b，1表示#，0表示无前缀 (default)
        :param tonality: 调式，默认自然大调
        :type var: int
        :type prefix: int
        
        创建一个抽象音程
        """
        self.law = tonality.mode + tuple(np.array(tonality.mode) + 12)

        self.name = ''
        if prefix < 0:
            self.name += "b" * -prefix
        elif prefix > 0:
            self.name += "#" * prefix
        self.name += str(var)
        self.location = self.law[var - 1] + prefix

        self.var = var
        self.delta = prefix

    def get_note(self, root: str, height: int = 4):
        """
        :param root: 根音
        :param height: 根音音高
        :return: 根音的该音程对应的音符
        :type root: str
        :type height: int
        
        获得根音的该音程对应的音符
        """
        return all_phonic_num[findapn(root + str(height)) + self.location]

    def get_pure_tone(self, root: str):
        """
        :param root: 根音
        :return: 该音程下对应的音符

        获得根音的该音程对应的音符，无音高
        """
        return phonic[(findph(root) + self.location) % 12]

    def __eq__(self, other):
        try:
            if self.location == other.location:
                return True
        except AttributeError:
            return False
        return False

    def __str__(self):
        return self.name

    def __add__(self, other):
        if not isinstance(other, Interval):
            raise TypeError(f"Unsupported operation '+' between Interval and {type(other)}")
        return Interval.from_location(other.location + self.location)

    def __neg__(self):
        return Interval.from_location((-self.location) % 12)

    def __sub__(self, other):
        if not isinstance(other, Interval):
            raise TypeError(f"Unsupported operation '-' between Interval and {type(other)}")
        return self + (-other)

    @staticmethod
    def from_string(name: str):
        """
        :param name: 音程名
        :return: Interval类

        识别音程名称，返回音程类
        """
        result = re.match('(#|b*)(\d+)$', name)
        prefix = result.group(1)
        delta = 0
        for n in prefix:
            if n == '#':
                delta += 1
            elif n == 'b':
                delta -= 1
        return Interval(int(result.group(2)), delta)

    @staticmethod
    def from_root(root: str, target: str):
        return Interval.from_location(Interval.get_location(root, target))

    @staticmethod
    def from_location(location: int, tonality: Tonality = MajorScale(), target_var=None):
        law = tonality.mode + tuple(np.array(tonality.mode) + 12)
        arr_d = np.array(law) - (location % len(law))
        nearest = target_var if target_var else np.argmin(np.abs(arr_d))
        delta = -arr_d[nearest]
        return Interval(nearest + 1, delta)

    @staticmethod
    def get_location(root: str, target: str):
        return (findph(target) - findph(root)) % len(phonic)


def add_interval(iargs: list):
    if 0 in iargs:
        return []
    intervals = []
    for x in iargs:
        if x > 6:
            for addnote in range(7, x + 1, 2):
                intervals.append(Interval(addnote))
        elif x == 6:
            intervals.append(Interval(x))
    return intervals


def get_interval(chord_type: str):
    interval_deltas = interval_delta.get(chord_type)
    assert interval_deltas is not None
    initial = [Interval(1), Interval(3), Interval(5), Interval(7)]
    locations = []
    for i, x in enumerate(interval_deltas):
        initial[i] = Interval(initial[i].var, x)
        locations.append(initial[i].location)

    return tuple(initial), locations


i5_circle = [Interval.from_location((Interval(5).location * l) % 12) for l in range(12)]


class Chord:
    def __init__(self, root: str, chord_type: str = MAJOR, *args, **kwargs):
        """
        :param root: 根音
        :param chord_type: 和弦类型
        :param args: 七和弦（7），六和弦（6），默认0（三和弦），别的和弦以此类推，对于6/9和弦，args处可填6,9

        :key add: 加音符
        :key sus: 挂和弦
        :key others: 延伸音(Tension)，例如b5,b9,#9,#11,b13，可以是Interval组成的元组，也可以是单个Interval

        :raise AttributeError: args为空时
        :raise Exception: 乐理错误

        用于生成一个和弦
        """

        self.interval = [Interval(1), Interval(3), Interval(5)]
        self.omit_3 = 5 in args
        self.args = args
        self.kwargs = kwargs
        # max(args)<=13
        if len(args) == 0:
            args = [0]

        self.interval += add_interval(args)
        # omit_3 if 5
        if self.omit_3:
            self.interval.pop(1)

        # Fix add,sus
        self.has_sus = False
        if kwargs.get("add") is not None:
            add = kwargs["add"]
            if add / 2 == int(add / 2) and add < 7:
                kwargs["add"] += 7
            self.interval.append(Interval(kwargs["add"]))
        if kwargs.get("sus") is not None and not self.omit_3:
            sus = kwargs["sus"]
            if sus / 2 != int(sus / 2) and sus >= 9:
                kwargs["sus"] -= 7
                self.has_sus = True
            self.interval[1] = Interval(kwargs["sus"])

        if not chord_type in interval_delta.keys():
            raise Exception("未知的和弦名称" + str(chord_type))

        # Chord Tones
        delta_dict = interval_delta.get(chord_type)
        for i, key in enumerate([Interval(1), Interval(3), Interval(5), Interval(7)]):
            t = find(self.interval, key)
            if len(t) == 0:
                continue
            t = t[0]
            interval = self.interval[t]
            self.interval[t] = Interval(interval.var, interval.delta + delta_dict[i])

        # Tensions & Altered Tensions
        others = kwargs.get("others", ())

        def replace_others(intervals: list, others_: tuple):
            """
            替换或添加延伸音
            """
            int_vars = [i.var for i in intervals]
            avoid_locations = [i.location + 1 for i in intervals]

            for o in others_:
                end = False
                for i, var in enumerate(int_vars):
                    if var == o.var:
                        intervals[i] = o
                        end = True
                        break
                if end:
                    continue

                if o.location in avoid_locations:
                    warnings.warn(f"存在避免音：{Interval.from_location(o.location - 1)}"
                                  f"与延伸音{others_}构成小九度音程，可能会使和弦不和谐")

                intervals.append(o)

        if isinstance(others, Interval):
            replace_others(self.interval, (others,))
        elif others is not None:
            assert all(isinstance(i, Interval) for i in others), \
                "All elements must be Interval in parameter 'others'"
            replace_others(self.interval, others)

        self.root = root

        # 生成和弦标记
        num_surfix = ""
        for x in args:
            num_surfix += str(x) if not x in (0, 1, 3) else ""
            num_surfix += "/"
        num_surfix = num_surfix[:-1]
        other_surfix = ""
        if kwargs.get("sus") is not None:
            other_surfix += "sus" + str(kwargs["sus"])
        if kwargs.get("add") is not None:
            other_surfix += "add" + str(kwargs["add"])
        if others is not None:
            if isinstance(others, Interval):
                other_surfix += "(%s)" % others.name
            elif len(others) > 0:
                other_surfix += "(%s)" % ','.join(map(str, others))

        chord_type_surfix = chord_type

        if not self.has_7 and chord_type_surfix in (MAJOR, DOMINANT):
            chord_type_surfix = ''
            self.chord_type = MAJOR

        self.surfix = chord_type_surfix + num_surfix + other_surfix
        self.name = self.root + self.surfix
        self.type = chord_type

        self.interval.sort(key=lambda t: t.location)

    def __repr__(self):
        return f"""{type(self).__name__}({self.name}, {self.type}, {', '.join(self.args)}, kwargs={self.kwargs})"""

    @property
    def pure_tones(self):
        return [x.get_pure_tone(self.root) for x in self.interval]

    def name_at_tonic(self, r, tonality: Tonality = MajorScale()) -> str:
        """
        :param r: 音调
        :param tonality: 调式
        :return: 返回Degree Name （级数名）
        """
        return self.level_at_tonic(r, tonality) + self.surfix

    def level_at_tonic(self, r, tonality: Tonality = MajorScale()) -> str:
        n = findph(r)
        rn = findph(self.root)
        i = Interval.from_location((rn - n) % 12, tonality)
        x = ('b', '', '#')
        return degree_num.get((i.var - 1) % 7 + 1) + x[i.delta + 1]

    def is_tonic_of(self, r, tonality: Tonality) -> bool:
        instance = tonality.get_instance_on(r)
        if not self.pure_tones in instance:
            return False
        return set(tonality.get_chord(instance.index(self.root), root=r).pure_tones) == set(self.pure_tones)

    def getChord(self, height: int = 4, inversion: int = 0):
        if inversion > len(self.interval) - 1:
            raise AttributeError(self.name + " chord only support inversions from 0 to " + str(len(self.interval) - 1))
        result = []
        root_index = findapn(self.root + str(height))
        assert root_index != -1
        phonic_len = len(phonic)
        for i, x in enumerate(self.interval):
            if i >= inversion > 0:
                index = root_index + x.location - phonic_len
                if index < 0:
                    raise AttributeError(self.name + " at this height don't support inversion " + str(inversion))
                result.append(all_phonic_num[root_index + x.location - phonic_len])
            else:
                result.append(all_phonic_num[root_index + x.location])

        return result

    def __str__(self):
        return self.name

    def __eq__(self, other):
        if not isinstance(other, Chord):
            raise TypeError(f"不支持Chord类与{type(other)}的相等比较")
        return set(self.pure_tones) == set(other.pure_tones)

    @property
    def has_7(self):
        return 7 in [i.var for i in self.interval]

    @property
    def lowest(self):
        """
        :return: 返回该和弦的最低音
        """
        return self.root

    @property
    def tritone_intervals(self):
        """
        :return: 返回该和弦具备的三全音对应根音音程
        """

        # 并非(not)所有(all)的音对应的三全音都不(not)在pure_tones元组里(in)
        existing_tritones = []
        for i, tone_inter in enumerate(self.interval):
            to_t = Interval(5, -1).get_pure_tone(tone_inter.get_pure_tone(self.root))
            for to_tone_inter in self.interval[i + 1:]:
                if to_tone_inter.get_pure_tone(self.root) == to_t:
                    existing_tritones.append((tone_inter, to_tone_inter))
        return existing_tritones

    @property
    def tritones(self):
        """
        :return: 返回该和弦具备的三全音
        """
        return list(map(lambda item: (item[0].get_pure_tone(self.root),
                                      item[1].get_pure_tone(self.root)),
                        self.tritone_intervals))

    def roughly_equals(self, other):
        """
        比较两个和弦是否大致相等 (意味着不考虑add,sus,others,6音)

        该运算不满足交换律，即``a.roughly_equals(b)``不代表``b.roughly_equals(a)``
        """
        if not isinstance(other, Chord):
            raise TypeError(f"不支持Chord类与{type(other)}的相等比较")

        if self.root != other.root:
            return False

        def get_chord_tones(chord: Chord):
            chord_intervals = []
            for i in chord.interval:
                if i.var in (1, 3, 5, 7):
                    chord_intervals.append(i.location)
            return set(chord_intervals)

        return get_chord_tones(self).issubset(get_chord_tones(other))

    @staticmethod
    # 这里的优先级有待微调
    def priority_of(chord_type: str, args: tuple):
        priority = {
            MMAJOR: 8,
            MAJOR: 10,
            MINOR: 10,
            AUGMENTED: 3,
            DIMISHED: 4,
            DOMINANT: 8,
        }
        return priority[chord_type] - 2.5 * (6 in args)

    @staticmethod
    def from_name(name: str):
        """
        :param name: 和弦名
        :return: Chord类
        """
        primary = re.match('([bA-Z]+)(maj|M|△|m|°|min|dim|aug)?(\d*)', name)
        if not primary:
            raise Exception(f"非法和弦名: {name}")
        root = primary.group(1)
        kwargs = {'others': []}
        chord_type = primary.group(2)
        arg = primary.group(3) if primary.group(3) else 0
        if chord_type is None:
            chord_type = DOMINANT
        elif chord_type in ('M', '△'):
            chord_type = MAJOR
        elif chord_type == 'min':
            chord_type = MINOR
        elif chord_type == '°':
            chord_type = MINOR
            kwargs['others'].append(Interval(5, -1))

        surfix = name[primary.span()[1]:]
        for item in re.findall('([a-z]+\d+)', surfix):
            result = re.match('([a-z]+)(\d+)', item)
            kwargs[result.group(1)] = int(result.group(2))

        others_surfix = surfix[re.match('([a-z]+\d+)*', surfix).span()[1]:]
        try:
            others_result = re.match('\(([^)]+)\)', others_surfix).group(1).split(',')
            for n in others_result:
                kwargs['others'].append(Interval.from_string(n))
        except AttributeError:
            pass

        try:
            return Chord(root, chord_type, int(arg), **kwargs)
        except IndexError as e:
            raise Exception(f"非法音程: {e}")

    @staticmethod
    def from_tones(tones_input, root: str = None):
        """
        :param tones_input: 音符输入，类型iter
        :param root: 根音（为None自行检测）
        :return: 和弦和误差（多余的音符）
        :type root: str
        
        将若干个具有特定音程的音符转化为对应的和弦
        """
        if tones_input.__len__() < 2:
            return

        tones = list(tones_input)
        tones.sort(key=findph)  # 如果排序是必要的
        if root is None:
            chords = []
            for tone in tones:
                probable_chord, appendix = Chord.from_tones(tones_input, tone)
                chords.append((probable_chord, appendix))
            alg = lambda tupl: Chord.priority_of(tupl[0].type, tupl[0].args) - 5 * len(tupl[1][0]) - 3 * len(tupl[1][1])
            chords.sort(key=alg, reverse=True)
            if 'dimsus4' in chords[0][0].name:
                return chords[1] if len(chords) > 2 else None
            return chords[0] if len(chords) > 1 else None

        interval_set = set([Interval.get_location(root, tone) for tone in tones_input])

        maximum = -1
        settled_type = None
        for ctype in interval_delta.keys():
            oset = set(get_interval(ctype)[1])
            similarity = len(interval_set & oset)
            if similarity > maximum:
                settled_type = ctype
                maximum = similarity

                if similarity >= len(tones):
                    break
        # 根据相似度生成初始基本和弦
        primary = Chord(root, settled_type)

        initial_interval_set = interval_set.copy()
        # 去除已经包含在内的音程
        interval_set -= set([x.location for x in primary.interval])

        num = [0]
        # 存在7音?
        i7 = get_interval(settled_type)[1][3]
        if i7 in interval_set:
            num = [7]
            interval_set.remove(i7)
            for x in range(9, 14, 2):
                i = Interval(x).location % 12
                if not i in interval_set:
                    break
                num = [x]
                interval_set.remove(i)

        i6 = Interval(6).location
        if i6 in interval_set:
            num = [6] + num if num != [0] else [6]
            interval_set.remove(i6)

        intervals = [Interval.from_location(i) for i in interval_set]

        vars = map(lambda x: x.var, intervals)

        others = ()
        for x in (13, 5, 9, 11):
            for y in (-1, 0, 1):
                if (x, y) in ((11, -1), (13, 1)):  # b11和#13严格上不允许存在
                    continue

                i = Interval(x, y)
                if i.location % 12 in interval_set and not (i.var == 5 and 5 in vars):
                    others += (i,)
                    interval_set.remove(i.location % 12)

        sus = None
        add = None
        left_tones = []
        # 如果3音不存在于原来的音程集合里，可以允许sus
        give_sus = not Interval(3).location in initial_interval_set

        for x in (2, 4, 6, 9, 11, 13):
            i = Interval(x).location
            if i in interval_set:
                if give_sus:
                    sus = x
                    interval_set.remove(i)
                    continue

                if not add:
                    add = x
                else:
                    left_tones.append(Interval(x).name)
                interval_set.remove(i)

        intervals = [Interval.from_location(i) for i in interval_set]
        left_tones += list(map(lambda x: x.name, intervals))

        final = Chord(*(root, settled_type, *num), add=add, sus=sus, others=others)
        return final, (set(final.pure_tones) - set(tones), left_tones)


def secondary_dominant_tendency(dominant_chord: Chord, tonic: str, tonality: Tonality = MajorScale()):
    """
    :param dominant_chord: 属和弦
    :param tonic: 位于音调
    :param tonality: 调式
    :return: dominant_chord在该调式上对应的解决和弦

    背景
    ------

    本部分请阅读HarmonicMinorTonality类

    在和声小调调式中，为了使得属和弦V具有三全音音程，而将自然小调调式的Vm7改为了V7

    ------

    这种方法对于其他的和弦同样有所效果——
    例如C大调中的和弦进行示例：Cmaj7 -> C7 -> Fmaj7。
    将Cmaj7强行转化为C7再解决，C7就叫做副属和弦（Secondary Dominant）

    示例
    ------

    解决方向以C大调为例：

    | Diatonic Chord | Dominant Form | Next Chord
    | ---- | ---- | ---- |
    | C,Cmaj7 | C7 | F,Fmaj7 |
    | Dm,Dm7 | D7 | G,G7 |
    | Em,Em7 | E7 | Am,Am7 |   (使用次数多)
    | Am,Am7 | A7 | Dm,Dm7 |
    | Bm(b5),Bm7(b5) | B7 | Em,Em7 |

    副属和弦根音 朝着 解决方向和弦根音 的音程是Interval(4)，即纯四度音程（强进行）
    该原理和G->C一样
    注意IV或IVmaj7的情况可能不太一样，解决后和弦并不在调内
    """
    if dominant_chord.type != DOMINANT:
        raise Exception("Require a dominant chord")
    root_list = tonality.get_instance_on(tonic)
    if not dominant_chord.root in root_list:
        raise Exception("Root tone of this dominant chord should be in tonality")

    next_root = Interval(4).get_pure_tone(dominant_chord.root)
    return tonality.get_chord(root_list.index(next_root), root=tonic, with_7=dominant_chord.has_7)


class ChordSlash(Chord):
    def __init__(self, root: str, chord_type: str = MAJOR, *args, **kwargs):
        """
        :key lowest: 最低音，例如F(maj)/G中，最低音是G，不填默认为根音

        带转位（最低音）的和弦
        """
        super().__init__(root, chord_type, *args, **kwargs)
        self.__lowest = kwargs.get('lowest', self.root)

    @property
    def is_normal_chord(self):
        return self.__lowest in super(ChordSlash, self).pure_tones

    def __str__(self):
        s = super(ChordSlash, self).__str__()
        return s if self.root == self.__lowest else s + '/' + self.__lowest

    @property
    def pure_tones(self):
        a = super(ChordSlash, self).pure_tones
        if self.__lowest != self.root:
            a.insert(0, self.__lowest)
        return a

    @property
    def lowest(self):
        """
        :return: 返回该和弦的最低音
        """
        return self.__lowest

    def getChord(self, height: int = 4, inversion: int = 0):
        """
        :param height: 音高
        :param inversion: 无效
        :return: 带音高组成音
        """
        if self.is_normal_chord:
            return super(ChordSlash, self).getChord(height, super(ChordSlash, self).pure_tones.index(self.__lowest))

        return super(ChordSlash, self).getChord(height) + [self.__lowest + str(height - 1)]


chord_func = ('T', 'M', 'SD', 'D')
tonality_priorities = [
    MajorScale(),
    MinorScale(),
    HarmonicMinorScale(),
    Tonality(mixolydian_mode),
    MelodicMinorScale(),
    Tonality(dorian_mode),
    Tonality(lydian_mode),
    Tonality(phygian_mode),
    Tonality(locrian_mode),
]


def get_func_strength(tonic: str, chord: Chord, tonality: Tonality = MajorScale()):
    """
    :param tonic: 调
    :param chord: 和弦
    :param tonality: 调式
    :return: 返回该调内该和弦的功能和强度，对应chord_func的索引：T为主，M为中（可主可属），SD为下属，D为属，-1为未知
    """
    if chord.tritone_intervals:
        return 3  # D

    try:
        level = tonality.mode.index(Interval.get_location(tonic, chord.lowest)) + 1
    except ValueError:
        return -1  # Unknown

    if level in (2, 4):
        return 2  # SD
    elif level in (5, 7):
        return 3  # D
    elif level in (6, 1):
        return 0  # T
    elif level == 3:
        return 1  # M
    else:
        return -1  # Unknown


class ChordProgressionNode:
    """和弦进行节点"""

    def __init__(self, chord: Chord = None, next=None, bpm=None, duration=2,
                 tonic: str = 'C', tonality: Tonality = MajorScale()):
        self.chord = chord
        self.next = next
        self.tonic = tonic
        self.tonality = tonality
        self.duration = duration
        self.independent_bpm = bpm

    @property
    def is_dominant_motion(self) -> bool:
        """
        :return: 该节点是否为属进行
        """
        if not self.is_complete_motion:
            return False

        return Interval(4).get_pure_tone(self.chord.lowest) == self.next.chord.lowest

    @property
    def is_complete_motion(self) -> bool:
        """
        :return: 该节点是否完整拥有本节点的和弦和下一个节点的和弦
        """
        return self.next is not None and self.chord is not None and self.next.chord is not None

    def adapt_to_tonality(self):
        """
        调整ChordProgressionNode.tonality(调式)使其迎合在tonic上的该和弦
        该方法常用于判断借用和弦
        """
        if not self.chord:
            return

        # 如果为属和弦且根音在调内，先判断不是附属和弦再判断借用其它调式和弦
        if self.chord.root in self.tonality.get_instance_on(self.tonic) and \
                self.chord.type == DOMINANT and self.has_resolved_tritones:
            return

        for tonality in tonality_priorities:
            instance = tonality.get_instance_on(self.tonic)
            try:
                i = instance.index(self.chord.root)
                diatonic_chord = tonality.get_chord(i, instance=instance, with_7=self.chord.has_7)
                if self.chord.roughly_equals(diatonic_chord):
                    self.tonality = tonality
                    break
            except ValueError:
                continue

    @property
    def chord_func(self):
        """和弦在该tonic上的功能，返回三元素元组，依次为：[变化量, 本和弦功能, 下一和弦功能]"""
        if not self.is_complete_motion:
            return 0, '/', '/'

        this = get_func_strength(self.tonic, self.chord, self.tonality)
        if self.is_dominant_motion and self.has_resolved_tritones:  # 附属和弦直接判定为下一和弦为T
            nex = 0
        else:
            nex = get_func_strength(self.tonic, self.next.chord, self.next.tonality)

        if this == -1:
            if nex == -1:
                return 0, '/', '/'
            return 0, '/', chord_func.__getitem__(nex)
        elif nex == -1:
            return 0, chord_func.__getitem__(this), '/'

        return nex - this, chord_func.__getitem__(this), chord_func.__getitem__(nex)

    def get_dominant_data(self, inter: int = 1):
        """
        属进行信息
        """
        tends = []
        tends_i = []

        if not self.is_complete_motion:
            return 0, 0, 0, tends
        assert isinstance(self.next.chord, Chord)

        level = 3 if self.is_dominant_motion else 0
        up = 0
        down = 0

        for sptl in self.chord.interval:
            for optl in self.next.chord.interval:
                spt = sptl.get_pure_tone(self.chord.root)
                opt = optl.get_pure_tone(self.next.chord.root)

                if (spt, opt) in tends:
                    continue

                if 0 < Interval.get_location(spt, opt) <= inter:
                    tends.append((spt, opt))
                    tends_i.append((sptl, optl))
                    level += 1  # 与主和弦相邻半音那么加一
                    up += 1
                    if opt == self.next.chord.root:
                        level += 0.5  # 导音再加0.5
                elif 12 > Interval.get_location(spt, opt) >= 12 - inter:
                    tends.append((spt, opt))
                    tends_i.append((sptl, optl))
                    level += 1  # 与主和弦相邻半音那么加一
                    down += 1
                    if opt == self.next.chord.root:
                        level += 0.5  # 导音再加0.5

        return level, up, down, tends, tends_i

    @property
    def tendency_level(self):
        """
        :return: 作为属功能的和弦对于下一（主）和弦的倾向性强度，0为无，如果为主和弦转属和弦，仍然返回正数，根据chord_func分析

        规则
        ------

        若存在强进行，level+=3 (D->T)

        若存在与主和弦相邻半音的音数，level+=数量

        其中若存在导音额外+0.5

        则对于 Fmaj7，Fmaj7/G, G, Db7, G7 五种过渡到Cmaj就会是这样：

        顺序：强进行+相邻个数+导音个数*0.5

        Fmaj7 -> 0+1+0=1

        Fmaj7/G -> 3+1+0=4

        G -> 3+1+0.5=4.5

        Db7 -> 0+4+0.5*2=5.0

        G7 -> 3+2+0.5=5.5
        """
        return self.get_dominant_data()[0]

    @property
    def resolved_tritones(self):
        """
        :return: 下一和弦解决了的三全音，
        """
        # 预计算 tones 数据，只计算一次
        tones = self.get_dominant_data()[3]
        resolved_tritones = []

        # 提前返回 False，避免不必要的遍历
        tritones = self.chord.tritones
        if not tritones:
            return resolved_tritones
        for tritone1, tritone2 in tritones:
            next_tones = self._find_resolved_tritones(tones, tritone1, tritone2)

            # 若没有找到两个音符解决三全音，继续
            if len(next_tones) < 2:
                continue

            # 检查音程是否为 4 或 8 半音
            if self._is_valid_tritone_resolution(next_tones[0], next_tones[1]):
                resolved_tritones.append(((tritone1, tritone2), next_tones))

        return resolved_tritones

    @property
    def has_resolved_tritones(self):
        """
        是否所有三全音都被解决
        """
        return set(map(lambda x: x[0], self.resolved_tritones)).issuperset(self.chord.tritones)

    @staticmethod
    def _find_resolved_tritones(tones, tritone1, tritone2):
        """检查并返回三全音对的解决音符"""
        next_tones = []
        for first_tone, next_tone in tones:
            # 如果音符是三全音对之一，记录解消后的音符
            if first_tone in (tritone1, tritone2):
                next_tones.append(next_tone)
        return tuple(next_tones)

    @staticmethod
    def _is_valid_tritone_resolution(tone1, tone2):
        """检查两个音符是否符合三全音解消的要求（大三度或小三度）"""
        interval = Interval.get_location(tone1, tone2)
        # 解消要求：上行一个、下行一个，或反向解消，结果必须为 4 或 8 半音
        return interval in (4, 8)

    def time(self, bpm):
        """
        :param bpm: BPM
        :return: Playing Duration, It returns a costant value when the node has an independent bpm
        """
        if self.independent_bpm:
            return self.duration * 60 / self.independent_bpm
        else:
            return self.duration * 60 / bpm


class ChordProgressionGroup:
    """和弦组，采用单向链表"""

    def __init__(self, initial_tonic: str = 'C',
                 bpm: int = 120,
                 beat_per_column: int = 4,
                 initial_tonality: Tonality = MajorScale()):
        self.head = ChordProgressionNode(tonic=initial_tonic, tonality=initial_tonality)  # 头节点
        self.tonic = initial_tonic
        self.tonality = initial_tonality
        self.normal_chord_types = self.tonality.chord_types
        self.bpm = bpm
        self.beat_per_column = beat_per_column

    def create(self, l, order):
        if order:
            self.create_tail(l)
        else:
            self.create_head(l)

    def create_tail(self, l):
        """尾插法"""
        for item in l:
            self.append(item)

    def create_head(self, l):
        """头插法"""
        for item in l:
            self.insert(0, item)

    def clear(self):
        """清空和弦进行组"""
        self.head = ChordProgressionNode(tonic=self.tonic, tonality=self.tonality)

    def is_empty(self):
        """判断该组是否为空"""
        return self.head.next is None

    @property
    def length(self):
        """返回组的长度"""
        p = self.head.next
        length = 0
        while p is not None:
            p = p.next
            length += 1
        return length

    def get_node(self, i):
        """读取和弦进行组的第i个和弦节点"""
        p = self.head.next
        j = 0
        while j < i and p is not None:
            p = p.next  # p 从 list[j]到list[j+1]
            j += 1  # j同步
        # 此时j==i，因此j索引就是p的位置，否则为非法位置
        if j > i or p is None:
            raise IndexError(f"该和弦组不存在第{j}个和弦")
        return p

    def get(self, i):
        """读取和弦进行组的第i个和弦"""
        return self.get_node(i).chord

    def insert(self, i, x: Chord, tonic=None, tonality: Tonality = None, beats=1):
        """插入和弦x作为组中第i个和弦"""
        p = self.head
        j = -1
        while j < i - 1 and p is not None:
            p = p.next  # p 从 list[j]到list[j+1]
            j += 1  # j同步
        if i < j or p is None:
            raise IndexError("插入位置非法")
        # 此时j==i-1

        tonality = self.tonality if not tonality else tonality
        tonic = self.tonic if not tonic else tonic
        if p.next is None:
            p.next = ChordProgressionNode(x, duration=beats, tonic=tonic, tonality=tonality)
        else:
            p.next = ChordProgressionNode(x, p.next, duration=beats, tonic=tonic, tonality=tonality)

    def insert_group(self, i: int, group):
        """将另一串链表插入"""
        assert isinstance(group, ChordProgressionGroup), "插入对象必须是 ChordProgressionGroup"
        if group.is_empty():
            return  # 空链表无需插入

        # 找到第 i-1 个节点的位置，若 i = 0 则返回 head
        node = self.get_node(i - 1) if i > 0 else self.head

        # 保存原链表在 i 位置后的部分
        following_nodes = node.next

        # 插入 group 的链表
        node.next = group.head.next

        # 找到插入组的尾部节点
        while node.next is not None:
            node = node.next

        # 将原链表的后续部分连接到插入组尾部
        node.next = following_nodes

    def insert_list(self, i, l):
        """将一串列表插入"""
        for item in l:
            self.insert(i, item)
            i += 1

    def remove(self, i) -> Chord:
        """删除第i个元素"""
        p = self.head
        j = -1
        while j < i - 1 and p is not None:
            p = p.next
            j += 1

        if i - 1 < j or p.next is None:
            raise IndexError("删除位置不合法")

        a = p.next.chord
        p.next = p.next.next

        return a

    def append(self, x: Chord, tonic=None, tonality: Tonality = None, beats=1):
        """在末尾添加一个和弦"""
        p = self.head
        while p.next is not None:
            p = p.next
        p.next = ChordProgressionNode(x, duration=beats, tonic=tonic if tonic else self.tonic,
                                      tonality=tonality if tonality else self.tonality)

    def index(self, x: Chord):
        """返回和弦 x 首次出现的位序号"""
        p = self.head.next
        j = 0
        while p is not None:
            if x == p.chord:
                return j
            p = p.next
            j += 1
        raise ValueError(f"无法从组中找到 {x} 和弦")

    def display(self):
        """输出和弦组中各个数据元素的值"""
        p = self.head.next
        while p is not None:
            print(p.chord, end=' ')
            p = p.next
        print()

    def insert_dim_passing(self, i: int, with_7: bool = False, change_beats=True):
        """
        将在i处的和弦和i+1处的和弦之间添加一个经过减和弦 (Passing Diminsh)

        若不可以添加，则返回IndexError或Exception
        """
        node = self.get_node(i)
        if node.next is None:
            raise IndexError("i+1位置必须存在和弦")

        assert Interval.get_location(node.chord.root, node.next.chord.root) == 2, "两个和弦根音必须相隔1个全音"

        dim_root = Interval(2, -1).get_pure_tone(node.chord.root)
        if with_7:
            if node.chord.type != MAJOR:
                dim = Chord(dim_root, MINOR, 7, others=(Interval(5, -1)))
            else:
                dim = Chord(dim_root, DIMISHED, 7)
        else:
            dim = Chord(dim_root, DIMISHED)

        if change_beats:
            node.duration /= 2
            self.insert(i + 1, dim, beats=node.duration)
        else:
            self.insert(i + 1, dim)

    def replace_with_tritone_substitution(self, i):
        """将组内一个属和弦替换成其对应的里和弦"""
        node = self.get_node(i)
        node.chord = get_tritone_substitution(node.chord)

    def find_cliche(self, start_i: int = 0):
        """
        :param start_i: 起始点
        :return: 套句所在索引[0]和链表[1]
        """
        start_node = self.get_node(start_i)
        cliche_list = []
        interval = None
        length = 0

        l = 0

        while start_node.next is not None:
            this = start_node.chord.lowest
            next_ = start_node.next.chord.lowest
            i = Interval.get_location(this, next_)

            if not interval and i in (1, 11):
                cliche_list.append(start_node.chord)
                interval = i
                length += 1
            elif interval is not None and i == interval:
                cliche_list.append(start_node.chord)
                length += 1
            elif interval is not None and i != interval:
                if length < 3:
                    cliche_list.clear()
                    length = 0
                    interval = None
                else:
                    cliche_list.append(start_node.chord)
                    break

            start_node = start_node.next
            l += 1

        if start_node.next is None:
            return

        return l - length, tuple(cliche_list)

    def find_pedal_point(self, start_i: int = 0):
        """
        :param start_i: 起始寻找位置
        :return: 最近的持续低音索引[0]和列表[1]
        """
        start_node = self.get_node(start_i)
        pedal_list = []
        root = None
        length = 0

        l = 0

        while start_node.next is not None:
            this = start_node.chord.lowest

            if not root:
                pedal_list.append(start_node.chord)
                root = this
                length += 1
            elif root and this == root:
                pedal_list.append(start_node.chord)
                length += 1
            elif root and this != root:
                if length < 3:
                    pedal_list.clear()
                    length = 0
                else:
                    pedal_list.append(start_node.chord)
                    break

            start_node = start_node.next
            l += 1

        if start_node.next is None:
            return

        return l - length, tuple(pedal_list)

    def find_251_motion(self, start_i: int = 0):
        """
        :param start_i: 起始寻找位置
        :return: 251和弦进行的开始索引和I和弦的根音以及调式
        """

        start_node = self.get_node(start_i)

        loc = 0

        while start_node.is_complete_motion and start_node.next.is_complete_motion:
            this = start_node.chord  # 2
            next_ = start_node.next.chord  # 5
            next_next = start_node.next.next.chord  # 1

            tonality_guess = (MajorScale(), MinorScale(), Tonality(dorian_mode))

            if next_.has_7 and next_.type == DOMINANT:
                for tonality in tonality_guess:
                    cts = tonality.chord_types
                    ins = tonality.get_instance_on(next_next.root)

                    if ins[2 - 1] == this.root and ins[5 - 1] == next_.root \
                            and cts[2 - 1] == this.type and cts[5 - 1] == next_.type:
                        return loc, (this, next_, next_next), tonality

            start_node = start_node.next
            loc += 1

        return

    def change_tonic(self, tonic: str, start: int = 0, end: int = None, change_chord: bool = True,
                     change_chord_type: bool = True, tonality: Tonality = None):
        """
        :param tonic: 修改后的调根音
        :param start: 起始索引，默认0
        :param end: 结束索引，默认结尾
        :param change_chord: 是否和弦随其改变，默认True
        :param change_chord_type: 是否和弦类型随其改变，默认True
        :param tonality: 修改后的调式，若为None则不变

        改变一段进行的调根音
        """

        start_node = self.get_node(start)
        end = end if end else self.length
        types = tonality.chord_types if tonality else self.tonality.chord_types

        while start_node is not None and start < end:
            ori_tonic = start_node.tonic
            start_node.tonic = tonic

            if change_chord and start_node.chord is not None:
                i = Interval.from_root(ori_tonic, start_node.chord.root)
                ctype = types[i.var - 1] if change_chord_type else start_node.chord.type

                if isinstance(start_node.chord, ChordSlash):
                    start_node.chord = ChordSlash(i.get_pure_tone(tonic),
                                                  ctype,
                                                  *start_node.chord.args,
                                                  **start_node.chord.kwargs,
                                                  lowest=start_node.chord.lowest)
                else:
                    start_node.chord = Chord(i.get_pure_tone(tonic),
                                             ctype,
                                             *start_node.chord.args,
                                             **start_node.chord.kwargs)

            start_node = start_node.next
            start += 1

    def call_chain(self, func: callable, start: int = 0, end: int = None, **kwargs):
        """
        :param func: 要对每个节点进行的方法，任何一个func开头必须包含两个参数，node:ChordProgressionNode和当前索引index:int
        :param start: 起始位置
        :param end: 末尾位置
        :param kwargs: func的其它参数，尾随在node,index之后
        """
        start_node = self.get_node(start)
        end = end if end else self.length

        while start_node is not None and start < end:
            func(start_node, start, **kwargs)
            start_node = start_node.next
            start += 1

    def change_bpm(self, bpm: int, start: int = 0, end: int = None):
        def setbpm(cpnode: ChordProgressionNode, _, bpm):
            cpnode.independent_bpm = bpm

        self.call_chain(setbpm, start, end, bpm=bpm)

    def adapt_group_tonality(self):
        """
        组内每一个和弦都调用adapt_to_tonality()
        """
        p = self.head
        while p.next is not None:
            p = p.next
            p.adapt_to_tonality()


class Motion251Group(ChordProgressionGroup):
    """
    创建一个251进行，分大调251与小调251，
    前大调后小调，请使用Tonality(dorian_mode)
    """

    def __init__(self, as_1, tonality: Tonality = MajorScale(), with_7=False):
        super().__init__()
        self.create_tail(
            (tonality.get_chord(2 - 1, root=as_1, with_7=with_7),
             Chord(tonality.get_instance_on(as_1)[5 - 1], DOMINANT, 7),
             tonality.get_chord(1 - 1, root=as_1, with_7=with_7))
        )

    @staticmethod
    def get_next_51(chord: Chord, tonality: Tonality = MajorScale(), with_7=False) -> ChordProgressionGroup:
        g = ChordProgressionGroup()
        t2 = Interval(4).get_pure_tone(chord.root)
        g.create_tail(
            (Chord(t2, DOMINANT, 7),
             tonality.get_chord(0, with_7=with_7, root=Interval(4).get_pure_tone(t2)))
        )
        return g


class ChordIterator:
    def __init__(self, group: ChordProgressionGroup):
        self.current_node = group.head

    def __next__(self):
        self.current_node = self.current_node.next
        return self.current_node.chord

    def next_node(self):
        self.current_node = self.current_node.next
        return self.current_node

    @property
    def has_next(self):
        return self.current_node.next is not None


class SecondaryDominantGroup(ChordProgressionGroup):
    """
    创建一个原和弦转附属和弦再解决的进行
    """

    def __init__(self, ori: Chord, at_tone, tonality=MajorScale()):
        super().__init__()
        dom = Chord(ori.root, DOMINANT, 7, kwargs=ori.kwargs)
        solve = secondary_dominant_tendency(dom, at_tone, tonality)

        self.create_tail((
            ori,
            dom,
            solve
        ))


def get_tritone_substitution(dominant_chord: Chord):
    """获取一个属和弦对应的三全音代理，根音互相在五度圈中成180度"""
    if dominant_chord.type != DOMINANT:
        raise Exception("非属和弦不存在对应的里和弦")
    return Chord(Interval(5, -1).get_pure_tone(dominant_chord.root), DOMINANT, 7)


if __name__ == '__main__':
    a = Chord.from_tones(('C', 'E', 'G', 'B'))
    print(a[0].name)

    cpg = ChordProgressionGroup()
    cpg.create_tail((
        Chord("C", MAJOR, 7),
        Chord("F", MAJOR, 7),
        Chord("G", MAJOR, 7),
        Chord("A", MINOR, 7)
    ))

    cpg.insert_dim_passing(1, True)
    cpg.insert_dim_passing(3, True)
    cpg.display()

    g251 = Motion251Group('G', Tonality(dorian_mode))
    g251.display()
    cpg.insert_group(cpg.length, Motion251Group.get_next_51(cpg.get(cpg.length - 1), with_7=True,
                                                            tonality=Tonality(dorian_mode)))
    cpg.remove(cpg.length - 1)
    cpg.display()

    print(get_tritone_substitution(Chord("G", DOMINANT, 7)))

    candidates = (
        (
            Chord('F'),
            Chord('F', MAJOR, 7),
            Chord('D', MINOR),
            Chord('D', MINOR, 7),
            Chord('F', MINOR),
            Chord('F', MINOR, 7),
            Chord('F', MMAJOR, 7),
            Chord('F', MINOR, 6),
            Chord('D', DIMISHED),
            Chord('D', MINOR, 7, others=Interval(5, -1)),
            Chord('Ab'),
            Chord('Ab', MAJOR, 7),
            Chord('Bb', DOMINANT, 7),
            Chord('Db'),
            Chord('Db', MAJOR, 7),
            ChordSlash('F', lowest='G'),
            ChordSlash('F', MAJOR, 7, lowest='G'),
            ChordSlash('D', MINOR, lowest='G'),
            ChordSlash('D', MINOR, 7, lowest='G'),
            Chord('G'),
            Chord('Db', DOMINANT, 7),
            Chord('G', DOMINANT, 7)
        )
    )

    for c in candidates:
        node = ChordProgressionNode(c, ChordProgressionNode(Chord('C')))
        print(f"{node.chord} -> {node.next.chord}: {node.tendency_level} {node.get_dominant_data()[3]}")

    cliche = ChordProgressionGroup()
    cliche.create_tail((
        Chord('C'),
        Chord('C', AUGMENTED),
        Chord('C', DOMINANT, 6),
        Chord('C', DOMINANT, 7)
    ))

    print()
    print("Cliche:")
    node = cliche.head
    while node.next is not None:
        node = node.next
        if node.next:
            print(f"{node.chord} -> {node.next.chord}: {node.get_dominant_data()}")

    cliche = ChordProgressionGroup()
    cliche.create_tail((
        Chord('C', sus=4),
        Chord('C'),
        Chord('C', sus=2),
        Chord('C')
    ))

    print()
    print("Cliche2:")
    node = cliche.head
    while node.next is not None:
        node = node.next
        if node.next:
            print(f"{node.chord} -> {node.next.chord}: {node.get_dominant_data(2)}")

    cliche = ChordProgressionGroup()
    cliche.create_tail((
        Chord('A', MINOR),
        Chord('A', MMAJOR, 7),
        Chord('A', MINOR, 7),
        Chord('A', MINOR, 6)
    ))

    print()
    print("Cliche3:")
    node = cliche.head
    while node.next is not None:
        node = node.next
        if node.next:
            print(f"{node.chord} -> {node.next.chord}: {node.get_dominant_data()}")

    cliche = ChordProgressionGroup()
    cliche.create_tail((
        Chord('C'),
        ChordSlash('G', lowest='B'),
        ChordSlash('G', MINOR, lowest='Bb'),
        ChordSlash('F', lowest='A'),
        ChordSlash('F', MINOR, lowest='Ab'),
        ChordSlash('C', lowest='G'),
        ChordSlash('Gb', MINOR, 7, others=Interval(5, -1)),
        Chord('F', MAJOR, 7)
    ))

    print()
    print("Cliche4:")
    node = cliche.head
    while node.next is not None:
        node = node.next
        if node.next:
            print(f"{node.chord} -> {node.next.chord}: {node.get_dominant_data()}")

    cliche = ChordProgressionGroup()
    cliche.create_tail((
        Chord('C'),
        Chord('Db', DIMISHED),
        Chord('D', MINOR),
        Chord('Eb', DIMISHED)
    ))

    print("Cliche5:")
    node = cliche.head
    while node.next is not None:
        node = node.next
        if node.next:
            print(f"{node.chord} -> {node.next.chord}: {node.get_dominant_data()}")

    print()
    print()

    group = ChordProgressionGroup()
    # group.create_tail((
    #     Chord('G', DOMINANT, 7, others=Interval(9, -1)),
    #     Chord('C', MAJOR)
    # ))

    group.create_tail((
        Chord('E', DOMINANT, 7, others=Interval(9, 1)),
        Chord('E', DOMINANT, 7, others=Interval(9, -1)),
        Chord('A', MINOR)
    ))

    snode = group.head.next
    assert isinstance(snode, ChordProgressionNode)
    print(snode.chord, '->', snode.next.chord)
    print(snode.is_dominant_motion)
    print(snode.get_dominant_data())
    print(snode.chord_func)

    print(Chord('G', DOMINANT, 7).tritones)
    print()

    node = ChordProgressionNode(Chord('Db', DOMINANT, 7), next=ChordProgressionNode(Chord('C', MAJOR, 7)))
    node.adapt_to_tonality()
    print(node.tonality.name)
    print(Chord.from_name('Csus4add13(b9)').name)
