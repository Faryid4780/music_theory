# music_theory2.py
# encoding: utf-8
import numpy as np
from abc import *
########################

# 大三和弦 + 大三度：1-3-5-7（大七度） --> 大大七和弦
# 小三和弦 + 大三度：1-b3-5-7（大七度） --> 小大七和弦
# 减三和弦 + 大三度：1-b3-b5-b7（小七度） --> 减小七和弦
# 大三和弦 + 小三度：1-3-5-b7（小七度） --> 大小七和弦
# 小三和弦 + 小三度：1-b3-5-b7（小七度） --> 小小七和弦
# 增三和弦 + 小三度：1-3-#5-7（大七度） --> 增大七和弦
# 减三和弦 + 小三度：1-b3-b5-bb7（减七度） --> 减减七和弦
#
# 大小七和弦 --> 属七和弦
# 大大七和弦 --> 大七和弦
# 小小七和弦 --> 小七和弦
# 减小七和弦 --> 半减七和弦
# 减减七和弦 --> 减七和弦
#
# 大七和弦：Cmaj7
# 属七和弦：C7
# 小七和弦：Cm7
# 半减七和弦：Cm7(b5)
# 减七和弦：Cdim7
# 小大七和弦：CmM7
# 增大七和弦：Cmaj7(#5)

MAJOR = 'maj'
MINOR = 'm'
DIMISHED = 'dim'
AUGMENTED = 'aug'
DOMINANT = ''
MMAJOR = 'mmaj'

rome_num = {1:'I',
            2:'II',
            3:'III',
            4:'IV',
            5:'V',
            6:'VI',
            7:'VII'}
num_rome = dict(zip(rome_num.values(),rome_num.keys()))



twelvetone_equal_temperament = 27.5 * np.logspace(0, 7.25, 88, endpoint=True, base=2)
phonic_dict = {'C':3,'Db':4,'D':5,'Eb':6,'E':7,'F':8,'Gb':9,'G':10,'Ab':11,'A':0,'Bb':1,'B':2}
phonic = list(phonic_dict.keys())
all_phonic = phonic[9:12]+(phonic * 8)+[phonic[0]]
all_phonic_num = []
index = 9
for p in all_phonic:
    location_on_piano = int(index/12)
    all_phonic_num.append(p+str(location_on_piano))
    index+=1
full_half_law = (0,2,4,5,7,9,11)
full_2_half_law = full_half_law+(0+12,2+12,4+12,5+12,7+12,9+12,11+12)
natural_chord_law = [MAJOR,MINOR,MINOR,MAJOR,MAJOR,MINOR,DIMISHED]

#与maj的音程偏移量，-1是b，1是#，-2是bb
interval_delta = {MAJOR:(0,0,0,0),MINOR:(0,-1,0,-1), AUGMENTED:(0,0,1,0), DIMISHED:(0,-1,-1,-2), DOMINANT:(0,0,0,-1),
                  MMAJOR:(0,-1,0,0)}


def find(target_list:list, target_element):
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
    def __init__(self, law):
        """
        :param law: 顺阶法则
        :type law: tuple

        自定义调性
        """
        self.law = law

    def get_level_of(self, root, tone) -> int:
        return self.get_instance_on(root).index(tone) + 1

    def get_rome_level_of(self, root, tone) -> str:
        return rome_num.get(self.get_level_of(root, tone))

    def get_instance_on(self, tone) -> tuple:
        all_phonic
        i = phonic.index(tone)
        return tuple([phonic[(i + n) % 12] for n in self.law])
    
    def get_chord(self,root,i:int):
        instance = self.get_instance_on(root)
        tones = [instance[(i-1+l)%len(instance)] for l in (0,2,4,6)]
        return Chord.from_tones(tones,instance[i-1])[0]

    
    @property
    def chord_types(self) -> tuple:
        return tuple([self.get_chord('C',i).type for i in range(1,len(self.law)+1)])
            
    @property
    def intervals(self) -> tuple:
        return tuple([Interval.fromLocation(n,target_var=i) for i,n in enumerate(self.law)])

    @property
    def has_tritone(self) -> bool:
        """
        :return: 属和弦V是否存在三全音

        返回是否属和弦具有三全音，这决定了该调性中属和弦的不稳定性，即决定了Dominant Motion的强度
        """
        i_s = self.intervals
        return i_s[3].location - i_s[6].location == -6

    @property
    def has_aug(self) -> bool:
        """
        :return: 该调性是否具有增音程

        增音程是和声小调的特征音程（唱起来会较为困难）
        """
        return 3 in np.diff(np.array([i.location for i in self.intervals]))


class MajorTonality(Tonality):
    """
    自然大调调性
    """
    def __init__(self):
        super(MajorTonality, self).__init__(full_half_law)


class MinorTonality(Tonality):
    """
    自然小调调性
    """
    def __init__(self):
        super(MinorTonality, self).__init__((0, 2, 3, 5, 7, 8, 10))
        
class HarmonicMinorTonality(Tonality):
    """
    和声小调调性
    \n
    作为一个存在强进行的调性，属和弦V应当存在对主和弦I较强的趋向性\n
    然而，MinorTonality类（自然小调调性）的Vm7不存在Tritone（三全音抑或增四度，Interval为b5）\n
    而Tritone恰恰决定了属和弦V的趋向性\n
    \n
    原因：\n
    导音距离I音和III音（稳定的音，因为出现在I和弦上）仅仅差一个半音，具有强倾向性，因此不稳定\n
    自然大调存在两个导音（以C大调为例）：B,F\n
    自然而然，两个导音构成的音程就会很不稳定，把它叫做三全音或增四度（详情见Interval类）\n
    \n
    因此，自然小调需要进行微调，将Vm7变为V7时，V7的三音和七音会形成三全音\n
    做法是使VII音需要上行半音，于是就形成了和声小调
    """
    def __init__(self):
        super(HarmonicMinorTonality, self).__init__((0, 2, 3, 5, 7, 8, 11))

class MelodicMinorScale(Tonality):
    """
    旋律小调调性

    在和声小调中，V音和VI音差了三个半音，构成一个增音程 (Interval为2)\n
    因此会相对不是很舒服，为了解决这一点，旋律小调将和声小调的V音向上移动半音\n
    这样就会避免增音程，同时V,VI,VII会有大调内味，因此听上去更加明朗
    """
    def __init__(self):
        super(MelodicMinorScale, self).__init__((0, 2, 3, 5, 7, 9, 11))

class Interval:
    def __init__(self, var:int, prefix:int=0, tonality:Tonality=MajorTonality()):
        """
        :param var: 音程
        :param prefix: 前缀，-1表示b，1表示#，0表示无前缀 (default)
        :param tonality: 调性，默认自然大调
        :type var: int
        :type prefix: int
        
        创建一个抽象音程
        """
        self.law = tonality.law+tuple(np.array(tonality.law)+12)

        self.name = ''
        if prefix < 0:
            self.name+="b"*-prefix
        elif prefix > 0:
            self.name+="#"*prefix
        self.name+=str(var)
        self.location=self.law[var-1]+prefix

        self.var = var
        self.delta = prefix

    def get_note(self, root:str, height:int=4):
        """
        :param root: 根音
        :param height: 根音音高
        :return: 根音的该音程对应的音符
        :type root: str
        :type height: int
        
        获得根音的该音程对应的音符
        """
        return all_phonic_num[findapn(root+str(height))+self.location]

    def __eq__(self, other):
        try:
            if self.location == other.location:
                return True
        except AttributeError:
            return False
        return False

    def __str__(self):
        return self.name

    @staticmethod
    def fromRoot(root:str, target:str):
        return Interval.fromLocation(Interval.getLocation(root,target))

    @staticmethod
    def fromLocation(location:int, tonality:Tonality=MajorTonality(), target_var=None):
        law = tonality.law + tuple(np.array(tonality.law) + 12)
        arr_d = np.array(law)-(location%len(law))
        nearest = target_var if target_var else np.argmin(np.abs(arr_d))
        delta = -arr_d[nearest]
        return Interval(nearest+1, delta)

    @staticmethod
    def getLocation(root:str, target:str):
        return (findph(target)-findph(root))%len(phonic)

    @staticmethod
    def sort_method(input_):
        return input_.location


def add_interval(iargs:list):
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

def get_interval(chord_type:str):
    interval_deltas = interval_delta.get(chord_type)
    assert interval_deltas != None
    initial = [Interval(1),Interval(3),Interval(5),Interval(7)]
    locations = []
    for i,x in enumerate(interval_deltas):
        initial[i] = Interval(initial[i].var, x)
        locations.append(initial[i].location)

    return tuple(initial),locations


        

class Chord:
    def __init__(self, root:str, chord_type:str, *args, **kwargs):
        """
        :param root: 根音
        :param chord_type: 和弦类型
        :param args: 七和弦（7），六和弦（6），默认0（三和弦），别的和弦以此类推，对于6/9和弦，args处可填6,9，args不得为空
        :param kwargs: 此处还有加音add:int, 挂音sus:int, 延伸音others:Interval
        :raise AttributeError: args为空时
        :raise Exception: 乐理错误
        用于生成一个和弦
        """

        self.interval = [Interval(1), Interval(3), Interval(5)]
        self.omit_3 = False
        # 不能大于13
        if len(args) == 0:
            args = [0]

        # 组成6和弦，7和弦，9和弦，11和弦，...
        combine=[]
        if len(args) != 0:
            combine=list(args)
        else:
            raise AttributeError("和弦数字类型不得为空")

        self.interval+=add_interval(combine)
        # 纯五和弦去三音
        if 5 in combine:
            self.interval.pop(1)
            self.omit_3 = True

        # 分析add,sus（无omit）
        if kwargs.get("add") != None:
            add = kwargs["add"]
            if add/2 == int(add/2) and add<7:
                kwargs["add"]+=7
            self.interval.append(Interval(kwargs["add"]))
        if kwargs.get("sus") != None and not self.omit_3:
            sus = kwargs["sus"]
            if sus/2 != int(sus/2) and sus>=9:
                kwargs["sus"]-=7
            newi = self.interval[1]
            self.interval[1] = Interval(kwargs["sus"])


        if not chord_type in interval_delta.keys():
            raise Exception("未知的和弦名称"+str(chord_type))

        # 分析和弦类型
        delta_dict = interval_delta.get(chord_type)
        for i,key in enumerate([Interval(1),Interval(3),Interval(5),Interval(7)]):
            t = find(self.interval,key)
            if len(t) == 0:
                continue
            t = t[0]
            interval = self.interval[t]
            self.interval[t] = Interval(interval.var,interval.delta+delta_dict[i])

        # 分析others（和弦标记括号内的内容），与此同时，计算无音高判别的音程列表
        others = kwargs.get("others")
        for i,x in enumerate(self.interval):
            if others != None and isinstance(others, Interval) and others.var==x.var:
                x = Interval(x.var,others.delta+x.delta)
                self.interval[i] = x

        self.pure_tones = [phonic[(findph(root)+x.location)%12] for x in self.interval]
        self.root = root

        # 生成和弦标记
        num_surfix = ""
        for x in combine:
            num_surfix += str(x) if not x in (0,1,3) else ""
            num_surfix += "/"
        num_surfix = num_surfix[:-1]
        other_surfix = ""
        if kwargs.get("sus") != None:
            other_surfix+="sus"+str(kwargs["sus"])
        if kwargs.get("add") != None:
            other_surfix+="add"+str(kwargs["add"])
        if others != None:
            other_surfix+="(%s)"%others.name

        self.surfix = chord_type+num_surfix+other_surfix
        self.name = self.root+self.surfix
        self.type = chord_type

        self.interval.sort(key=Interval.sort_method)

    def name_at_diatonic(self,r,tonality:Tonality=MajorTonality()) -> str:
        return self.level_at_diatonic(r, tonality) + self.surfix

    def level_at_diatonic(self,r,tonality:Tonality=MajorTonality()) -> str:
        n = all_phonic.index(r)
        rn = all_phonic.index(self.root)
        i = Interval.fromLocation(rn - n, tonality)
        print(i)
        x = ('b', '', '#')
        return rome_num.get((i.var)%8+2,'VII') + x[i.delta + 1]

    def is_diatonic_of(self,r,tonality:Tonality) -> bool:
        instance = tonality.get_instance_on(r)
        if not self.pure_tones in instance:
            return False
        return set(tonality.get_chord(r,instance.index(self.root)).pure_tones) == set(self.pure_tones)

    def getChord(self,height:int=4, inversion:int=0):
        if inversion > len(self.interval)-1:
            raise AttributeError(self.name+" chord only support inversions from 0 to "+str(len(self.interval)-1))
        result = []
        root_index = findapn(self.root+str(height))
        assert root_index != -1
        phonic_len = len(phonic)
        for i,x in enumerate(self.interval):
            if i >= inversion and inversion > 0:
                index = root_index+x.location-phonic_len
                if index < 0:
                    raise AttributeError(self.name+" at this height don't support inversion "+str(inversion))
                result.append(all_phonic_num[root_index+x.location-phonic_len])
            else:
                result.append(all_phonic_num[root_index+x.location])

        return result

    def __str__(self):
        return self.name

    @staticmethod
    # 这里的优先级有待微调
    def priority_of(chord_type:str):
        priority = {
            MMAJOR:8,
            MAJOR:10,
            MINOR:10,
            AUGMENTED:3,
            DIMISHED:5,
            DOMINANT:8,
        }
        return priority[chord_type]

    @staticmethod
    def from_tones(tones_input: tuple, root: str = None):
        """
        :param tones_input: 音符输入 
        :param root: 根音（为None自行检测）
        :return: 和弦和误差（多余的音符）
        :type root: str
        :type tones_input: tuple
        
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
                chords.append((probable_chord,appendix))
            alg = lambda tupl : Chord.priority_of(tupl[0].type)-5*len(tupl[1][0])-3*len(tupl[1][1])
            chords.sort(key=alg, reverse=True)
            if 'dimsus4' in chords[0][0].name:
                return chords[1] if len(chords) > 2 else None
            return chords[0] if len(chords) > 1 else None

        interval_set = set([Interval.getLocation(root, tone) for tone in tones_input])

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

        i6 = Interval(6).location
        for x in range(9, 14, 2):
            i = Interval(x).location%12
            if not i in interval_set:
                break
            num = [x]
            interval_set.remove(i)

        if i6 in interval_set:
            num = [6] + num if num != [0] else [6]
            interval_set.remove(i6)

        intervals = [Interval.fromLocation(i) for i in interval_set]

        vars = map(lambda x:x.var, intervals)
        others = None
        for x in (5, 9, 11, 13):
            for y in (-1, 1):
                i = Interval(x, y)
                if i.location in interval_set and not (i.var == 5 and 5 in vars):
                    others = i
                    interval_set.remove(i.location)
                    break

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

        left_tones += list(map(lambda x:x.name,intervals))

        args = ",".join(map(str, num))
        final = eval(f"Chord(root, settled_type, {args}, add=add, sus=sus, others=others)")
        assert isinstance(final, Chord)
        return final, (set(final.pure_tones) - set(tones),left_tones)


class ChordFunc:

    def __init__(self,chord:Chord, in_ph:str, tonality:Tonality):
        self.in_ph = in_ph
        self.chord = chord
        self.level = num_rome.get(chord.level_at_diatonic(in_ph))
        if 'b' in self.level or '#' in self.level or not self.level:
            raise Exception("该和弦离调，不支持判断功能")


def secondary_dominant_tendency(dominant_chord:Chord, at_ph:str, tonality:Tonality=MajorTonality()):
    """
    :param dominant_chord: 属和弦
    :param at_ph: 位于音调
    :param tonality: 调性
    :return: dominant_chord在该调性上对应的解决和弦

    --- 本部分请阅读HarmonicMinorTonality类\n
    在和声小调调性中，为了使得属和弦V具有三全音音程，而将自然小调调性的Vm7改为了V7\n
    ---

    这种方法对于其他的和弦同样有所效果
    例如C大调中的和弦进行示例：Cmaj7 -> C7 -> Fmaj7
    将Cmaj7强行转化为C7再解决，C7就叫做副属和弦（Secondary Dominant）


    解决方向以C大调为例：

    | Diatonic Chord | Dominant Form | Next Chord
    | ---- | ---- | ---- |
    | C,Cmaj7 | C7 | F,Fmaj7 |
    | Dm,Dm7 | D7 | G,G7 |
    | Em,Em7 | E7 | Am,Am7 |   (使用次数多)
    | Am,Am7 | A7 | Dm,Dm7 |
    | Bm(b5),Bm7(b5) | B7 | Em,Em7 |


    设副属和弦root所在Tonality对应音阶表的索引为i，
    则副属和弦的解决方向和弦根音索引为(i+3)%8。即右移3单位
    该原理和G->C一样
    """
    if not dominant_chord.type == DOMINANT:
        raise Exception("Require a dominant chord")

    root_list = tonality.get_instance_on(at_ph)
    return tonality.get_chord(at_ph, (root_list.index(dominant_chord.root)+3)%len(root_list))




if __name__ == '__main__':
    r = Chord.from_tones(("C", "Eb", "G", "Bb"))
    print(r[0],r[1:])

    m_t = HarmonicMinorTonality()
    print(m_t.has_aug)





















