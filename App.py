from tensorflow_probability import distributions as td, layers as tl
from datetime import time, timedelta
from statistics import mean, median
from essentia import standard as es
from spotipy import Spotify, oauth2
from gdown import download_folder
from sclib import SoundcloudAPI
from pytubefix import YouTube
from tensorflow import keras
from base64 import b64encode
from requests import get
from pickle import load
import tensorflow as tf
import streamlit as st
import librosa
import numpy
import math
import re
import os

st.set_page_config('EgGMAn', ':musical_note:', 'wide')
st.sidebar.link_button('Contact Us', 'https://forms.gle/A4vWuEAp4pPEY4sf9', use_container_width=True)

if not os.path.exists('data'):
    download_folder('https://drive.google.com/drive/folders/1AWUnFrzD8N-2bRyfm8m1oQNleQG_lKVZ')

class Conv1(keras.Model):
    def __init__(self, channel, kernel, stride, padding):
        super(Conv1, self).__init__()
        self.cv = keras.layers.Conv2D(channel, kernel, stride, padding)
        self.bn = keras.layers.BatchNormalization()

    def call(self, x):
        return tf.nn.relu(self.cv(self.bn(x)))

class Conv2(keras.Model):
    def __init__(self, channel, kernel, stride, padding):
        super(Conv2, self).__init__()
        self.cv1 = Conv1(channel, kernel, stride, padding)
        self.cv2 = Conv1(channel, kernel, stride, padding)

    def call(self, x):
        return self.cv2(self.cv1(x))

class Conv5(keras.Model):
    def __init__(self, channel, kernel, stride, padding):
        super(Conv5, self).__init__()
        self.cv1 = Conv1(channel[0], (kernel[0], 1), (stride[0], 1), padding[0])
        self.cv2 = Conv2(channel[0], (1, kernel[1]), (1, stride[1]), padding[1])
        self.cv3 = Conv2(channel[1], (1, kernel[1]), (1, stride[1]), padding[1])

    def call(self, x=0, y=0):
        return self.cv3(x), self.cv2(tf.nn.relu(self.cv1(x) + y))

class Encoder(keras.Model):
    def __init__(self, c_m, c_n):
        super(Encoder, self).__init__()
        self.cv1 = keras.layers.Conv2D(c_n, (1, 1), activation='relu')
        self.cv2 = keras.layers.Conv2D(c_m, (bin, 1), activation='relu')
        self.cv3 = Conv5((c_m, c_n), (bin, 8), (1, 4), ('valid', 'same'))
        self.cv4 = Conv5((c_m, c_n), (bin, 8), (1, 4), ('valid', 'same'))
        self.fc1 = keras.layers.Dense(c_m)
        self.fc2 = keras.layers.Dense(c_m)

    def call(self, x):
        x, y = self.cv3(self.cv1(x))
        x, y = self.cv4(x, y)
        y = tf.nn.relu(self.cv2(x) + y)
        y = tf.reshape(y, (-1, y.shape[-1]))
        y = self.fc2(tf.nn.relu(self.fc1(y)))
        return y

@st.cache_resource(max_entries=1)
def model(n):
    m = Encoder(512, 32)
    m(tf.zeros((1, bin, seq, 1)))
    m.set_weights(load(open(n, 'rb')))
    return m

@st.cache_data(max_entries=1)
def table(n):
    t = load(open(n, 'rb'))
    a = 200 / (numpy.stack(t['vec']).max(0) - numpy.stack(t['vec']).min(0))
    b = a * numpy.stack(t['vec']).min(0) + 100
    return t, a, b

@st.cache_data(ttl='9m')
def music(u):
    if u:
        try:
            if 'youtube' in u:
                YouTube(u).streams.get_by_itag(251).download(filename='music.mp3')
            elif 'soundcloud' in u:
                SoundcloudAPI().resolve(u).write_mp3_to(open('music.mp3', 'wb+'))
            elif 'spotify' in u:
                open('music.mp3', 'wb').write(get(f'{sp.track(re.sub("intl-.*?/", "", u))["preview_url"]}.mp3').content)
            else:
                open('music.mp3', 'wb').write(get(u).content)
            s = f'data:audio/mp3;base64,{b64encode(open("music.mp3", "rb").read()).decode()}'
            st.markdown(f'<audio src="{s}" controlslist="nodownload" controls></audio>', True)
            return librosa.load('music.mp3', sr=sr, duration=30)[0]
        except:
            st.error(f'Error: Unable to access {u}')
    return numpy.empty(0)

def scene(c, s):
    with c:
        st.header(s)
        u = st.multiselect(f'State of {s}', ['オープニング', 'エンディング', 'タイトル', 'イベント', 'チュートリアル', 'ゲーム失敗', 'ゲーム成功', 'ハイライト', 'ワールドマップ', 'フィールド', 'ダンジョン', 'ステージ', 'ショップ', 'メニュー', '選択画面', '休憩ポイント'], placeholder='Opening, Dungeon, etc')
        t = st.multiselect(f'Time of {s}', ['春', '夏', '秋', '冬', '朝', '昼', '夜', '明方', '夕方', '休日', '原始', '古代', '中世', '近代', '現代', '未来'], placeholder='Spring, Morning, etc')
        w = st.multiselect(f'Weather of {s}', ['虹', '星', '晴れ', '曇り', '霧', '砂', '雪', '雷', '雨', '小雨', '大雨', '突風', '混沌', 'オーロラ'], placeholder='Sunny, Cloudy, etc')
        b = st.multiselect(f'Biome of {s}', ['異次元', '虚無', '宇宙', '大陸', '海', '空', '東', '西', '南', '北', '氷', '炎', '花', '毒', '沼', '湖', '泉', '滝', '川', '島', '岩', '崖', '山岳', '峡谷', '洞窟', '温泉', '水中', '水辺', '岸辺', '浜辺', '砂漠', '荒野', '草原', '森林', 'サバンナ', 'ジャングル'], placeholder='Sea, Sky, etc')
        p = st.multiselect(f'Place of {s}', ['アジト', 'オフィス', 'カジノ', 'カフェ', 'キャバクラ', 'ジム', 'タワー', 'ナイトクラブ', 'ビル', 'プール', 'リゾート', 'レストラン', 'ロビー', '遺跡', '飲み屋', '駅', '仮想現実', '家', '外国', '街', '学校', '宮殿', '競技場', '教会', '橋', '軍事基地', '劇場', '研究機関', '公園', '工場', '港', '行政機関', '裁判所', '寺', '式場', '宿泊施設', '城塞', '神社', '神殿', '船', '村', '地下道', '庭', '邸宅', '鉄道', '店', '田舎', '都会', '動物園', '道路', '廃屋', '博物館', '畑', '飛行機', '病院', '墓', '牧場', '遊園地', '路地裏', '牢獄'], placeholder='Office, Casino, etc')
        q = st.multiselect(f'Person of {s}', ['ゆるキャラ', 'ヒーロー', 'ヒロイン', 'スパイ', 'ライバル', 'ラスボス', 'ボス', 'モブ', '大衆', '貴族', '偉人', '仲間', '孤独', '平穏', '不穏', '敵'], placeholder='Hero, Rival, etc')
        a = st.multiselect(f'Action of {s}', ['戦う', '泳ぐ', '走る', '飛ぶ', '会話', '回想', '休憩', '出会う', '別れる', '勝利', '敗北', '探検', '特訓', '謎解き', '買い物', '恋愛'], placeholder='Battle, Run, etc')
        st.subheader(f'Mood of {s}')
        l, r = st.columns(2, gap='medium')
        with l:
            v = st.slider(f'Valence of {s}', -1.0, 1.0, (-1.0, 1.0))
        with r:
            z = st.slider(f'Arousal of {s}', -1.0, 1.0, (-1.0, 1.0))
    return T['scn'].map(lambda i: set(u + t + w + b + p + q + a).issubset(i)) & T['pn'].between(v[0], v[1]) & T['ap'].between(z[0], z[1])

def idx(a, v):
    i = numpy.searchsorted(a, v)
    return mean(a[i-1:i+1]) if 0 < i < len(a) else v

def mel(y):
    return librosa.feature.melspectrogram(y=y, sr=sr, hop_length=sr//fps, n_mels=bin)

def stft(y):
    return librosa.magphase(librosa.stft(y=y, hop_length=sr//fps, n_fft=2*bin-2))[0]

def mold(y, b, p=-1e-99):
    y = stft(y[idx(b, 10 * sr):idx(b, 20 * sr)])
    y = numpy.pad(y, ((0, 0), (0, max(0, seq - y.shape[1]))), constant_values=p)
    return y[None, :, :seq, None]

def rand(l, s):
    return numpy.random.normal(l, s)

def vec(y, s):
    t, b = librosa.beat.beat_track(y=y, sr=sr, units='samples')
    u, v = numpy.split(M.predict(mold(y, b))[0], 2)
    k, m, f = es.KeyExtractor(sampleRate=sr)(y)
    p, c = es.PitchMelodia(sampleRate=sr)(y)
    a = ['C', 'G', 'D', 'A', 'E', 'B', 'F#', 'C#', 'Ab', 'Eb', 'Bb', 'F'].index(k) * math.pi / 6
    return numpy.r_[es.Loudness()(y), median(p[mean(c) < c]), t, f if 'a' in s else -f, f * math.cos(a), f * math.sin(a), rand(u, s * tf.math.softplus(v))]

sp = Spotify(auth_manager=oauth2.SpotifyClientCredentials(st.secrets['id'], st.secrets['pw']))
sr = 22050
seq = 256
fps = 25
bin = 1025
M = model('data/model.pkl')
T, a, b = table('data/table.pkl')

st.image('eggman.png')
st.markdown('EgGMAn (Engine of Game Music Analogy) search for game music considering game and scene feature')

st.header('Source Music')
y = music(st.text_input('URL of Source Music', placeholder='Spotify, SoundCloud, YouTube, Hotlink'))

l, r = st.columns(2, gap='large')
p = scene(l, 'Source Scene')
q = scene(r, 'Target Scene')

st.header('Target Music')
with st.popover('Search Option'):
    i = st.multiselect('Ignore Artist', ['ANDY', 'BGMer', 'Nash Music Library', 'Seiko', 'TAZ', 'hitoshi', 'zukisuzuki', 'たう', 'ガレトコ', 'ユーフルカ'])
    j = st.multiselect('Ignore Site', ['BGMer', 'BGMusic', 'Nash Music Library', 'PeriTune', 'Senses Circuit', 'zukisuzuki BGM', 'ガレトコ', 'ユーフルカ', '音の園'])
    t = st.slider('Time Range', time(0), time(1), (time(0), time(1)), timedelta(seconds=10), 'mm:ss')
    s = st.slider('Random Scale', 0.0, 1.0, 1.0)
if st.button(f'Search {"EgGMAn" if y.size else "Random"}', type='primary'):
    try:
        if y.size:
            p, q = T[p & ~q], T[p & ~q]
            z = a * vec(y, s) - b - p['vec'].mean() + q['vec'].mean()
        else:
            q = T[q]
            z = rand(q['vec'].mean(), s * numpy.stack(q['vec']).std(0))
        o = q[~q['Artist'].isin(i) & ~q['Site'].isin(j) & q['Time'].between(t[0], t[1])]
        st.dataframe(o.iloc[numpy.argsort(((numpy.stack(o['vec']) - z) ** 2).sum(1))[:99], :5], column_config={'URL': st.column_config.LinkColumn(), 'Time': st.column_config.TimeColumn(format='mm:ss')})
    except:
        st.error('Too many conditions')
