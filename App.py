from datetime import time, timedelta
from essentia import standard as es
from statistics import median, mean
from gdown import download_folder
from numpy.random import normal
from numpy.linalg import norm
from tensorflow import keras
from yt_dlp import YoutubeDL
from base64 import b64encode
from os.path import exists
from requests import get
from pickle import load
import tensorflow as tf
import streamlit as st
import librosa
import numpy
import math

st.set_page_config('EgGMAn', ':musical_note:', 'wide', 'expanded')
st.sidebar.link_button('Contact Us', 'https://forms.gle/A4vWuEAp4pPEY4sf9', use_container_width=True)

if not exists('data'):
    download_folder('https://drive.google.com/drive/folders/1vjKbuINZh1a03lFPYWQdyMtYrQrJcFLS')

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
    def __init__(self, x_n, y_n, z_n):
        super(Encoder, self).__init__()
        self.cv1 = keras.layers.Conv2D(x_n, (1, 1), activation='relu')
        self.cv2 = keras.layers.Conv2D(y_n, (bin, 1), activation='relu')
        self.cv3 = Conv5((y_n, x_n), (bin, 8), (1, 4), ('valid', 'same'))
        self.cv4 = Conv5((y_n, x_n), (bin, 8), (1, 4), ('valid', 'same'))
        self.fc1 = keras.layers.Dense(z_n)
        self.fc2 = keras.layers.Dense(z_n)

    def call(self, x):
        x, y = self.cv3(self.cv1(x))
        x, y = self.cv4(x, y)
        y = tf.nn.relu(self.cv2(x) + y)
        y = tf.reshape(y, (-1, y.shape[-1]))
        y = self.fc2(tf.nn.relu(self.fc1(y)))
        return y

@st.cache_resource(max_entries=1)
def model(n):
    m = Encoder(32, 256, 128)
    m(tf.zeros((1, bin, seq, 1)), training=False)
    m.set_weights(load(open(n, 'rb')))
    return m

@st.cache_data(max_entries=1)
def table(n):
    t = load(open(n, 'rb'))
    a = 200 / (numpy.stack(t['vec']).max(0) - numpy.stack(t['vec']).min(0))
    b = a * numpy.stack(t['vec']).min(0) + 100
    return t, a, b

@st.cache_data(max_entries=1)
def music(m, n):
    if m:
        try:
            if n == 'Web Service':
                yd.download([m])
            elif n == 'Direct Link':
                open('music.mp3', 'wb').write(get(m).content)
            elif n == 'Audio File':
                open('music.mp3', 'wb').write(m.getvalue())
            st.markdown(f'<audio src="data:audio/mp3;base64,{b64encode(open("music.mp3", "rb").read()).decode()}" controlslist="nodownload" controls></audio>', True)
            return librosa.load('music.mp3', sr=sr, duration=30)[0]
        except:
            st.error(f'Unable to access {m}')
    return numpy.empty(0)

@st.cache_data(max_entries=3)
def query(q):
    return T.query(q)

def scn(c, s):
    with c:
        st.header(s)
        u = st.multiselect(f'State of {s}', ('オープニング', 'エンディング', 'タイトル', 'イベント', 'チュートリアル', 'メニュー画面', 'ショップ画面', 'マップ画面', 'ゲーム失敗', 'ゲーム成功', 'ハイライト', 'アナウンス', 'マイルーム', 'フィールド', 'ダンジョン', 'ステージ'), placeholder='オープニング/ダンジョン/...')
        t = st.multiselect(f'Time of {s}', ('春', '夏', '秋', '冬', '朝', '昼', '夜', '明方', '夕方', '休日', '原始', '古代', '中世', '近代', '現代', '未来'), placeholder='春/朝/...')
        w = st.multiselect(f'Weather of {s}', ('星', '虹', '晴れ', '曇り', '霧', '砂', '雪', '雷', '雨', '小雨', '大雨', '突風', '混沌', '朝焼け', '夕焼け', 'オーロラ'), placeholder='晴れ/曇り/...')
        b = st.multiselect(f'Biome of {s}', ('異次元', '虚無', '宇宙', '大陸', '海', '空', '東', '西', '南', '北', '氷', '炎', '花', '毒', '沼', '湖', '泉', '滝', '川', '島', '岩', '崖', '山岳', '峡谷', '洞窟', '温泉', '水中', '水辺', '岸辺', '浜辺', '砂漠', '荒野', '草原', '森林', 'サバンナ', 'ジャングル'), placeholder='草原/砂漠/...')
        p = st.multiselect(f'Place of {s}', ('仮想空間', '都会', '田舎', '街', '村', '橋', '道路', '路地裏', '地下道', '航空機', '鉄道', '船', '港', '駅', '店', '墓', '寺', '神社', '教会', '公園', '学校', '法廷', '病院', '劇場', '式場', '競技場', '博物館', '動物園', '遊園地', '飲み屋', '宿泊施設', '研究機関', '軍事基地', '城塞', '牢獄', '倉庫', '工場', '牧場', '畑', '庭', '家', '邸宅', '廃屋', '遺跡', '宮殿', '神殿', 'ロビー', 'タワー', 'ビル', 'ジム', 'プール', 'エステ', 'カフェ', 'レストラン', 'リゾート', 'オフィス', 'アジト', 'カジノ', 'キャバクラ', 'ナイトクラブ'), placeholder='街/店/...')
        q = st.multiselect(f'Person of {s}', ('ゆるキャラ', 'ヒーロー', 'ヒロイン', 'スパイ', 'ライバル', 'ラスボス', 'ボス', 'モブ', '大衆', '貴族', '偉人', '仲間', '孤独', '平穏', '不穏', '敵'), placeholder='ヒロイン/ラスボス/...')
        a = st.multiselect(f'Action of {s}', ('戦い', '探検', '潜入', '追跡', '逃走', '走る', '飛行', '泳ぐ', '会話', '休憩', '遊ぶ', '買い物', '謎解き', '練習', '支度', '出掛ける', '仕事', '挑戦', '挑発', '悪事', '犯罪', '騙す', '謀略', '議論', '集会', '食事', '癒し', '祈り', '儀式', '公演', '結婚', '恋愛', '魅惑', '声援', '勝利', '敗北', '破滅', '復讐', '仲違い', '仲直り', '出会う', '別れ', '登場', '説明', '回想', '感謝', '感動', '葛藤', '決意', '失意', '混乱', '慌てる', '危機', '好機', '騒動', 'ギャグ', 'レース', 'スポーツ', 'ギャンブル', 'パーティー'), placeholder='戦い/会話/...')
        st.subheader(f'Mood of {s}')
        l, r = st.columns(2, gap='medium')
        with l:
            v = st.slider(f'Valence of {s}', -1.0, 1.0, (-1.0, 1.0))
        with r:
            z = st.slider(f'Arousal of {s}', -1.0, 1.0, (-1.0, 1.0))
    return ''.join(f"scn.str.contains('{i}') and " for i in u + t + w + b + p + q + a) + f"{v[0]} <= pn <= {v[1]} and {z[0]} <= ap <= {z[1]}"

def set(c):
    with c:
        a = st.multiselect('Ignore Artist', ('ANDY', 'BGMer', 'Nash Music Library', 'Seiko', 'TAZ', 'hitoshi', 'zukisuzuki', 'たう', 'ガレトコ', 'ユーフルカ'), placeholder='')
        s = st.multiselect('Ignore Site', ('BGMer', 'BGMusic', 'Nash Music Library', 'PeriTune', 'Senses Circuit', 'zukisuzuki BGM', 'ガレトコ', 'ユーフルカ', '音の園'), placeholder='')
        t = st.slider('Time Range', time(0), time(0, 59), (time(0), time(0, 59)), timedelta(seconds=10), 'mm:ss')
        r = st.slider('Random Rate', 0.0, 1.0, 1.0)
    return f"not Artist in {str(a)} and not Site in {str(s)} and '{t[0].strftime('%M:%S')}' <= Time <= '{t[1].strftime('%M:%S')}'", r

def idx(n, v):
    i = n.searchsorted(v)
    return mean(n[i-1:i+1]) if 0 < i < len(n) else v

def mel(y):
    return librosa.feature.melspectrogram(y=y, sr=sr, hop_length=sr//fps, n_mels=bin)

def stft(y):
    return librosa.magphase(librosa.stft(y=y, hop_length=sr//fps, n_fft=2*bin-2))[0]

def mold(y, b, p=-1e-99):
    y = stft(y[idx(b, 10 * sr):idx(b, 20 * sr)])
    y = numpy.pad(y, ((0, 0), (0, max(0, seq - y.shape[1]))), constant_values=p)
    return y[None, :, :seq, None]

def core(p, q):
    return numpy.median(numpy.stack(query(f'({p}) and not ({q})')['vec']), 0)

def vec(y, r):
    t, b = librosa.beat.beat_track(y=y, sr=sr, units='samples')
    m, v = numpy.split(M.predict(mold(y, b))[0], 2)
    k, s, f = es.KeyExtractor(sampleRate=sr)(y)
    p, c = es.PitchMelodia(sampleRate=sr)(y)
    a = ['C', 'G', 'D', 'A', 'E', 'B', 'F#', 'C#', 'Ab', 'Eb', 'Bb', 'F'].index(k) * math.pi / 6
    return numpy.r_[es.Loudness()(y), median(p[mean(c) < c]), t, f if 'a' in s else -f, f * math.cos(a), f * math.sin(a), normal(m, r * tf.math.softplus(v))]

yd = YoutubeDL({'outtmpl': 'music', 'playlist_items': '1', 'format': 'bestaudio', 'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'mp3'}], 'postprocessor_args': ['-ss', '0', '-t', '30'], 'overwrites': True})
sr = 22050
seq = 256
fps = 25
bin = 1025
M = model('data/model.pkl')
T, a, b = table('data/table.pkl')

st.image('imgs/logo.png')
st.markdown('<center>EgGMAn (Engine of Game Music Analogy) searches for game music considering game-wise consistency and scene-wise individuality</center>', True)

st.header('Source Music')
m = st.segmented_control('Mode of Source Music', ('Web Service', 'Direct Link', 'Audio File'), default='Web Service')
y = music(st.file_uploader('File of Source Music') if 'File' in m else st.text_input('URL of Source Music'), m)

c = st.columns(2, gap='large')
p = scn(c[0], 'Source Scene')
q = scn(c[1], 'Target Scene')

st.header('Target Music')
o, r = set(st.popover('Search Option'))
if st.button(f'Search {"EgGMAn" if y.size else "Random"}', type='primary'):
    try:
        if y.size:
            t = query(f'not ({p}) and ({q}) and ({o})')
            z = a * vec(y, r) - b - core(p, q) + core(q, p)
        else:
            t = query(f'({q}) and ({o})')
            z = normal(t['vec'].mean(), r * numpy.stack(t['vec']).std(0)) 
        st.dataframe(t.iloc[norm(numpy.stack(t['vec']) - z, axis=1).argsort()[:99], :5].reset_index(drop=True), column_config={'URL': st.column_config.LinkColumn()})
    except:
        st.error('No music matches the conditions')
