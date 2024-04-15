from yt_dlp import YoutubeDL
from pandas import DataFrame
from base64 import b64encode
from tensorflow import keras
from statistics import mean
from random import choice
from requests import get
from time import sleep
from re import sub
import tensorflow_probability as tfp
import tensorflow as tf
import streamlit as st
import spotipy
import librosa
import numpy
import os

st.set_page_config('EgGMAn', ':musical_note:', 'wide')
st.sidebar.link_button('Contact Us', 'https://forms.gle/A4vWuEAp4pPEY4sf9', use_container_width=True)

class Conv1(keras.Model):
    def __init__(self, channel, kernel, stride, padding):
        super(Conv1, self).__init__()
        self.cv = keras.layers.Conv2D(channel, kernel, stride, padding)
        self.bn = keras.layers.BatchNormalization()

    def call(self, x):
        return tf.nn.relu(self.cv(self.bn(x)))

class ConvT1(keras.Model):
    def __init__(self, channel, kernel, stride, padding):
        super(ConvT1, self).__init__()
        self.cv = keras.layers.Conv2DTranspose(channel, kernel, stride, padding)
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

class ConvT2(keras.Model):
    def __init__(self, channel, kernel, stride, padding):
        super(ConvT2, self).__init__()
        self.cvt1 = ConvT1(channel, kernel, stride, padding)
        self.cvt2 = ConvT1(channel, kernel, stride, padding)

    def call(self, x):
        return self.cvt2(self.cvt1(x))

class Conv5(keras.Model):
    def __init__(self, channel, kernel, stride, padding):
        super(Conv5, self).__init__()
        self.cv1 = Conv1(channel[0], (kernel[0], 1), (stride[0], 1), padding[0])
        self.cv2 = Conv2(channel[0], (1, kernel[1]), (1, stride[1]), padding[1])
        self.cv3 = Conv2(channel[1], (1, kernel[1]), (1, stride[1]), padding[1])

    def call(self, x, y):
        return self.cv3(x), self.cv2(tf.nn.relu(self.cv1(x) + y))

class ConvT5(keras.Model):
    def __init__(self, channel, kernel, stride, padding):
        super(ConvT5, self).__init__()
        self.cvt1 = ConvT1(channel[0], (kernel[0], 1), (stride[0], 1), padding[0])
        self.cvt2 = ConvT2(channel[0], (1, kernel[1]), (1, stride[1]), padding[1])
        self.cvt3 = ConvT2(channel[1], (1, kernel[1]), (1, stride[1]), padding[1])

    def call(self, x, y):
        return self.cvt2(tf.nn.relu(self.cvt1(y) + x)), self.cvt3(y)

class Encoder(keras.Model):
    def __init__(self, a_n, v_n):
        super(Encoder, self).__init__()
        self.cv1 = keras.layers.Conv2D(a_n, (1, 1), activation='relu')
        self.cv2 = Conv5((a_n + v_n, a_n), (x_n, 8), (1, 4), ('valid', 'same'))
        self.cv3 = Conv5((a_n + v_n, a_n), (x_n, 8), (1, 4), ('valid', 'same'))
        self.cv4 = keras.layers.Conv2D(a_n + v_n, (x_n, 1), activation='relu')
        self.fc1 = keras.layers.Dense(a_n + v_n)
        self.fc2 = keras.layers.Dense(a_n + v_n)

    def call(self, x):
        x = self.cv1(x)
        x, y = self.cv2(x, 0.0)
        x, y = self.cv3(x, y)
        x = self.cv4(x)
        y = tf.nn.relu(x + y)
        y = tf.reshape(y, (-1, y.shape[-1]))
        y = self.fc1(y)
        y = tf.nn.relu(y)
        y = self.fc2(y)
        return y

class Decoder(keras.Model):
    def __init__(self, a_n, v_n):
        super(Decoder, self).__init__()
        self.fc1 = keras.layers.Dense(a_n + v_n)
        self.fc2 = keras.layers.Dense(a_n + v_n)
        self.cvt1 = ConvT5((a_n, a_n + v_n), (x_n, 8), (1, 4), ('valid', 'same'))
        self.cvt2 = ConvT5((a_n, a_n + v_n), (x_n, 8), (1, 4), ('valid', 'same'))
        self.cvt3 = keras.layers.Conv2DTranspose(a_n, (x_n, 1), activation='relu')
        self.cvt4 = keras.layers.Conv2DTranspose(1, (1, 1), activation='relu')

    def call(self, z):
        y = self.fc1(z)
        y = tf.nn.relu(y)
        y = self.fc2(y)
        y = tf.reshape(y, (-1, 1, 1, y.shape[-1]))
        x, y = self.cvt1(0.0, y)
        x, y = self.cvt2(x, y)
        y = self.cvt3(y)
        x = tf.nn.relu(x + y)
        x = self.cvt4(x)
        return x

class VAE(keras.Model):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder(z_n, z_n * (z_n + 1) // 2)
        self.decoder = Decoder(z_n, z_n * (z_n + 1) // 2)
        self.sample = tfp.layers.MultivariateNormalTriL(z_n, activity_regularizer=tfp.layers.KLDivergenceRegularizer(tfp.distributions.Independent(tfp.distributions.Normal(tf.zeros(z_n), 1), 1)))

    def call(self, x):
        x = self.encoder(x)
        z = self.sample(x)
        y = self.decoder(z)
        return y

    def get_z(self, x, r):
        x = self.encoder(x, training=False)
        z = tf.convert_to_tensor(self.sample(x)) if r else x[:,:z_n]
        return z.numpy()

def download(n):
    while not os.path.exists(n):
        open(n, 'wb').write(get(f'http://virgo.is.chs.nihon-u.ac.jp/~h5419056/EgGMAn/{n}').content)
        sleep(1)

def trim(y):
    b = librosa.beat.beat_track(y=y, sr=sr, hop_length=sr//fps)[1]
    if len(b) < 9:
        return y[:sr*sec]
    s = mean(b[:2])
    i = numpy.searchsorted(b, s + sec * fps) - 1
    return y[sr*s//fps:sr*mean(b[i:i+2])//fps]

def stft(y):
    return librosa.magphase(librosa.stft(y=y, hop_length=sr//fps, n_fft=2*x_n-1))[0]

def pad(y):
    return numpy.pad(y, ((0, x_n-y.shape[0]), (0, seq-y.shape[1])), constant_values=-1e-300)

def collate(Y):
    return numpy.array([pad(stft(trim(y))[:x_n,:seq]) for y in Y])[:,:,:,numpy.newaxis]

def filter(s):
    return {k for k in Z if all(i in S[k] for i in s)}

def center(K):
    return numpy.mean(numpy.array([Z[k] for k in K]), axis=0)

@st.cache_resource(max_entries=1)
def load_vae(n):
    download(n)
    m = VAE()
    m(tf.random.normal([1, x_n, seq, 1]))
    m.load_weights(n)
    return m

@st.cache_data(max_entries=4)
def load_npy(n):
    download(n)
    return numpy.load(n, allow_pickle=True).item()

@st.cache_data(ttl='9m')
def load_mp3(u):
    if u:
        try:
            if 'audiostock' in u:
                open('tmp.mp3', 'wb').write(get(f'{u}/play').content)
            elif 'spotify' in u:
                open('tmp.mp3', 'wb').write(get(f'{sp.track(sub("intl-.*?/", "", u))["preview_url"]}.mp3').content)
            elif 'youtube' in u:
                yd.download([u])
            src = f'data:audio/mp3;base64,{b64encode(open("tmp.mp3", "rb").read()).decode()}'
            st.markdown(f'<audio src="{src}" controlslist="nodownload" controls></audio>', True)
            return librosa.load('tmp.mp3', sr=sr, offset=sec, duration=2*sec)[0]
        except:
            st.error(f'Error: Unable to access {u}')
    return numpy.empty(0)

yd = YoutubeDL({'outtmpl': 'tmp', 'playlist_items': '1', 'quiet': True, 'format': 'mp3/bestaudio/best', 'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'mp3'}], 'overwrites': True})
sp = spotipy.Spotify(auth_manager=spotipy.oauth2.SpotifyClientCredentials(st.secrets['id'], st.secrets['pw']))
sr = 22050
fps = 25
sec = 10
seq = 256
z_n = 32
x_n = 1024

M = load_vae('vae.h5')
Z = load_npy('vec.npy')
S = load_npy('scn.npy')
U = load_npy('url.npy')

st.title('EgGMAn')
st.markdown('- EgGMAn (Engine of Game Music Analysis) search for game music considering game and scene feature at the same time')

st.subheader('Source Music')
u = st.text_input('Source URL')
y = load_mp3(u)

l, r = st.columns(2, gap='medium')

with l:
    st.subheader('Source Scene')
    sss = st.multiselect('State of Source Scene', ['オープニング', 'タイトル', 'チュートリアル', 'ゲームオーバー', 'ゲームクリア', 'セレクト', 'ショップ', 'ミニイベント', 'セーブエリア', 'ワールドマップ', 'ダンジョン', 'ステージ', 'エンディング'], help='help')
    tss = st.multiselect('Time of Source Scene', ['春', '夏', '秋', '冬', '朝', '昼', '夜', '夕方', '休日', '古代', '中世', '近代', '現代', '未来'])
    wss = st.multiselect('Weather of Source Scene', ['晴れ', '虹', '雲', '嵐', '雪', '砂', '雨', '小雨', '混沌'])
    bss = st.multiselect('Biome of Source Scene', ['水上', '水中', '海', '湖', '川', '山', '島', '浜辺', '洞窟', '砂漠', '荒野', '草原', '熱帯', '森', '炎', '空', '宇宙', '異次元'])
    pss = st.multiselect('Place of Source Scene', ['仮想現実', '外国', '都会', '田舎', '街', 'アジト', 'オフィス', 'ビル', 'ジム', '農地', '牧場', '工場', '研究所', '軍事基地', '学校', '公園', '病院', '法廷', '競技場', '美術館', '飛行機', '電車', '船', '橋', 'シアター', 'カジノ', '遊園地', '城', '遺跡', '神社', '寺院', '教会', '宮殿', '神殿', '聖域', 'レストラン', 'カフェ', 'ホテル', 'バー', '酒場', '店', '家', '廃墟', '高台'])
    qss = st.multiselect('Person of Source Scene', ['主人公', '相棒', '仲間', '先人', '観衆', '日常', '非常', '敵', '孤独', '裏切者', '中ボス', 'ラスボス', 'ライバル', 'マスコット', 'ヒロイン', 'モブ'])
    ass = st.multiselect('Action of Source Scene', ['移動', '走る', '泳ぐ', '飛ぶ', '運動', '競走', '遊ぶ', '休む', '考える', '閃く', '作業', '戦う', '潜入', '探索', '追う', '逃げる', '取引き', '宴', '勝利', '回想', '覚醒', '感動', '説得', '決意', '成長', '悩む', '出会い', '別れ', '登場', '不穏', '平穏', '解説', '熱狂', '困惑', '謀略', '犯罪', '暴力', 'ふざける', 'あおる', '恋愛', '感謝', '癒す', '励ます', '出掛ける'])

with r:
    st.subheader('Target Scene')
    sts = st.multiselect('State of Target Scene', ['オープニング', 'タイトル', 'チュートリアル', 'ゲームオーバー', 'ゲームクリア', 'セレクト', 'ショップ', 'ミニイベント', 'セーブエリア', 'ワールドマップ', 'ダンジョン', 'ステージ', 'エンディング'])
    tts = st.multiselect('Time of Target Scene', ['春', '夏', '秋', '冬', '朝', '昼', '夜', '夕方', '休日', '古代', '中世', '近代', '現代', '未来'])
    wts = st.multiselect('Weather of Target Scene', ['晴れ', '虹', '雲', '嵐', '雪', '砂', '雨', '小雨', '混沌'])
    bts = st.multiselect('Biome of Target Scene', ['水上', '水中', '海', '湖', '川', '山', '島', '浜辺', '洞窟', '砂漠', '荒野', '草原', '熱帯', '森', '炎', '空', '宇宙', '異次元'])
    pts = st.multiselect('Place of Target Scene', ['仮想現実', '外国', '都会', '田舎', '街', 'アジト', 'オフィス', 'ビル', 'ジム', '農地', '牧場', '工場', '研究所', '軍事基地', '学校', '公園', '病院', '法廷', '競技場', '美術館', '飛行機', '電車', '船', '橋', 'シアター', 'カジノ', '遊園地', '城', '遺跡', '神社', '寺院', '教会', '宮殿', '神殿', '聖域', 'レストラン', 'カフェ', 'ホテル', 'バー', '酒場', '店', '家', '廃墟', '高台'])
    qts = st.multiselect('Person of Target Scene', ['主人公', '相棒', '仲間', '先人', '観衆', '日常', '非常', '敵', '孤独', '裏切者', '中ボス', 'ラスボス', 'ライバル', 'マスコット', 'ヒロイン', 'モブ'])
    ats = st.multiselect('Action of Target Scene', ['移動', '走る', '泳ぐ', '飛ぶ', '運動', '競走', '遊ぶ', '休む', '考える', '閃く', '作業', '戦う', '潜入', '探索', '追う', '逃げる', '取引き', '宴', '勝利', '回想', '覚醒', '感動', '説得', '決意', '成長', '悩む', '出会い', '別れ', '登場', '不穏', '平穏', '解説', '熱狂', '困惑', '謀略', '犯罪', '暴力', 'ふざける', 'あおる', '恋愛', '感謝', '癒す', '励ます', '出掛ける'])

st.subheader('Target Music')
if st.button(f'Search by {"EgGMAn" if y.size else "Random"}', type='primary'):
    p = filter(sss + tss + wss + bss + pss + qss + ass)
    q = filter(sts + tts + wts + bts + pts + qts + ats)
    if y.size:
        p, q = p - q, q - p
    if p and q:
        z = M.get_z(collate([y]), True)[0] + center(q) - center(p) if y.size else Z[choice(list(q))]
        d = [U[k] for k in sorted(q, key=lambda k: numpy.linalg.norm(Z[k]-z))[:99]]
        st.dataframe(DataFrame(d, columns=['URL', 'Name', 'Artist', 'Time']), column_config={'URL': st.column_config.LinkColumn()})
    else:
        st.error('Too many conditions')
