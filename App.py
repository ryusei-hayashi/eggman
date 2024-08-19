from tensorflow_probability import distributions as td, layers as tl
from statistics import mean, median
from essentia import standard as es
from gdown import download_folder
from tensorflow import keras
from yt_dlp import YoutubeDL
from pandas import DataFrame
from base64 import b64encode
from os.path import exists
from random import choice
from requests import get
from re import sub
import tensorflow as tf
import streamlit as st
import spotipy
import librosa
import numpy

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
        self.ct = keras.layers.Conv2DTranspose(channel, kernel, stride, padding)
        self.bn = keras.layers.BatchNormalization()

    def call(self, x):
        return tf.nn.relu(self.ct(self.bn(x)))

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
        self.ct1 = ConvT1(channel, kernel, stride, padding)
        self.ct2 = ConvT1(channel, kernel, stride, padding)

    def call(self, x):
        return self.ct2(self.ct1(x))

class Conv5(keras.Model):
    def __init__(self, channel, kernel, stride, padding):
        super(Conv5, self).__init__()
        self.cv1 = Conv1(channel[0], (kernel[0], 1), (stride[0], 1), padding[0])
        self.cv2 = Conv2(channel[0], (1, kernel[1]), (1, stride[1]), padding[1])
        self.cv3 = Conv2(channel[1], (1, kernel[1]), (1, stride[1]), padding[1])

    def call(self, x=0, y=0):
        return self.cv3(x), self.cv2(tf.nn.relu(self.cv1(x) + y))

class ConvT5(keras.Model):
    def __init__(self, channel, kernel, stride, padding):
        super(ConvT5, self).__init__()
        self.ct1 = ConvT1(channel[0], (kernel[0], 1), (stride[0], 1), padding[0])
        self.ct2 = ConvT2(channel[0], (1, kernel[1]), (1, stride[1]), padding[1])
        self.ct3 = ConvT2(channel[1], (1, kernel[1]), (1, stride[1]), padding[1])

    def call(self, x=0, y=0):
        return self.ct2(tf.nn.relu(self.ct1(x) + y)), self.ct3(x)

class Encoder(keras.Model):
    def __init__(self, c_n):
        super(Encoder, self).__init__()
        self.cv1 = keras.layers.Conv2D(z_n, (1, 1), activation='relu')
        self.cv2 = Conv5((c_n, z_n), (bin, 8), (1, 4), ('valid', 'same'))
        self.cv3 = Conv5((c_n, z_n), (bin, 8), (1, 4), ('valid', 'same'))
        self.cv4 = keras.layers.Conv2D(c_n, (bin, 1), activation='relu')
        self.fc1 = keras.layers.Dense(c_n)
        self.fc2 = keras.layers.Dense(c_n)

    def call(self, x):
        x = self.cv1(x)
        x, y = self.cv2(x=x)
        x, y = self.cv3(x=x, y=y)
        x = self.cv4(x)
        y = tf.nn.relu(x + y)
        y = tf.reshape(y, (-1, y.shape[-1]))
        y = self.fc1(y)
        y = tf.nn.relu(y)
        y = self.fc2(y)
        return y

class Decoder(keras.Model):
    def __init__(self, c_n):
        super(Decoder, self).__init__()
        self.fc1 = keras.layers.Dense(c_n)
        self.fc2 = keras.layers.Dense(c_n)
        self.ct1 = ConvT5((z_n, c_n), (bin, 8), (1, 4), ('valid', 'same'))
        self.ct2 = ConvT5((z_n, c_n), (bin, 8), (1, 4), ('valid', 'same'))
        self.ct3 = keras.layers.Conv2DTranspose(z_n, (bin, 1), activation='relu')
        self.ct4 = keras.layers.Conv2DTranspose(1, (1, 1), activation='relu')

    def call(self, z):
        y = self.fc1(z)
        y = tf.nn.relu(y)
        y = self.fc2(y)
        y = tf.reshape(y, (-1, 1, 1, y.shape[-1]))
        x, y = self.ct1(x=y)
        x, y = self.ct2(x=y, y=x)
        y = self.ct3(y)
        x = tf.nn.relu(x + y)
        x = self.ct4(x)
        return x

class VAE(keras.Model):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder(z_n * (z_n + 3) // 2)
        self.decoder = Decoder(z_n * (z_n + 3) // 2)
        self.mvntril = tl.MultivariateNormalTriL(z_n, activity_regularizer=tl.KLDivergenceRegularizer(td.MultivariateNormalTriL(tf.zeros(z_n)), True, weight=1e-4))

    def call(self, x):
        x = self.encoder(x)
        z = self.mvntril(x).sample()
        y = self.decoder(z)
        return y

    def get_z(self, x, r):
        d = self.mvntril(self.encoder(x, training=False))
        z = d.sample() if r else d.mean()
        return z.numpy()

def stft(y):
    return librosa.magphase(librosa.stft(y=y, hop_length=sr//fps, n_fft=2*bin-1))[0]

def mel(y):
    return librosa.feature.melspectrogram(y=y, sr=sr, hop_length=sr//fps, n_mels=bin)

def idx(a, v):
    i = numpy.searchsorted(a, v)
    return mean(a[i-1:i+1]) if 0 < i < len(a) else v

def mold(y, b, p=-1e-300):
    y = stft(y[idx(b, 10 * sr):idx(b, 20 * sr)])
    y = numpy.pad(y, ((0, 0), (0, max(0, seq - y.shape[1]))), constant_values=p)
    return y[None, :, :seq, None]

def vec(y):
    p, c = es.PitchMelodia(sampleRate=sr)(y)
    k, s, f = es.KeyExtractor(sampleRate=sr)(y)
    t, b = librosa.beat.beat_track(y=y, sr=sr, units='samples')
    a = ['C', 'G', 'D', 'A', 'E', 'B', 'F#', 'C#', 'Ab', 'Eb', 'Bb', 'F'].index(k) * math.pi / 6
    return numpy.r_[es.Loudness()(y), median(p[mean(c) < c]), t, f if 'a' in s else -f, f * math.cos(a), f * math.sin(a), M.get_z(mold(y, b), False)[0]]

def filter(s):
    return {k for k in S if all(i in S[k] for i in s)}

def center(K):
    return numpy.mean(numpy.array([Z[k] for k in K]), 0)

@st.cache_resource(max_entries=1)
def load_vae(n):
    m = VAE()
    m(tf.zeros((1, bin, seq, 1)))
    m.load_weights(n)
    return m

@st.cache_data(max_entries=4)
def load_npy(n):
    return numpy.load(n, allow_pickle=True).item()

@st.cache_data(ttl='9m')
def load_mp3(u):
    if u:
        try:
            if 'youtube' in u:
                yd.download([u])
            elif 'spotify' in u:
                open('tmp.mp3', 'wb').write(get(f'{sp.track(sub("intl-.*?/", "", u))["preview_url"]}.mp3').content)
            else:
                open('tmp.mp3', 'wb').write(get(u).content)
            src = f'data:audio/mp3;base64,{b64encode(open("tmp.mp3", "rb").read()).decode()}'
            st.markdown(f'<audio src="{src}" controlslist="nodownload" controls></audio>', True)
            return librosa.load('tmp.mp3', sr=sr, duration=30)[0]
        except:
            st.error(f'Error: Unable to access {u}')
    return numpy.empty(0)

if not exists('data'):
    download_folder('https://drive.google.com/drive/folders/1dtQgYKSeulm3mNS9auJ8axRpxW9Fdz4-')

yd = YoutubeDL({'outtmpl': 'tmp', 'playlist_items': '1', 'quiet': True, 'format': 'mp3/bestaudio/best', 'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'mp3'}], 'overwrites': True})
sp = spotipy.Spotify(auth_manager=spotipy.oauth2.SpotifyClientCredentials(st.secrets['id'], st.secrets['pw']))
sr = 22050
seq = 256
fps = 25
bin = 1025
z_n = 32

M = load_vae('data/vae.h5')
Z = load_npy('data/vec.npy')
S = load_npy('data/scn.npy')
U = load_npy('data/url.npy')

st.image('eggman.png')
st.markdown('- EgGMAn (Engine of Game Music Analysis) search for game music considering game and scene feature at the same time')

st.subheader('Source Music')
y = load_mp3(st.text_input('URL of Source Music', placeholder='Spotify, YouTube, Hotlink'))

l, r = st.columns(2, gap='medium')

with l:
    st.subheader('Source Scene')
    sss = st.multiselect('State of Source Scene', ['オープニング', 'エンディング', 'タイトル', 'イベント', 'チュートリアル', 'ゲーム失敗', 'ゲーム成功', 'ハイライト', 'ワールドマップ', 'フィールド', 'ダンジョン', 'ステージ', 'ショップ', 'メニュー', '選択画面', '休憩ポイント'], placeholder='Opening, Dungeon, ...')
    tss = st.multiselect('Time of Source Scene', ['春', '夏', '秋', '冬', '朝', '昼', '夜', '明方', '夕方', '休日', '原始', '古代', '中世', '近代', '現代', '未来'], placeholder='Spring, Morning, ...')
    wss = st.multiselect('Weather of Source Scene', ['虹', '星', '晴れ', '曇り', '霧', '砂', '雪', '雷', '雨', '小雨', '大雨', '突風', '混沌', 'オーロラ'], placeholder='Sunny, Cloudy, ...')
    bss = st.multiselect('Biome of Source Scene', ['異次元', '虚無', '宇宙', '大陸', '海', '空', '東', '西', '南', '北', '氷', '炎', '花', '毒', '沼', '湖', '泉', '滝', '川', '島', '岩', '崖', '山岳', '峡谷', '洞窟', '温泉', '水中', '水辺', '岸辺', '浜辺', '砂漠', '荒野', '草原', '森林', 'サバンナ', 'ジャングル'], placeholder='Sea, Sky, ...')
    pss = st.multiselect('Place of Source Scene', ['アジト', 'オフィス', 'カジノ', 'カフェ', 'キャバクラ', 'ジム', 'タワー', 'ナイトクラブ', 'ビル', 'プール', 'リゾート', 'レストラン', 'ロビー', '遺跡', '飲み屋', '駅', '仮想現実', '家', '外国', '街', '学校', '宮殿', '競技場', '教会', '橋', '軍事基地', '劇場', '研究機関', '公園', '工場', '港', '行政機関', '裁判所', '寺', '式場', '宿泊施設', '城塞', '神社', '神殿', '船', '村', '地下道', '庭', '邸宅', '鉄道', '店', '田舎', '都会', '動物園', '道路', '廃屋', '博物館', '畑', '飛行機', '病院', '墓', '牧場', '遊園地', '路地裏', '牢獄'], placeholder='Office, Casino, ...')
    qss = st.multiselect('Person of Source Scene', ['ゆるキャラ', 'ヒーロー', 'ヒロイン', 'スパイ', 'ライバル', 'ラスボス', 'ボス', 'モブ', '大衆', '貴族', '偉人', '仲間', '孤独', '平穏', '不穏', '敵'], placeholder='Hero, Rival, ...')
    ass = st.multiselect('Action of Source Scene', ['戦う', '泳ぐ', '走る', '飛ぶ', '会話', '回想', '休憩', '出会う', '別れる', '勝利', '敗北', '探検', '特訓', '謎解き', '買い物', '恋愛'], placeholder='Battle, Run, ...')

with r:
    st.subheader('Target Scene')
    sts = st.multiselect('State of Target Scene', ['オープニング', 'エンディング', 'タイトル', 'イベント', 'チュートリアル', 'ゲーム失敗', 'ゲーム成功', 'ハイライト', 'ワールドマップ', 'フィールド', 'ダンジョン', 'ステージ', 'ショップ', 'メニュー', '選択画面', '休憩ポイント'], placeholder='Opening, Dungeon, ...')
    tts = st.multiselect('Time of Target Scene', ['春', '夏', '秋', '冬', '朝', '昼', '夜', '明方', '夕方', '休日', '原始', '古代', '中世', '近代', '現代', '未来'], placeholder='Spring, Morning, ...')
    wts = st.multiselect('Weather of Target Scene', ['虹', '星', '晴れ', '曇り', '霧', '砂', '雪', '雷', '雨', '小雨', '大雨', '突風', '混沌', 'オーロラ'], placeholder='Sunny, Cloudy, ...')
    bts = st.multiselect('Biome of Target Scene', ['異次元', '虚無', '宇宙', '大陸', '海', '空', '東', '西', '南', '北', '氷', '炎', '花', '毒', '沼', '湖', '泉', '滝', '川', '島', '岩', '崖', '山岳', '峡谷', '洞窟', '温泉', '水中', '水辺', '岸辺', '浜辺', '砂漠', '荒野', '草原', '森林', 'サバンナ', 'ジャングル'], placeholder='Sea, Sky, ...')
    pts = st.multiselect('Place of Target Scene', ['アジト', 'オフィス', 'カジノ', 'カフェ', 'キャバクラ', 'ジム', 'タワー', 'ナイトクラブ', 'ビル', 'プール', 'リゾート', 'レストラン', 'ロビー', '遺跡', '飲み屋', '駅', '仮想現実', '家', '外国', '街', '学校', '宮殿', '競技場', '教会', '橋', '軍事基地', '劇場', '研究機関', '公園', '工場', '港', '行政機関', '裁判所', '寺', '式場', '宿泊施設', '城塞', '神社', '神殿', '船', '村', '地下道', '庭', '邸宅', '鉄道', '店', '田舎', '都会', '動物園', '道路', '廃屋', '博物館', '畑', '飛行機', '病院', '墓', '牧場', '遊園地', '路地裏', '牢獄'], placeholder='Office, Casino, ...')
    qts = st.multiselect('Person of Target Scene', ['ゆるキャラ', 'ヒーロー', 'ヒロイン', 'スパイ', 'ライバル', 'ラスボス', 'ボス', 'モブ', '大衆', '貴族', '偉人', '仲間', '孤独', '平穏', '不穏', '敵'], placeholder='Hero, Rival, ...')
    ats = st.multiselect('Action of Target Scene', ['戦う', '泳ぐ', '走る', '飛ぶ', '会話', '回想', '休憩', '出会う', '別れる', '勝利', '敗北', '探検', '特訓', '謎解き', '買い物', '恋愛'], placeholder='Battle, Run, ...')

st.subheader('Target Music')
stm = st.multiselect('Site of Target Music', ['BGMer', 'BGMusic', 'Nash Music Library', 'PeriTune', 'Senses Circuit', 'zukisuzuki BGM', 'ガレトコ', 'ユーフルカ', '音の園'], ['BGMer', 'BGMusic', 'Nash Music Library', 'PeriTune', 'Senses Circuit', 'zukisuzuki BGM', 'ガレトコ', 'ユーフルカ', '音の園'])
if st.button(f'Search by {"EgGMAn" if y.size else "Random"}', type='primary'):
    p, q = filter(sss + tss + wss + bss + pss + qss + ass), filter(sts + tts + wts + bts + pts + qts + ats)
    if y.size:
        p, q = p - q, q - p
    if p and q:
        z = Z['a'] * vec(y) + center(q) - center(p) - Z['b'] if y.size else Z[choice(list(q))]
        d = [U[k] for k in sorted(q, key=lambda k: numpy.linalg.norm(Z[k] - z)) if k[0] in stm][:99]
        st.dataframe(DataFrame(d, columns=['URL', 'Name', 'Artist', 'Time']), column_config={'URL': st.column_config.LinkColumn()})
    else:
        st.error('Too many conditions')
