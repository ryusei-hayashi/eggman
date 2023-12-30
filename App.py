from gdown import download_folder
from yt_dlp import YoutubeDL
from pandas import DataFrame
from base64 import b64encode
from tensorflow import keras
from statistics import mean
from requests import get
import tensorflow_probability as tfp
import tensorflow as tf
import streamlit as st
import spotipy
import librosa
import numpy
import os

if not os.path.exists('data'):
    download_folder(id='1jwaqTqRFvQzVMNbvkNrZJn5Mq8WkGxCi')

st.set_page_config('EgGMAn', ':egg:', 'wide')
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
        self.sample = tfp.layers.MultivariateNormalTriL(z_n, activity_regularizer=tfp.layers.KLDivergenceRegularizer(tfp.distributions.Independent(tfp.distributions.Normal(tf.zeros(z_n), 1), 1), weight=1e-4))

    def call(self, x):
        x = self.encoder(x)
        z = self.sample(x)
        y = self.decoder(z)
        return y

    def get_z(self, x, v):
        x = self.encoder(x, training=False)
        z = tf.convert_to_tensor(self.sample(x)) if v else x[:,:z_n]
        return z.numpy()

@st.cache_resource(max_entries=1)
def load_h5(f):
    m = VAE()
    m(tf.random.normal([1, x_n, seq, 1]))
    m.load_weights(f)
    return m

@st.cache_data(max_entries=4)
def load_np(f):
    return numpy.load(f, allow_pickle=True).item()

@st.cache_data(max_entries=1)
def download(s):
    try:
        if w == 'Spotify API':
            open('tmp.mp3', 'wb').write(get(f'{sp.track(s.replace("intl-ja/", ""))["preview_url"]}.mp3').content)
        elif w == 'Audiostock':
            open('tmp.mp3', 'wb').write(get(f'{s}/play.mp3').content)
        elif w == 'YoutubeDL':
            yd.download([s])
        elif w == 'Uploader':
            open('tmp.mp3', 'wb').write(s.getbuffer())
        src = f'data:audio/mp3;base64,{b64encode(open("tmp.mp3", "rb").read()).decode()}'
        st.markdown(f'<audio src="{src}" controlslist="nodownload" controls></audio>', True)
        return librosa.load('tmp.mp3', sr=sr, offset=10, duration=2*sec)[0]
    except:
        st.error(f'Error: Unable to access {s}')
        return numpy.zeros(1)

def trim(y):
    b = librosa.beat.beat_track(y=y, sr=sr, hop_length=sr//fps)[1]
    if len(b) < 9:
        return y[:sr*sec]
    s = mean(b[:2])
    i = numpy.searchsorted(b, s + sec * fps) - 1
    return y[sr*s//fps:sr*mean(b[i:i+2])//fps]

def stft(y):
    return librosa.magphase(librosa.stft(y=y, hop_length=sr//fps, n_fft=2*x_n-1))[0]

def cqt(y):
    return librosa.magphase(librosa.cqt(y=y, hop_length=sr//fps, n_bins=x_n, bins_per_octave=x_n//7))[0]

def mel(y):
    return librosa.feature.melspectrogram(y=y, hop_length=sr//fps, n_mels=x_n)

def pad(y):
    return numpy.pad(y, ((0, x_n-y.shape[0]), (0, seq-y.shape[1])), constant_values=-1e-300)

def collate(Y):
    return numpy.array([pad(stft(trim(y))[:x_n,:seq]) for y in Y])[:,:,:,numpy.newaxis]

def filter(s, v, a):
    return [k for k in Z if all(i in S[k] for i in s) and v[0] < V[k][0] < v[1] and a[0] < V[k][1] < a[1]]

def center(K):
    return numpy.mean(numpy.array([Z[k] for k in K]), axis=0)

yd = YoutubeDL({'outtmpl': 'tmp', 'playlist_items': '1', 'quiet': True, 'format': 'mp3/bestaudio/best', 'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'mp3'}], 'overwrites': True})
sp = spotipy.Spotify(auth_manager=spotipy.oauth2.SpotifyClientCredentials(st.secrets['id'], st.secrets['pw']))
sr = 22050
fps = 25
sec = 10
seq = 256
z_n = 32
x_n = 1024

M = load_h5('data/vae.h5')
Z = load_np('data/vec.npy')
S = load_np('data/scn.npy')
V = load_np('data/vad.npy')
U = load_np('data/url.npy')

st.title('EgGMAn')
st.write('EgGMAn (Engine of Game Music Analysis) retrieves music that has both the worldview of the game and the atmosphere of the scene.')

st.subheader('Input Music')
w = st.selectbox('Input Way', ['Spotify API', 'Audiostock', 'YoutubeDL', 'Uploader'])
if w == 'Uploader':
    s = st.file_uploader('Upload File')
else:
    s = st.text_input('Input URL')
if s:
    y = download(s)
    if os.path.exists('tmp.mp3'):
        os.remove('tmp.mp3')

l, r = st.columns(2, gap='medium')

with l:
    st.subheader('Scene of Input Music')
    sim = st.multiselect('State of input music', ['オープニング', 'タイトル', 'チュートリアル', 'ゲームオーバー', 'ゲームクリア', 'セレクト', 'ショップ', 'ミニイベント', 'セーブエリア', 'ワールドマップ', 'ダンジョン', 'ステージ', 'エンディング'])
    tim = st.multiselect('Time of input music', ['春', '夏', '秋', '冬', '朝', '昼', '夜', '夕方', '休日', '古代', '中世', '近代', '現代', '未来'])
    wim = st.multiselect('Weather of input music', ['晴れ', '虹', '雲', '嵐', '雪', '砂', '雨', '小雨', '混沌'])
    bim = st.multiselect('Biome of input music', ['水上', '水中', '海', '湖', '川', '山', '島', '浜辺', '洞窟', '砂漠', '荒野', '草原', '熱帯', '森', '炎', '空', '宇宙', '異次元'])
    pim = st.multiselect('Place of input music', ['仮想現実', '外国', '都会', '田舎', '街', 'アジト', 'オフィス', 'ビル', 'ジム', '農地', '牧場', '工場', '研究所', '軍事基地', '学校', '公園', '病院', '法廷', '競技場', '美術館', '飛行機', '電車', '船', '橋', 'シアター', 'カジノ', '遊園地', '城', '遺跡', '神社', '寺院', '教会', '宮殿', '神殿', '聖域', 'レストラン', 'カフェ', 'ホテル', 'バー', '酒場', '店', '家', '廃墟', '高台'])
    qim = st.multiselect('Person of input music', ['主人公', '相棒', '仲間', '先人', '観衆', '日常', '非常', '敵', '孤独', '裏切者', '中ボス', 'ラスボス', 'ライバル', 'マスコット', 'ヒロイン', 'モブ'])
    aim = st.multiselect('Action of input music', ['移動', '走る', '泳ぐ', '飛ぶ', '運動', '競走', '遊ぶ', '休む', '考える', '閃く', '作業', '戦う', '潜入', '探索', '追う', '逃げる', '取引き', '宴', '勝利', '回想', '覚醒', '感動', '説得', '決意', '成長', '悩む', '出会い', '別れ', '登場', '不穏', '平穏', '解説', '熱狂', '困惑', '謀略', '犯罪', '暴力', 'ふざける', 'あおる', '恋愛', '感謝', '癒す', '励ます', '出掛ける'])
    vim = st.slider('Valence of input music', -1.0, 1.0, (-1.0, 1.0))
    zim = st.slider('Arousal of input music', -1.0, 1.0, (-1.0, 1.0))

with r:
    st.subheader('Scene of Output Music')
    som = st.multiselect('State of output music', ['オープニング', 'タイトル', 'チュートリアル', 'ゲームオーバー', 'ゲームクリア', 'セレクト', 'ショップ', 'ミニイベント', 'セーブエリア', 'ワールドマップ', 'ダンジョン', 'ステージ', 'エンディング'])
    tom = st.multiselect('Time of output music', ['春', '夏', '秋', '冬', '朝', '昼', '夜', '夕方', '休日', '古代', '中世', '近代', '現代', '未来'])
    wom = st.multiselect('Weather of output music', ['晴れ', '虹', '雲', '嵐', '雪', '砂', '雨', '小雨', '混沌'])
    bom = st.multiselect('Biome of output music', ['水上', '水中', '海', '湖', '川', '山', '島', '浜辺', '洞窟', '砂漠', '荒野', '草原', '熱帯', '森', '炎', '空', '宇宙', '異次元'])
    pom = st.multiselect('Place of output music', ['仮想現実', '外国', '都会', '田舎', '街', 'アジト', 'オフィス', 'ビル', 'ジム', '農地', '牧場', '工場', '研究所', '軍事基地', '学校', '公園', '病院', '法廷', '競技場', '美術館', '飛行機', '電車', '船', '橋', 'シアター', 'カジノ', '遊園地', '城', '遺跡', '神社', '寺院', '教会', '宮殿', '神殿', '聖域', 'レストラン', 'カフェ', 'ホテル', 'バー', '酒場', '店', '家', '廃墟', '高台'])
    qom = st.multiselect('Person of output music', ['主人公', '相棒', '仲間', '先人', '観衆', '日常', '非常', '敵', '孤独', '裏切者', '中ボス', 'ラスボス', 'ライバル', 'マスコット', 'ヒロイン', 'モブ'])
    aom = st.multiselect('Action of output music', ['移動', '走る', '泳ぐ', '飛ぶ', '運動', '競走', '遊ぶ', '休む', '考える', '閃く', '作業', '戦う', '潜入', '探索', '追う', '逃げる', '取引き', '宴', '勝利', '回想', '覚醒', '感動', '説得', '決意', '成長', '悩む', '出会い', '別れ', '登場', '不穏', '平穏', '解説', '熱狂', '困惑', '謀略', '犯罪', '暴力', 'ふざける', 'あおる', '恋愛', '感謝', '癒す', '励ます', '出掛ける'])
    vom = st.slider('Valence of output music', -1.0, 1.0, (-1.0, 1.0))
    zom = st.slider('Arousal of output music', -1.0, 1.0, (-1.0, 1.0))

st.subheader('Output Music')
if st.button('Retrieve', type='primary'):
    p = filter(sim + tim + wim + bim + pim + qim + aim, vim, zim)
    q = filter(som + tom + wom + bom + pom + qom + aom, vom, zom)
    if p and q:
        z = M.get_z(collate([y]), True)[0] + center(q) - center(p)
        d = DataFrame([U[k] for k in sorted(q, key=lambda k: numpy.linalg.norm(Z[k]-z))[:50]], columns=['URL', 'Name', 'Artist', 'Time'])
        st.dataframe(d, column_config={'URL': st.column_config.LinkColumn()})
    else:
        st.error('Error: No music to fit the input scene')
