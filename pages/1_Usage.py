import streamlit as st

st.set_page_config('EgGMAn', ':egg:', 'wide')
st.sidebar.link_button('Contact Us', 'https://forms.gle/A4vWuEAp4pPEY4sf9', use_container_width=True)
if st.sidebar.button('Clear Cache', use_container_width=True):
    st.cache_data.clear()

def view(i, w, h):
    s = f'https://drive.google.com/file/d/{i}/preview'
    st.markdown(f'<iframe src="{s}" width="{w}" height="{h}"></iframe>', True)

st.title('Usage')

st.header('Video')
view('1X0O0hd9EHayuhPaUyPzXYJZljInnRtgO', 700, 420)

st.header('Text')

st.subheader('Input Music')
st.markdown('- Input music to be used in the created game\n- Choose input way from Spotify API, Audiostock, YoutubeDL, Uploader\n- YoutubeDL has many [Supported Sites](https://github.com/yt-dlp/yt-dlp/blob/master/supportedsites.md)')

l, r = st.columns(2, gap='medium')

with l:
    st.subheader('Scene of Input Music')
    st.markdown('- Input the scene of input music\n- Valence and Arousal are based on [Circumplex Model](https://en.wikipedia.org/wiki/Emotion_classification#Circumplex_model)')

with r:
    st.subheader('Scene of Output Music')
    st.markdown('- Input the scene of output music\n- Valence and Arousal are based on [Circumplex Model](https://en.wikipedia.org/wiki/Emotion_classification#Circumplex_model)')
