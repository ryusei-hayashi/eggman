import streamlit as st

st.set_page_config('EgGMAn', ':musical_note:', 'wide')
st.sidebar.link_button('Contact Us', 'https://forms.gle/A4vWuEAp4pPEY4sf9', use_container_width=True)

def iframe(u, w, h):
    st.markdown(f'<iframe src="{u}/preview" width="{w}" height="{h}"></iframe>', True)

st.title('Usage')

iframe('https://drive.google.com/file/d/1X0O0hd9EHayuhPaUyPzXYJZljInnRtgO', 700, 420)

st.subheader('Source Music')
st.markdown('- Enter the URL of the music to use in developing game\n- URL support Spotify, YouTube, Hotlink')

l, r = st.columns(2, gap='medium')

with l:
    st.subheader('Source Scene')
    st.markdown('- Enter the scene where Source Music is used\n- There are about 200 scenes including "Opening", "Spring", etc')

with r:
    st.subheader('Target Scene')
    st.markdown('- Enter the scene where Target Music is used\n- There are about 200 scenes including "Opening", "Spring", etc')

st.subheader('Target Music')
st.markdown('- Enter the site of No Copyright Music\n- EgGMAn search music for Target Scene in developing game\n- EgGMAn searche randomly if there is no Source Music')
