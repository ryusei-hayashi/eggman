import streamlit as st

st.set_page_config('EgGMAn', ':musical_note:', 'wide')
st.sidebar.link_button('Contact Us', 'https://forms.gle/A4vWuEAp4pPEY4sf9', use_container_width=True)

def view(i, w, h):
    s = f'https://drive.google.com/file/d/{i}/preview'
    st.markdown(f'<iframe src="{s}" width="{w}" height="{h}"></iframe>', True)

st.title('Usage')

st.header('Video')
view('1X0O0hd9EHayuhPaUyPzXYJZljInnRtgO', 700, 420)

st.header('Text')

st.subheader('Source Music')
st.markdown('- Enter the URL of Source Music to be included in the developing game\n- URL support Spotify, YouTube, Audiostock')

l, r = st.columns(2, gap='medium')

with l:
    st.subheader('Source Scene')
    st.markdown('- Enter the tag of Source Scene to include Source Music\n- Enter from about 170 tags, including "Opening", "Spring", etc')

with r:
    st.subheader('Target Scene')
    st.markdown('- Enter the tag of Target Scene to include Target Music\n- Enter from about 170 tags, including "Opening", "Spring", etc')

st.subheader('Target Music')
st.markdown('- Search Target Music to be included in Target Scene\n- Search by EgGMAn if Source Music exists, or by Random if not')
