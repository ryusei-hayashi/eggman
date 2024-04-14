import streamlit as st

st.set_page_config('EgGMAn', ':musical_note:', 'wide')
st.sidebar.link_button('Contact Us', 'https://forms.gle/A4vWuEAp4pPEY4sf9', use_container_width=True)

def view(i, w, h):
    s = f'https://drive.google.com/file/d/{i}/preview'
    st.markdown(f'<iframe src="{s}" width="{w}" height="{h}"></iframe>', True)

st.title('System')

st.header('Video')
view('1a167-_xiTrvA9JtiEfI8RST_T8996Stu', 700, 420)

st.header('Text')

st.subheader('Source Music')
st.markdown('- Convert Source Music to the vector z in VAE')

l, r = st.columns(2, gap='medium')

with l:
    st.subheader('Source Scene')
    st.markdown('- Create a set of music to use in the same scene as Source Scene\n- Convert a set of music to a set of vector in VAE\n- Compute the center p of a set of vector')

with r:
    st.subheader('Target Scene')
    st.markdown('- Create a set of music to use in the same scene as Target Scene\n- Convert a set of music to a set of vector in VAE\n- Compute the center q of a set of vector')

st.subheader('Target Music')
st.markdown('- Move the vector z in the vector q - p direction\n- Search music near the moved vector z')
