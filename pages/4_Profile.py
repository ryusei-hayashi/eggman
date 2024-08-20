import streamlit as st

st.set_page_config('EgGMAn', ':musical_note:', 'wide')
st.sidebar.link_button('Contact Us', 'https://forms.gle/A4vWuEAp4pPEY4sf9', use_container_width=True)

def profile(n, a, p, m):
    st.subheader(n)
    st.markdown(f'- Affiliation: {a}\n- Position: {p}\n- Mail: {m}')

st.title('Profile')

st.header('Developer')
profile('Ryusei Hayashi', 'Nihon University, Sound Computing Laboratory', 'Student', 'ryusei＠kthrlab.jp')

st.header('Supporter')
profile('Tetsuro Kitahara', 'Nihon University, Sound Computing Laboratory', 'Professor', 'kitahara＠kthrlab.jp')
