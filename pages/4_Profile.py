import streamlit as st

st.set_page_config('EgGMAn', ':musical_note:', 'wide', 'expanded')
st.sidebar.link_button('Contact Us', 'https://forms.gle/A4vWuEAp4pPEY4sf9', use_container_width=True)

def profile(n, d):
    st.subheader(n)
    st.markdown('\n'.join(f'- {k}: {d[k]}' for k in d))

st.title('Profile')

st.header('Developer')
profile('Ryusei Hayashi', {'Affiliation': 'Nihon University, Sound Computing Laboratory', 'Position': 'Student', 'Mail': 'ryusei＠kthrlab.jp'})

st.header('Supporter')
profile('Tetsuro Kitahara', {'Affiliation': 'Nihon University, Sound Computing Laboratory', 'Position': 'Professor', 'Mail': 'kitahara＠kthrlab.jp'})
