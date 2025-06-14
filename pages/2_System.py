import streamlit as st

st.set_page_config('EgGMAn', ':musical_note:', 'wide')
st.sidebar.link_button('Contact Us', 'https://forms.gle/A4vWuEAp4pPEY4sf9', use_container_width=True)

def iframe(u, w, h):
    st.markdown(f'<iframe src="{u}/preview" width="{w}" height="{h}"></iframe>', True)

st.title('System')

iframe('https://drive.google.com/file/d/1a167-_xiTrvA9JtiEfI8RST_T8996Stu', 700, 420)

st.subheader('Source Music')
st.markdown('- Extract vector $z_p$ from Source Music')

l, r = st.columns(2, gap='small')

with l:
    st.subheader('Source Scene')
    st.markdown('- Create a set of music used in the same scene as Source Scene\n- Extract a set of vectors from the set of music\n- Compute vector $c_p$ of Source Scene from the center of the set of vectors')

with r:
    st.subheader('Target Scene')
    st.markdown('- Create a set of music used in the same scene as Target Scene\n- Extract a set of vectors from the set of music\n- Compute vector $c_q$ of Target Scene from the center of the set of vectors')

st.subheader('Target Music')
st.markdown('- Predict vector $z_q$ of Target Music from $z_p, c_p, c_q$\n - Compute the distance from $z_q$ to each music vector\n- Sort music in ascending order by distance')
