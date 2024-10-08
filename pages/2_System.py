import streamlit as st

st.set_page_config('EgGMAn', ':musical_note:', 'wide')
st.sidebar.link_button('Contact Us', 'https://forms.gle/A4vWuEAp4pPEY4sf9', use_container_width=True)

def iframe(u, w, h):
    st.markdown(f'<iframe src="{u}/preview" width="{w}" height="{h}"></iframe>', True)

st.title('System')

iframe('https://drive.google.com/file/d/1a167-_xiTrvA9JtiEfI8RST_T8996Stu', 700, 420)

st.subheader('Source Music')
st.markdown('- Convert Source Music to vector ___z___')

l, r = st.columns(2, gap='medium')

with l:
    st.subheader('Source Scene')
    st.markdown('- Create a set of music to use in the same scene as Source Scene\n- Convert a set of music to a set of vector\n- Compute the center ___p___ of a set of vector')

with r:
    st.subheader('Target Scene')
    st.markdown('- Create a set of music to use in the same scene as Target Scene\n- Convert a set of music to a set of vector\n- Compute the center ___q___ of a set of vector')

st.subheader('Target Music')
st.markdown("- Compute vector ___z'___ by moving vector ___z___ toward ___q___ - ___p___\n- Compute the distance of vector ___z'___ and each music vector\n- Show music in order of distance")
