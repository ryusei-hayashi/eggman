import streamlit as st

st.set_page_config('EgGMAn', ':musical_note:', 'wide')
st.sidebar.link_button('Contact Us', 'https://forms.gle/A4vWuEAp4pPEY4sf9', use_container_width=True)

def iframe(u, w, h):
    st.html(f'<iframe src="{u}/preview" width="{w}" height="{h}"></iframe>')

st.title('Library')

st.header('Paper')
iframe('https://drive.google.com/file/d/18H8dnbhL-D53BUWb09nGVWGWyot06OM6', 700, 990)

st.header('Slide')
iframe('https://drive.google.com/file/d/1MIN0am-lZV7TqGx20CjgG1CSQXyLkgOO', 700, 420)
