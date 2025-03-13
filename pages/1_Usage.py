import streamlit as st

st.set_page_config('EgGMAn', ':musical_note:', 'wide')
st.sidebar.link_button('Contact Us', 'https://forms.gle/A4vWuEAp4pPEY4sf9', use_container_width=True)

def iframe(u, w, h):
    st.markdown(f'<iframe src="{u}/preview" width="{w}" height="{h}"></iframe>', True)

def scene(c, s, m):
    with c:
        st.header(s)
        st.markdown(f'- Enter the scene that uses the {m}prepared music combining tags\n- There are about 200 tags such as opening, dungeon, etc')
        st.subheader(f'Mood of {s}')
        st.markdown('- Enter the mood of the scene with Valence-Arousal')
        l, r = st.columns(2, gap='medium')
        with l:
            st.markdown('- Valence: Rises in positive scenes and falls in negative scenes')
        with r:
            st.markdown('- Arousal: Rises in active scenes and falls in passive scenes')

st.title('Usage')

iframe('https://drive.google.com/file/d/1X0O0hd9EHayuhPaUyPzXYJZljInnRtgO', 700, 420)

st.header('Source Music')
st.markdown('- Enter the music prepared for the developing game\n  - Web Service: Enter the URL of [these sites](https://github.com/yt-dlp/yt-dlp/blob/master/supportedsites.md)\n  - Direck Link: Enter the URL of the audio file\n  - Audio file: Enter the uploaded audio file')

c = st.columns(2, gap='medium')
scene(c[0], 'Source Scene', 'prepared music')
scene(c[1], 'Target Scene', 'searched music')

st.header('Target Music')
st.markdown('- EgGMAn searches for game music of Target Scene in the developing game\n- EgGMAn randomly searches for game music of Target Scene without Source Music\n  - Ignore Artist: Enter the artist not to be listed\n  - Ignore Site: Enter the site not to be listed\n  - Time Range: Enter the time of music to be listed\n  - Random Scale: Enter the ratio to reflect music distribution')
