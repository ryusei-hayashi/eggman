# EgGMAn
- This repository is for Deploying [EgGMAn](https://eggman.streamlit.app) (Engine of Game Music Analysis)
- EgGMAn (Engine of Game Music Analysis) search for game music considering game and scene feature at the same time

## Usage
### Source Music
- Enter the URL of the music to use in developing game
- URL support Spotify, SoundCloud, YouTube, Hotlink

### Source Scene
- Enter the scene where Source Music is used
- There are about 200 scenes including "Opening", "Spring", etc

### Target Scene
- Enter the scene where Target Music is used
- There are about 200 scenes including "Opening", "Spring", etc

## System
### Source Music
- Convert Source Music to vector ___z___

### Source Scene
- Create a set of music to use in the same scene as Source Scene
- Convert a set of music to a set of vector
- Compute the center ___p___ of a set of vector

### Target Scene
- Create a set of music to use in the same scene as Target Scene
- Convert a set of music to a set of vector
- Compute the center ___q___ of a set of vector

### Target Music
- Compute vector ___z'___ by moving vector ___z___ toward ___q___ - ___p___
- Compute the distance of vector ___z'___ and each music vector
- Show music in order of distance

## Requirement
* [tensorflow-probability](https://www.tensorflow.org/probability)
* [tensorflow](https://www.tensorflow.org)
* [statistics](https://docs.python.org/3/library/statistics.html)
* [streamlit](https://streamlit.io)
* [requests](https://requests.readthedocs.io)
* [spotipy](https://spotipy.readthedocs.io)
* [librosa](https://librosa.org)
* [yt-dlp](https://github.com/yt-dlp/yt-dlp)
* [pandas](https://pandas.pydata.org)
* [numpy](https://numpy.org)

## Licence
* [MIT License](https://en.wikipedia.org/wiki/MIT_License)
