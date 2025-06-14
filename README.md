# EgGMAn
- This repository is for Deploying [EgGMAn (Engine of Game Music Analysis)](https://eggman.streamlit.app)
- EgGMAn searches for game music considering game-wise consistency and scene-wise individuality

## Usage
### Source Music
- Enter the music prepared for the developing game
  - Web Service: Enter the URL of these sites
  - Direck Link: Enter the URL of the audio file
  - Audio file: Enter the uploaded audio file

### Source Scene
- Enter the scene that uses the prepared music combining tags
- There are about 200 tags such as _Opening_, _Dungeon_, ...

#### Mood of Source Scene
- Enter the mood of the scene with Valence-Arousal
  - Valence: Rises in positive scenes and falls in negative scenes
  - Arousal: Rises in active scenes and falls in passive scenes

### Traget Scene
- Enter the scene that uses the searched music combining tags
- There are about 200 tags such as _Opening_, _Dungeon_, ...

#### Mood of Target Scene
- Enter the mood of the scene with Valence-Arousal
  - Valence: Rises in positive scenes and falls in negative scenes
  - Arousal: Rises in active scenes and falls in passive scenes

### Target Music
- EgGMAn searches for game music of Target Scene in the developing game
- Without Source Music, EgGMAn randomly searches for game music of Target Scene
  - Ignore Artist: Enter the artist not to be listed
  - Ignore Site: Enter the site not to be listed
  - Time Range: Enter the time of music to be listed
  - Random Rate: Enter the ratio to reflect music distribution

## System
### Source Music
- Convert Source Music to vector __z__

### Source Scene
- Create a set of music to use in the same scene as Source Scene
- Convert a set of music to a set of vector
- Compute the center __p__ of a set of vector

### Target Scene
- Create a set of music to use in the same scene as Target Scene
- Convert a set of music to a set of vector
- Compute the center __q__ of a set of vector

### Target Music
- Compute vector __z'__ by moving vector __z__ toward __q__ - __p__
- Compute the distance of vector __z'__ and each music vector
- Show music in order of distance

## Requirement
* [tensorflow-probability](https://www.tensorflow.org/probability)
* [tensorflow](https://www.tensorflow.org)
* [streamlit](https://streamlit.io)
* [essentia](https://essentia.upf.edu)
* [requests](https://requests.readthedocs.io)
* [librosa](https://librosa.org)
* [yt_dlp](https://github.com/yt-dlp/yt-dlp)
* [gdown](https://github.com/wkentaro/gdown)
* [numpy](https://numpy.org)

## Licence
* [MIT License](https://en.wikipedia.org/wiki/MIT_License)
