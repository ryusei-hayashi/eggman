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
- Extract vector $z_p$ from Source Music

### Source Scene
- Create a set of music used in the same scene as Source Scene
- Extract a set of vectors from the set of music
- Compute vector $c_p$ of Source Scene from the center of the set of vectors

### Target Scene
- Create a set of music used in the same scene as Target Scene
- Extract a set of vectors from the set of music
- Compute vector $c_q$ of Target Scene from the center of the set of vectors

### Target Music
- Predict vector $z_q$ of Target Music from $z_p, c_p, c_q$ 
- Compute the distance from $z_q$ to each music vector
- Sort music in ascending order by distance

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
