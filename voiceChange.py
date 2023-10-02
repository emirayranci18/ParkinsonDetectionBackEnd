import pydub
sound = pydub.AudioSegment.from_wav("recordingParkinsonPulled.wav")
sound.export("recordingParkinsonPulled.mp3", format="mp3")