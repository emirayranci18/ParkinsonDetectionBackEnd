import librosa
import numpy as np
import parselmouth
from parselmouth.praat import call, run_file
from pyEDM import EmbedDimension
from scipy.stats import entropy
from nolds import dfa
from pydub import AudioSegment

import wave
import struct

wavefile = wave.open('recordingParkinsonPulled.wav', 'r')

length = wavefile.getnframes()
for i in range(0, length):
    wavedata = wavefile.readframes(1)
    data = struct.unpack("<h", wavedata)

audio_path = "recordingParkinsonPulled.wav"
y, sr = librosa.load(audio_path)
sound = parselmouth.Sound(audio_path)


f0, voiced_flag, _ = librosa.pyin(y, fmin=75, fmax=300)
voiced_f0 = f0[voiced_flag > 0]
avg_f0 = voiced_f0.mean()

pitches, magnitudes = librosa.core.piptrack(y=y, sr=sr, fmin=75, fmax=300)
max_pitch_idx = magnitudes[:, 0].argmax()
max_pitch = pitches[max_pitch_idx, 0]
min_pitch = pitches[pitches > 0].min()


pitch = librosa.core.pitch_tuning(y)
jitter_percent = np.std(pitch) / np.mean(pitch) * 100
jitter_abs = np.abs(pitch).mean()

frame_length = int(sr * 0.03)
frames = librosa.util.frame(y, frame_length=frame_length, hop_length=frame_length)
apq_values = []
for frame in frames:
    apq = sum(frame**2) / len(frame)
    apq_values.append(apq)
mdvp_apq = sum(apq_values) / len(apq_values)



pitch = call(sound, "To Pitch", 0.0, 75, 300)
harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
pointProcess = call(pitch, "To PointProcess")
localJitter = call(pointProcess, "Get jitter (local)", 0, 0, 0.001, 0.03, 1.3)
localabsoluteJitter = call(pointProcess, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)
localabsoluteJitter=format(localabsoluteJitter, ".0e")
rapJitter = call(pointProcess, "Get jitter (rap)", 0, 0, 0.00068, 0.021, 1.3)
ppq5Jitter = call(pointProcess, "Get jitter (ppq5)", 0, 0, 0.0009, 0.02, 1.3)
ddpJitter = call(pointProcess, "Get jitter (ddp)", 0, 0, 0.002, 0.064, 1.3)
local_shimmer = call([sound, pointProcess], "Get shimmer (local)", 0, 0, 0, 300, 1.3, 1.6)
local_db_shimmer = call([sound, pointProcess], "Get shimmer (local_dB)", 0, 0, 0, 300, 1.3, 1.6)
apq3Shimmer = call([sound, pointProcess], "Get shimmer (apq3)", 0, 0, 0.004, 0.05, 1.3, 1.6)
aqpq5Shimmer = call([sound, pointProcess], "Get shimmer (apq5)", 0, 0, 0.005, 0.079, 1.3, 1.6)
ddaShimmer = call([sound, pointProcess], "Get shimmer (dda)", 0, 0, 0, 300, 1.3, 1.6)
hnr = call(harmonicity, "Get mean", 0, 0)

print("Average vocal fundamental frequency (MDVP:Fo(Hz)): {:.2f}".format(avg_f0))
print("Maximum vocal fundamental frequency (MDVP:Fhi(Hz)): {:.2f}".format(max_pitch))
print("Minimum vocal fundamental frequency (MDVP:Flo(Hz)): {:.2f}".format(min_pitch))
print("jitter_percent:", localJitter)
print("localabsoluteJitter:", localabsoluteJitter)
print("rapJitter:", rapJitter)
print("ppq5Jitter:", ppq5Jitter)
print("ddpJitter:", ddpJitter)
print("localShimmer:", local_shimmer)
print("localdbShimmer:", local_db_shimmer)
print("apq3Shimmer:", apq3Shimmer)
print("aqpq5Shimmer:", aqpq5Shimmer)
print("ddaShimmer:", ddaShimmer)
print("HNR:", hnr)


