#Completed? y
import librosa
import numpy as np
# downloaded numpy? y
#librosa? y
def extract_mfcc(audio_path, n_mfcc=20, n_fft=2048, hop_length=256):
    y, sr = librosa.load(audio_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    return mfccs.mean(axis=1)
