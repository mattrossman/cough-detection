from loader import load_audio_data_small, load_audio_data
import librosa
import librosa.display
import numpy as np

# hop_length = 512
# x, y = load_audio_data_small()
#
# audio = x[11]  # sample cough
# S = librosa.feature.melspectrogram(y=audio.astype(float), sr=sr, n_mels=128,
#                                    fmax=8000)
# S_dB = librosa.power_to_db(S, ref=np.max)
# librosa.display.specshow(S_dB, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel')


def get_mel_spectrogram(signal, sr):
    S = librosa.feature.melspectrogram(y=signal.astype(float), sr=sr, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)
    return S_dB


def get_features_small():
    x, y = load_audio_data_small()
    sr = 22050  # Small dataset uses 22.05K Hz
    features = np.array([get_mel_spectrogram(segment, sr=sr) for segment in x])
    return features


def get_features_large():
    x, y = load_audio_data()
    sr = 44100
    features = np.array([get_mel_spectrogram(segment, sr=sr) for segment in x])
    return features


def load_features(path):
    try:
        data = np.load(path)
        return data['arr_0'], data['arr_1']
    except FileNotFoundError:
        print('You need to serialize the data first')