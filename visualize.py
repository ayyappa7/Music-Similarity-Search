import numpy as np
import librosa.display
import matplotlib.pyplot as plt
import sklearn
def spectogram(y,sr):
    S = librosa.feature.melspectrogram(y, sr=sr)
    Sdb = librosa.amplitude_to_db(S)
    librosa.display.specshow(Sdb, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar()
    plt.show()

def normalize(Mfcc):
    scaler = sklearn.preprocessing.StandardScaler()
    return scaler.fit_transform(Mfcc)
def featurePlot(Mfcc):
    plt.plot(Mfcc)
    plt.show()

# x_train=np.load("D:/MSS/features/Mfcc_train.npy")
# y_train=np.load("D:/MSS/features/output_train.npy")
# x_test=np.load("D:/MSS/features/Mfcc_test.npy")
# y_test=np.load("D:/MSS/features/output_test.npy")
y,sr=librosa.load("D:/MSS/genresInWav/blues/blues.00001.wav",duration=30)
spectogram(y,sr)
Mfcc=np.transpose(librosa.feature.mfcc(y,sr))
Mfcc=normalize(Mfcc)
featurePlot(Mfcc)
y,sr=librosa.load("D:/MSS/genresInWav/blues/blues.00001.wav",sr=10000,duration=30)
spectogram(y,sr)
Mfcc=np.transpose(librosa.feature.mfcc(y,sr))
Mfcc=normalize(Mfcc)
featurePlot(Mfcc)
# y,sr=librosa.load("D:/MSS/genresInWav/classical/classical.00001.wav",duration=30)
# visualize(y,sr)

