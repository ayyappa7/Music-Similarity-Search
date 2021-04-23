import os,librosa,numpy as np
root="D:\MSS\TATGenreInWav\\";
os.chdir(root);
for genre_dir in os.scandir(root):
    os.chdir(genre_dir)
    for file in os.listdir(genre_dir):
        print(str(file))
        os.system("ffmpeg -i " + str(file) + " " + str(file[:-4]) + ".wav")
    os.system("del /f *.mp3")
# file=open("D:\MSS\TagATuneGenreInWav\\0\\barbara_leoni-human_needs-01-dont_rain_on_my_parade-175-204.mp3")
# y, sr = librosa.load(file);
# Mfcc=librosa.feature.mfcc(y=y, sr=sr,n_mfcc=13)
# print(Mfcc)
# print(np.size(Mfcc,axis=0))

