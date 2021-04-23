import os
root="D:/MSS/genresInWav/";
os.chdir(root);
for genre_dir in os.scandir(root):
    os.chdir(genre_dir)
    for file in os.listdir(genre_dir):
        os.system("sox " + str(file) + " " + str(file[:-3]) + ".wav")
    os.system("del /f *.au")
# y, sr = lib.load("D:\MSS\genresInWav\\blues.00001.wav");
# Mfcc=lib.feature.mfcc(y=y, sr=sr,n_mfcc=13)
# print(Mfcc)
# print(np.size(Mfcc,axis=0))

