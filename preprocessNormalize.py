import librosa as lib
import numpy as np
import os,sklearn.preprocessing
from sklearn.linear_model import LogisticRegression
def normalize(Mfcc):
    scaler = sklearn.preprocessing.StandardScaler()
    return scaler.fit_transform(Mfcc)

Mfcc_train = np.zeros(shape=(0,13))
output_train = np.zeros(0)
Mfcc_test = np.zeros(shape=(0,13))
output_test = np.zeros(0)
root="D:/MSS/genresInWav/";
os.chdir(root);
outputClass=0
out=open("D:/MSS/sample.txt","w")
for genre_dir in os.scandir(root):
    os.chdir(genre_dir)
    sampleNo=0
    for file in os.listdir(genre_dir):
        y, sr = lib.load(file,duration=30);
        Mfcc = lib.feature.mfcc(y=y, sr=sr,n_mfcc=13)
        Mfcc = np.transpose(Mfcc)
        Mfcc=normalize(Mfcc)
        output=np.repeat(outputClass,np.size(Mfcc,axis=0))
        if sampleNo<90:
            Mfcc_train=np.concatenate((Mfcc_train,Mfcc),axis=0)
            output_train=np.concatenate((output_train,output),axis=0)
            print("Done "+str(file)+" in train")
        else :
            Mfcc_test=np.concatenate((Mfcc_test, Mfcc), axis=0)
            output_test=np.concatenate((output_test, output), axis=0)
            print("Done " + str(file) + " in test")
        sampleNo+=1;
    outputClass+=1

np.save("D:\MSS\\features\\train.npy",Mfcc_train)
np.save("D:\MSS\\features\\test.npy",Mfcc_test)
np.save("D:\MSS\\features\\out_train.npy",output_train)
np.save("D:\MSS\\features\\out_test.npy",output_test)


