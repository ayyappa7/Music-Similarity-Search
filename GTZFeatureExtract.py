import os,sklearn,librosa,numpy as np
def normalize(Mfcc):
    scaler = sklearn.preprocessing.StandardScaler()
    return scaler.fit_transform(Mfcc)
def shapeFit(arr):
    if np.size(arr)<16796:
        len=16796-np.size(arr)
        arr=np.pad(arr,(0,len),mode="wrap")
        return arr
    return arr

root="D:/MSS/genresInWav/";
os.chdir(root);
outputClass=0
Mfcc_train=np.zeros(shape=(900,16796))
Mfcc_test=np.zeros(shape=(100,16796))
out_train=np.zeros(shape=(900))
out_test=np.zeros(shape=(100))
xtr=0
xts=0
ytr=0
yts=0
outputClass=0
for genre_dir in os.scandir(root):
    os.chdir(genre_dir)
    sampleNo=0
    for file in os.listdir(genre_dir):
        y,sr=librosa.load(file,duration=30)
        Mfcc=librosa.feature.mfcc(y,sr,n_mfcc=13)
        Mfcc=np.transpose(Mfcc)
        MfccInaRow=Mfcc.flatten()
        MfccInaRow=shapeFit(MfccInaRow)
        if sampleNo<90:
            Mfcc_train[xtr]=MfccInaRow
            xtr+=1
            out_train[ytr]=outputClass
            ytr+=1
            print("Done",str(file),"in train")
        else:
            Mfcc_test[xts]=MfccInaRow
            xts+=1
            out_test[yts]=outputClass
            yts+=1
            print("Done", str(file), "in test")
        sampleNo+=1
    outputClass+=1
np.save("D:\MSS\\features\\Mfcc_train.npy",Mfcc_train)
np.save("D:\MSS\\features\\Mfcc_test.npy",Mfcc_test)
np.save("D:\MSS\\features\\output_train.npy",out_train)
np.save("D:\MSS\\features\\output_test.npy",out_test)
