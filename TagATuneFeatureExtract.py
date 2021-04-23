import os,librosa,numpy as np
root="D:\MSS\TATGenreInWav\\";
os.chdir(root);
Mfcc_train_tat=np.zeros((0,16237 ))
out_train_tat=[]
outClass=0
for genre_dir in os.scandir(root):
    os.chdir(genre_dir)
    i=0
    for file in os.listdir(genre_dir):
        if i<500:
            y,sr=librosa.load(file,duration=29)
            Mfcc=librosa.feature.mfcc(y=y,sr=sr,n_mfcc=13)
            Mfcc = np.transpose(Mfcc)
            MfccInaRow = Mfcc.flatten()
            # print(MfccInaRow.shape)
            Mfcc_train_tat=np.vstack((Mfcc_train_tat,MfccInaRow))
            out_train_tat=np.append(out_train_tat,outClass)
            i+=1
            print("Done "+str(file))
            # print(Mfcc_train_tat.shape)
            # print(out_train_tat)
        else :
            break
    outClass+=1
    print(Mfcc_train_tat.shape)
np.save("D:\MSS\\features\\Mfcc_train_tat.npy",Mfcc_train_tat)
np.save("D:\MSS\\features\\out_train_tat.npy",out_train_tat)
