import sklearn,pickle,sklearn.metrics
import librosa as lib
import numpy as np
import os
def normalize(Mfcc):
    scaler = sklearn.preprocessing.StandardScaler().fit(Mfcc)
    return scaler.fit_transform(Mfcc)

xt=np.load("D:/MSS/features/Mfcc_test.npy")
yt=np.load("D:/MSS/features/output_test.npy")

xt=normalize(xt)
clf=pickle.load(open('NN.sav','rb'))
yp=clf.predict(xt)
print("score NN=",clf.score(xt,yt))
print(sklearn.metrics.confusion_matrix(yt,yp))
yprobNN=clf.predict_proba(xt)
# print(yprobNN)

clf=pickle.load(open('LR.sav','rb'))
yp=clf.predict(xt)
print("score LR=",clf.score(xt,yt))
print(sklearn.metrics.confusion_matrix(yt,yp))
yprobLR=clf.predict_proba(xt)
# print(yprobLR)

clf=pickle.load(open('SVM.sav','rb'))
yp=clf.predict(xt)
print("score SVM=",clf.score(xt,yt))
print(sklearn.metrics.confusion_matrix(yt,yp))
yprobSVM=clf.predict_proba(xt)
# print(yprobSVM)
yprob=np.array(0.3*yprobLR+0.2*yprobNN+0.5*yprobSVM)
yprob=yprob/3
print("+++++++++++++++++++++++++++++++++++++++++++++++")
yprob=yprob.reshape((100,10))
# print(yprob)
yprob=np.argmax(yprob,axis=1)
# print(yprob)
print(sklearn.metrics.confusion_matrix(yt,yprob))

