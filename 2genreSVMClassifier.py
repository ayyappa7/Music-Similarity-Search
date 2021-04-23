import sklearn.datasets
from sklearn import linear_model
from sklearn.externals import joblib
import numpy as np
import matplotlib.pyplot as plt
def normalize(Mfcc):
    scaler = sklearn.preprocessing.StandardScaler()
    return scaler.fit_transform(Mfcc)
x_train=np.load("D:/MSS/features/Mfcc_train.npy")
y_train=np.load("D:/MSS/features/output_train.npy")
x_test=np.load("D:/MSS/features/Mfcc_test.npy")
y_test=np.load("D:/MSS/features/output_test.npy")

# clf=sklearn.svm.SVC()
# clf.fit(x_train[:2600], y_train[:2600])
# joblib.dump(clf,"D/MSS/Model/svm.sav")
x=np.zeros(shape=(0,13))
y=np.zeros(shape=(0))
xt=np.zeros(shape=(0,13))
yt=np.zeros(shape=(0))
x=np.concatenate((x,np.transpose(x_train[0])),axis=0)
x=np.concatenate((x,np.transpose(x_train[100])),axis=0)
y=np.concatenate((y,np.repeat(0,np.size(x_train[0],axis=1))))
y=np.concatenate((y,np.repeat(1,np.size(x_train[100],axis=1))))
# xt=np.concatenate((xt,np.transpose(x_test[0])),axis=0)
# xt=np.concatenate((xt,np.transpose(x_test[10])),axis=0)
# yt=np.concatenate((yt,np.repeat(0,np.size(x_test[0],axis=1))),axis=0)
# yt=np.concatenate((yt,np.repeat(1,np.size(x_test[10],axis=1))),axis=0)
xt=np.transpose(x_test[15])
x=normalize(x)
xt=normalize(xt)
clf=sklearn.svm.SVC()
clf.fit(x, y)
yp=clf.predict(xt)
print(yp)
print("count 0=",(yp==0).sum())
print("count 1=",(yp==1).sum())

