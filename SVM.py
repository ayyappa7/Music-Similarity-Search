import sklearn.datasets
from sklearn import linear_model
from sklearn.externals import joblib
import pickle,numpy as np
import matplotlib.pyplot as plt
def normalize(Mfcc):
    scaler = sklearn.preprocessing.StandardScaler().fit(Mfcc)
    return scaler.fit_transform(Mfcc)
x=np.load("D:/MSS/features/Mfcc_train.npy")
y=np.load("D:/MSS/features/output_train.npy")
print(x.shape)
print(y.shape)
# print(x_train[0].shape)
# for i in range (900):
#         x=np.concatenate((x,np.transpose(x_train[i])),axis=0)
#         y=np.concatenate((y,i/100))
x=normalize(x)
clf=sklearn.svm.SVC(kernel='rbf',probability=True)
print("training Started")
clf.fit(x,y)
print("training end")
pickle.dump(clf,open("SVM.sav",'wb'))
print("dumped model")