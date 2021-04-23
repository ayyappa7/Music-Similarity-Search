import sklearn.datasets
from sklearn import linear_model
from sklearn.externals import joblib
import pickle
import numpy as np
import matplotlib.pyplot as plt
def normalize(Mfcc):
    scaler = sklearn.preprocessing.StandardScaler()
    return scaler.fit_transform(Mfcc)
x_train=np.load("D:/MSS/features/Mfcc_train.npy")
y_train=np.load("D:/MSS/features/output_train.npy")
# x_test=np.load("D:/MSS/features/Mfcc_test.npy")
# y_test=np.load("D:/MSS/features/output_test.npy")
x_train=normalize(x_train)
clf=sklearn.svm.SVC()
print("started Training")
clf.fit(x_train[:270], y_train[:270])
pickle.dump(clf,open('svm3g.sav','wb'))
print("training Done")


