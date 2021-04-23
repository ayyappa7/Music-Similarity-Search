import sklearn
from sklearn import linear_model
from sklearn.externals import joblib
import pickle,numpy as np
import matplotlib.pyplot as plt
def normalize(Mfcc):
    scaler = sklearn.preprocessing.StandardScaler()
    return scaler.fit_transform(Mfcc)
x=np.load("D:/MSS/features/Mfcc_train.npy")
y=np.load("D:/MSS/features/output_train.npy")
print(x.shape)
print(y.shape)
x=normalize(x)
clf=sklearn.linear_model.LogisticRegression()
print("training Started")
clf.fit(x,y)
print("training end")
pickle.dump(clf,open("LR.sav",'wb'))
print("dumped model")