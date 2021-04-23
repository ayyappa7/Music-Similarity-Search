import sklearn,sklearn.neural_network,pickle,os
import numpy as np
def normalize(Mfcc):
    scaler = sklearn.preprocessing.StandardScaler()
    return scaler.fit_transform(Mfcc)
x=np.load("D:/MSS/features/Mfcc_train.npy")
y=np.load("D:/MSS/features/output_train.npy")
x=normalize(x)
clf=sklearn.neural_network.MLPClassifier(hidden_layer_sizes=(1000,250),max_iter=10000,learning_rate_init=0.0001)
print("training Started")
clf.fit(x,y)
print("Training Done")
pickle.dump(clf,open("NN.sav","wb"))
print("dumped Model")
