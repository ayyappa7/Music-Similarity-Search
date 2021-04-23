from scipy.spatial import distance
import numpy as np
genre=1
mfcc_train=np.load("D:/MSS/features/Mfcc_train.npy")
mfcc_test=np.load("D:/MSS/features/Mfcc_test.npy")
mfcc=np.zeros((0,16796))
for i in range(10):
    if i == genre:
        mfcc=np.concatenate((mfcc,mfcc_train[i*90:i*90+90]),axis=0)
        # mfcc=np.concatenate((mfcc,mfcc_test[i*10:i*10+10]),axis=0)
print(mfcc.shape)
song=mfcc_test[13]
mindist=100000
matched=100000
i=0
distances=[]
for i in range(90):
    distances=np.append(distances,distance.euclidean(mfcc[i],song))
print(distances)
leastFive=distances.argsort()[:5]
print(leastFive)



