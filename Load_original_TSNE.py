
import numpy as np

import pandas as pd
import scipy.io as sio
from sklearn.decomposition import PCA
# X= sio.loadmat(r'.\datasets\Indian_pines_corrected.mat')['indian_pines_corrected']
# labels =sio.loadmat(r'.\datasets\Indian_pines_gt.mat')['indian_pines_gt']
# X = sio.loadmat(r'.\Datasets\Salinas_corrected.mat')['salinas_corrected']
# labels = sio.loadmat(r'.\Datasets\Salinas_gt.mat')['salinas_gt']
X = sio.loadmat(r'.\Datasets\PaviaU.mat')['paviaU']
labels = sio.loadmat(r'.\Datasets\PaviaU_gt.mat')['paviaU_gt']
# X = sio.loadmat('./datasets/Houston2013_Data\Houston2013_HSI.mat')['HSI']
# y = sio.loadmat('./datasets/Houston2013_Data\Houston2013_TE.mat')['TE_map']
# y1 = sio.loadmat('./datasets/Houston2013_Data\Houston2013_TR.mat')['TR_map']
# labels=y+y1

encoding_array = []
def  SNE(X,gt):
    h, w, d = X.shape
    encoding_array = []
    list = ['liu']
    # 由于DataFrame是Pandas库中的一种数据结构，它类似excel，是一种二维表，所以需要将list以二维列表的形式转化为DataFrame
    data = pd.DataFrame([list])
    data.to_csv('PU.csv', mode='a', header=False, index=False)
    for i in range(h):
        for j in range(w):
            if int(gt[i, j]) == 0:
                continue

            else:
                list = [gt[i, j]]
                data = pd.DataFrame([list])
                data.to_csv('PU.csv', mode='a', header=False, index=False)
                image_patch = X[i, j:j+1, :]
                encoding_array.append(image_patch)
    encoding_array = np.array(encoding_array)
    encoding_array = np.matrix(encoding_array)
    print(encoding_array.shape)
    np.save('PU.npy', encoding_array)
SNE(X,labels)


