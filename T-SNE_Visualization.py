import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

df = pd.read_csv('PU.csv')
class_list = np.unique(df['liu'])
print(class_list)
import seaborn as sns
#marker_list = [ '.','.','.','.','.','.','.','.','.','.','.','.','.','.','.','.',]
#marker_list = [ '.','.','.','.','.','.','.','.','.','.','.','.','.','.','.',]
marker_list = [ '.','.','.','.','.','.','.','.','.']
n_class = len(class_list) # 测试集标签类别数
palette = sns.hls_palette(n_class) # 配色方案
sns.palplot(palette)

encoding_array = np.load('PU.npy', allow_pickle=True)#10366.512
encoding_array=np.matrix(encoding_array )
print(encoding_array.shape)

from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, n_iter=10000)
X_tsne_2d = tsne.fit_transform(encoding_array)
print(X_tsne_2d.shape)

show_feature = 'liu'

plt.figure(figsize=(14, 14))
for idx, fruit in enumerate(class_list): # 遍历每个类别
    # 获取颜色和点型
    color = palette[idx]
    marker = marker_list[idx%len(marker_list)]

    # 找到所有标注类别为当前类别的图像索引号
    indices = np.where(df[show_feature]==fruit)
    plt.scatter(X_tsne_2d[indices, 0], X_tsne_2d[indices, 1], color=color, marker=marker)

plt.legend(fontsize=16, markerscale=1, bbox_to_anchor=(1, 1))
plt.xticks([])
plt.yticks([])
plt.savefig('t-SNEPU.png', dpi=300) # 保存图像
plt.savefig('t-SNEPU.pdf', dpi=300)
plt.show()
plt.pause(30)