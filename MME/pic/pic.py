import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

which = 10

a = np.load('%d.npy' % which)
pca = PCA(n_components = 2)
pca.fit(a)
a_2 = pca.transform(a)
b = np.load('%d_label.npy' % which)

show_num = 1000

plt.scatter(a_2[:show_num, 0], a_2[:show_num, 1], c=b[:show_num])
plt.show()