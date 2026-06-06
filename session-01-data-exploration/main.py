## Session 1

#using scikit-learn to create data - make_blobs
from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt

X,y = make_blobs(
    n_samples=200,
    n_features=3,
    center_box=(5, 10),
    cluster_std=.5,
    random_state=42,
    centers=10
)


#Parameters :
# center  = numbers of group
# if you do not yse center = ? by default it consider as 3 center ( 3 groups)
# center_box = (a,b)- between a and b the data will be scattered
# random state = 42 - every time you run get same  data



print(np.unique(y))
print(y.shape)
print(X.shape)

fig, axes = plt.subplots(1,3,figsize=(10,5))


axes[0].scatter(X[:, 0], X[:, 2], c=y, cmap='viridis')
axes[0].set_xlabel("Feature 1")
axes[0].set_ylabel("Feature 3")

axes[1].scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
axes[1].set_xlabel("Feature 1")
axes[1].set_ylabel("Feature 2")

axes[2].scatter(X[:, 1], X[:, 2], c=y, cmap='viridis')
axes[2].set_xlabel("Feature 2")
axes[2].set_ylabel("Feature 3")


plt.show()

