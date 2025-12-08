# pylint: disable=E1101

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.utils import Bunch
import numpy as np

iris: Bunch = load_iris()  # type: ignore
x = iris.data
y = iris.target

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

markers = ["o", "^", "s"]

for cls in np.unique(y):
    mask = y == cls
    ax.scatter(
        x[mask, 0],
        x[mask, 1],
        x[mask, 2],
        c=x[mask, 3],
        marker=markers[cls],
        label=iris.target_names[cls],
    )

ax.set_xlabel(iris.feature_names[0])
ax.set_ylabel(iris.feature_names[1])
ax.set_zlabel(iris.feature_names[2])
plt.legend()
plt.show()
