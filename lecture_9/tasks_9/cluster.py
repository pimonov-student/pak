import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import OPTICS
from sklearn.cluster import Birch
from sklearn.cluster import MeanShift

data = pd.read_csv('wells_info.csv')
class_count = len(np.unique(data['StateName']))
coordinates = np.array(data[['LatWGS84', 'LonWGS84']])

fig, axs = plt.subplots(2, 2)


kmeans_model = KMeans(n_clusters=class_count)
kmeans_res = kmeans_model.fit(coordinates)

optics_model = OPTICS(min_samples=class_count)
optics_res = optics_model.fit(coordinates)

birch_model = Birch(n_clusters=class_count)
birch_res = birch_model.fit(coordinates)

meanshift_model = MeanShift(min_bin_freq=class_count)
meanshift_res = meanshift_model.fit(coordinates)


axs[0, 0].set_title("KMeans")
axs[0, 0].scatter(coordinates[:, 0], coordinates[:, 1], c=kmeans_res.labels_, cmap='rainbow')
axs[0, 1].set_title("OPTICS")
axs[0, 1].scatter(coordinates[:, 0], coordinates[:, 1], c=optics_res.labels_, cmap='rainbow')
axs[1, 0].set_title("BIRCH")
axs[1, 0].scatter(coordinates[:, 0], coordinates[:, 1], c=birch_res.labels_, cmap='rainbow')
axs[1, 1].set_title("MeanShift")
axs[1, 1].scatter(coordinates[:, 0], coordinates[:, 1], c=meanshift_res.labels_, cmap='rainbow')

plt.show()
