import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot([1,2,3,4], [1,4,2,3])
plt.ylabel("some_numbers")
plt.savefig("1.png")

from sklearn.datasets import make_blobs
import pandas as pd

dataset, classes = make_blobs(
  n_samples=200, centers=4, n_features=2, cluster_std=0.5, random_state=0
)
df = pd.DataFrame(dataset, columns=["var1", "var2"])
print(df.head(2))
plt.clf()

from sklearn.cluster import KMeans

inertias = []
k_range = range(1, 11)
for k in k_range:
  model = KMeans(n_clusters=k, random_state=0, n_init="auto")
  model.fit(df)
  inertias.append(model.inertia_)

fig, ax = plt.subplots()
ax.plot(list(k_range), inertias, marker="o")
ax.set_xlabel('Number of clusters (k)')
ax.set_ylabel('Inertia')
ax.set_title("Elbow Method")
plt.savefig("2.png")
plt.clf()
