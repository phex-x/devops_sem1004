from scipy.cluster.vq import kmeans
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

#24.04
kmeans = KMeans(n_clusters=4, init='k-means++', random_state=0).fit(df)

print(kmeans)
print(kmeans.cluster_centers_)
print(kmeans.inertia_)
print(kmeans.n_iter_)

from collections import Counter
Counter(kmeans.labels_)
print(Counter(kmeans.labels_))

import seaborn as sns
import matplotlib.pyplot as plt
sns.scatterplot(data=df, x="var1", y="var2", hue=kmeans.labels_)
plt.savefig('3.png')
plt.clf()

sns.scatterplot(data=df, x="var1", y="var2", hue=kmeans.labels_)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c="r", s=80, label="centroids")
plt.legend()
plt.savefig('4.png')

#задание по вариантам
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt

# 1. Генерация данных
np.random.seed(42)

n = 100  # количество записей

data = {
    'param1': np.random.uniform(0.01, 1, n),
    'param2': np.random.randint(1, 301, n),
    'city': np.random.choice(['Самара', 'Тольятти', 'Чапаевск'], n)
}

df = pd.DataFrame(data)

print("=== Исходные данные (первые 5 строк) ===")
print(df.head())
print()

# 2. Кодирование городов
le = LabelEncoder()
df['city_encoded'] = le.fit_transform(df['city'])

print("=== Кодировка городов ===")
for city, code in zip(le.classes_, le.transform(le.classes_)):
    print(f"{city} -> {code}")
print()

# 3. Подготовка данных
X = df[['param1', 'param2', 'city_encoded']]

# 4. Масштабирование
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("=== Пример масштабированных данных (первые 5 строк) ===")
print(X_scaled[:5])
print()

# 5. Кластеризация
kmeans = KMeans(n_clusters=3, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)

print("=== Результаты кластеризации (первые 5 строк) ===")
print(df.head())
print()

# 6. Центры кластеров
print("=== Центры кластеров ===")
print(kmeans.cluster_centers_)
print()

# 7. Количество объектов в каждом кластере
print("=== Размеры кластеров ===")
print(df['cluster'].value_counts())
print()

# 8. Сохранение графика
plt.figure()
plt.scatter(df['param1'], df['param2'], c=df['cluster'])
plt.xlabel('param1')
plt.ylabel('param2')
plt.title('Кластеризация KMeans')
plt.savefig("clusters.png")  # сохраняем в файл
plt.close()

print("График сохранён в файл clusters.png")