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
  import numpy as np
  import pandas as pd
  from sklearn.cluster import KMeans
  from sklearn.preprocessing import LabelEncoder, StandardScaler
  import matplotlib.pyplot as plt
  import seaborn as sns
  from collections import Counter

  # =========================
  # 1. Генерация данных
  # =========================
  np.random.seed(42)
  n = 100

  data = {
      'param1': np.random.uniform(0.01, 1, n),
      'param2': np.random.randint(1, 301, n),
      'city': np.random.choice(['Самара', 'Тольятти', 'Чапаевск'], n)
  }

  df = pd.DataFrame(data)

  print("=== Исходные данные (первые 5 строк) ===")
  print(df.head())
  print()

  # =========================
  # 2. Кодирование категорий
  # =========================
  le = LabelEncoder()
  df['city_encoded'] = le.fit_transform(df['city'])

  print("=== Кодировка городов ===")
  for city, code in zip(le.classes_, le.transform(le.classes_)):
      print(f"{city} -> {code}")
  print()

  # =========================
  # 3. Подготовка данных
  # =========================
  X = df[['param1', 'param2', 'city_encoded']]

  # Масштабирование
  scaler = StandardScaler()
  X_scaled = scaler.fit_transform(X)

  print("=== Масштабированные данные (первые 5 строк) ===")
  print(X_scaled[:5])
  print()

  # =========================
  # 4. KMeans
  # =========================
  kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
  df['cluster'] = kmeans.fit_predict(X_scaled)

  print("=== Результат кластеризации (первые 5 строк) ===")
  print(df.head())
  print()

  # =========================
  # 5. Основные характеристики
  # =========================
  print("=== inertia (сумма квадратов расстояний) ===")
  print(kmeans.inertia_)
  print()

  print("=== Количество итераций ===")
  print(kmeans.n_iter_)
  print()

  print("=== Размеры кластеров ===")
  print(Counter(kmeans.labels_))
  print()

  # =========================
  # 6. Центроиды
  # =========================
  print("=== Центроиды (в масштабированном виде) ===")
  print(kmeans.cluster_centers_)
  print()

  # Обратное преобразование
  centroids_original = scaler.inverse_transform(kmeans.cluster_centers_)

  print("=== Центроиды (в исходных параметрах) ===")
  for i, center in enumerate(centroids_original):
      param1 = center[0]
      param2 = int(center[1])
      city_code = int(round(center[2]))
      city_name = le.inverse_transform([city_code])[0]

      print(f"Кластер {i}:")
      print(f"  param1 = {param1:.4f}")
      print(f"  param2 = {param2}")
      print(f"  city   = {city_name}")
      print()

  # =========================
  # 7. График 1 (без центроидов)
  # =========================
  plt.figure()
  sns.scatterplot(data=df, x="param1", y="param2", hue="cluster")
  plt.title("Кластеры")
  plt.savefig("clusters.png")
  plt.close()

  print("График clusters.png сохранён")

  # =========================
  # 8. График 2 (с центроидами)
  # =========================
  centroids_x = centroids_original[:, 0]
  centroids_y = centroids_original[:, 1]

  plt.figure()
  sns.scatterplot(data=df, x="param1", y="param2", hue="cluster")

  plt.scatter(centroids_x, centroids_y,
              c="red", s=120, marker="X", label="centroids")

  plt.legend()
  plt.title("Кластеры с центроидами")

  plt.savefig("clusters_with_centroids.png")
  plt.close()

  print("График clusters_with_centroids.png сохранён")