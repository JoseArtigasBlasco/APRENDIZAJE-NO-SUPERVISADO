# Realizamos un aprendizaje no supervisado utilizando el algoritmo K-means para la agrupación de datos.
# Vamos a utilizar el conjunto de datos Iris. Este conjunto de datos contiene 150 muestras
# de iris con 4 características cada una.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

iris = load_iris()
data = iris.data
target = iris.target

# Estandarizamos los datos
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Aplicamos el algoritmo K-means
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(data_scaled)
labels = kmeans.labels_

# Reducimos la dimensionalidad para visualización
pca = PCA(n_components=2)
data_pca = pca.fit_transform(data_scaled)


# Graficamos datos
plt.figure(figsize=(10, 6))
plt.scatter(data_pca[:, 0], data_pca[:, 1], c=labels, cmap='viridis', marker='o', edgecolor='k')
plt.xlabel('PCA Componente 1')
plt.ylabel('PCA Componente 2')
plt.title('Clustering de K-means en el conjunto de datos Iris')
plt.colorbar(label='Cluster')
plt.show()



# Predicción para una nueva observación

new_observation = np.array([[5.0, 3.6, 1.4, 0.2]])

# Estandarizamos utilizando el mismo escalador
new_observation_scaled = scaler.transform(new_observation)

# Predecimos el cluster para la nueva observación
predicted_cluster = kmeans.predict(new_observation_scaled)
print(f'La nueva observación pertenece al cluster: {predicted_cluster[0]}')


