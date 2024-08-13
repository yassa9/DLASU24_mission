import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

start_time = time.time()

# Loading data
X_train = np.load('../../X_train_tabular.npy')
y_train = np.load('../../y_train_tabular.npy')

# Creating feature columns for visualization
feat_cols = ['feature' + str(i) for i in range(X_train.shape[1])]
df = pd.DataFrame(X_train, columns=feat_cols)
df['label'] = np.argmax(y_train, axis=1)

print('Size of the dataframe: {}'.format(df.shape))

# Applying PCA to reduce dimensionality
pca = PCA(n_components=3)
pca_result = pca.fit_transform(df[feat_cols].values)

df['pca-one'] = pca_result[:, 0]
df['pca-two'] = pca_result[:, 1]
df['pca-three'] = pca_result[:, 2]

print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_[:3]))

# 2D PCA Plot
plt.figure(figsize=(12, 8))
sns.scatterplot(
    x='pca-one', y='pca-two',
    hue='label',
    palette=sns.color_palette("hsv", len(df['label'].unique())),
    data=df,
    legend="full",
    alpha=0.3
).set_title('2D PCA Visualization')

plt.savefig('../plots/pca_2d_visualization.png', dpi=300)

# 3D PCA Plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(
    df['pca-one'], df['pca-two'], df['pca-three'],
    c=df['label'], cmap='hsv', s=5, alpha=0.7
)
ax.set_title('3D PCA Visualization')
ax.set_xlabel('PCA Component 1')
ax.set_ylabel('PCA Component 2')
ax.set_zlabel('PCA Component 3')
plt.colorbar(scatter, ax=ax)
plt.savefig('../plots/pca_3d_visualization.png', dpi=300)

# Duration tracking
end_time = time.time()
elapsed_time = end_time - start_time

hours = int(elapsed_time // 3600)
minutes = int((elapsed_time % 3600) // 60)
seconds = int(elapsed_time % 60)

print(f"Total time taken: {hours:02}:{minutes:02}:{seconds:02}")
