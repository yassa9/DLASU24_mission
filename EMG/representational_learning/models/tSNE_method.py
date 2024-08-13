import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.manifold import TSNE

# start time tracking
start_time = time.time()

# loading data
X_train = np.load('../../X_train_tabular.npy')
y_train = np.load('../../y_train_tabular.npy')

# selecting a random subset of the data
n_samples = 20_000
np.random.seed(42)
indices = np.random.choice(X_train.shape[0], n_samples, replace=False)
X_subset = X_train[indices]
y_subset = y_train[indices]

# performing t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=500, verbose=2)
X_tsne = tsne.fit_transform(X_subset)

# adding results to a DataFrame for easier plotting
df_tsne = pd.DataFrame()
df_tsne['tsne-2d-one'] = X_tsne[:, 0]
df_tsne['tsne-2d-two'] = X_tsne[:, 1]
df_tsne['label'] = np.argmax(y_subset, axis=1)

# plotting
plt.figure(figsize=(16, 10))
sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="label",
    palette=sns.color_palette("hsv", len(np.unique(df_tsne['label']))),
    data=df_tsne,
    legend="full",
    alpha=0.8
)
plt.title('t-SNE Visualization')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.savefig('../plots/tsne_visualization.png', dpi=300)
plt.show()

# duration tracking
end_time = time.time()
elapsed_time = end_time - start_time

hours = int(elapsed_time // 3600)
minutes = int((elapsed_time % 3600) // 60)
seconds = int(elapsed_time % 60)

print(f"Duration: {hours:02}:{minutes:02}:{seconds:02}")
