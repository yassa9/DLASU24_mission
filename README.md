<a id="readme-top"></a>

<br />
<div align="center">
  <a href="https://github.com/yassa9/DLASU24_mission">
    <img src="other/logo.png" alt="Logo" width="88" height="80">
  </a>

  <h1 align="center" style="font-size: 60px;">DLASU24_mission</h1>

  <p align="center">
    Implementations & Experiments for DLASU24_mission.
    <br />

  </p>
</div>

<!-- ABOUT THE PROJECT -->
## About The Mission

The mission includes 3 problems where participants are tasked with solving advanced deep learning problems. The test includes:

  - `Finger Angle Prediction from EMG Data`: Participants will use a preprocessed dataset to build five models (Linear DNN, CNN, RNN, CNN-LSTM). The models must be trained for no more than 10 epochs.
  
  - `Data Representation`: Participants must write code to compress or represent the data in a low-dimensional format and visualize it.
  
  - `Oral Classification Paper`: Implementing specific methodologies from oral classification paper using InceptionResNetV2 or alternative approach.

For Full details, see that [PDF](https://github.com/yassa9/DLASU24_mission/blob/master/helpful_pdfs/DLASU24_mission.pdf)
  
<p align="right">(<a href="#readme-top">Back Top</a>)</p>

---

### Problem I
#### Problem Statement: 

Finger Angle Prediction from EMG Data

<p align="right">(<a href="#readme-top">Back Top</a>)</p>

---

### Problem II

#### Problem Statement:

For the same problem above with the same data used how we can represent the data ? \
write a code that compress the data or represent the data with low dimensional representation and plot it.

In `EMG` directory, there is a sub-dir called `representational_learning` containing 2 other sub-dirs. \
One is `models` and the other is `plots`, you can easily conclude what they're for ... 

#### My Solution Choice:

- Used `PCA` & `t-SNE` approaches.
- I wasn't comfortable at first applying them by `CPU` not `GPU`, tried to implement t-SNE on my own by `pyTorch` by sending tensors to `CUDA`, and it asked for 70TB of memory !!
- I implemented t-SNE using just `10 ~ 20k samples` as it is computationaly expensive.
- I found another approach called `UMAP` but I was satisfied by PCA and t-SNE results.

#### Assumptions Taken:

  - The number of components for PCA is set to 3, assuming that the original data's dimensionality can be effectively reduced to 3 without losing significant information.
  - PCA is used to reduce the dimensionality of the data for visualization purposes. It assumes that the first three principal components capture a significant portion of the variance.
  - For t-SNE: using a random subset of the data (20,000 samples) as it's super computational expensive to include whole dataset.
  - The number of components for t-SNE is set to 2.
    
<div align="center">
  <a href="https://github.com/yassa9/DLASU24_mission">
    <img src="EMG/representational_learning/plots/tsne_visualization.png" alt="tsne" width="480" height="300">
  </a>
  <p align="center">
    t-SNE 2D visualization 
    <br />

  </p>
</div>

#### Bottlenecks Found:

  - Reducing the data to only 3 components via PCA gonna lead to information loss.
  - PCA is a linear technique, which assumes that the variance in the data is best captured by linear combinations of the original features.
  - Depending on the size of the datasets, loading `X_train_tabular.npy` and `y_train_tabular.npy` files into memory might be a bottleneck.
  - t-SNE is computationally expensive, especially as the dataset size increases. Even with a subset of 20,000 samples, t-SNE can take a significant amount of time to run. The time complexity of t-SNE is `O(N^2)`, which can be a bottleneck for larger datasets.
  - Reducing the data to only 2 dimensions with t-SNE may oversimplify complex patterns. Although t-SNE captures non-linear relationships well, reducing to just 2 components might not always reveal the full structure of the data.
  - The choice of perplexity (set to 30) can significantly affect the results. Perplexity controls the effective number of neighbors used in the analysis, and inappropriate values can lead to suboptimal clustering. Fine-tuning this parameter requires experimentation, which is computationally expensive.

---

### Problem III

#### Problem Statement:

- You will need to review this [paper](https://github.com/yassa9/DLASU24_mission/blob/master/helpful_pdfs/oral_paper.pdf) and attempt to implement the described methodology to achieve the same results. The data required can be found [here](https://drive.google.com/drive/u/0/folders/1k24VOveceyqqYS4oaBR0iWLiDpsDEUk6).

#### My Solution Choice:

- In `DL_oral_paper` directory, there is `oral_classification.py` for implementing `InceptionResNetV2` method.

#### Assumptions Taken:  



#### Comparisons:
