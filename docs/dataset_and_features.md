# Dataset and Features

This document outlines the data pipeline, the physical meaning of the features, and the grid normalization techniques used to prepare the CircuitNet N28 benchmark for convolutional neural networks.

## Data Lineage and Preparation

The data engineering for this project is divided into two phases: raw extraction and shard preprocessing.

### 1. Raw Data Extraction
The original dataset is the CircuitNet N28 benchmark. The raw data consisted of over 10,000 individual chip designs with features stored as separate `.npy` files across multiple directories. 
* **Original Data Source:** [CircuitNet Google Drive](https://drive.google.com/drive/u/1/folders/1GjW-1LBx1563bg3pHQGvhcEyK2A9sYUB)
* **Preparation Logic:** The complete process of exploring the raw data, aligning coordinates, and compiling the matrices into training batches is documented in `notebooks/colab_dataset_preparation.ipynb`.

### 2. Preprocessed Training Data
To eliminate I/O bottlenecks during deep learning training, the raw files were compiled into batched `.npz` shards. 
* **Training Dataset:** [Kaggle: CircuitNet Preprocessed Features](https://www.kaggle.com/datasets/chupparetti/circuitnet-preprocessed-features/)
* **EDA Logic:** The exploratory data analysis of these specific shards is documented in `notebooks/kaggle_dataset_exploration.ipynb`.

---

## Grid Normalization and Masking

A core challenge in applying CNNs to physical design is that chips are not uniform in size (e.g., one design might be 234x233 GCells, while another is 289x288). Neural networks, however, require fixed-size input tensors.

**The Solution:**
1. **Zero-Padding:** Every design was zero-padded and anchored to the top-left to fit a uniform **320x320** grid.
2. **The Mask Array:** To prevent the model from learning "fake" zero-congestion patterns in the padded areas, every sample includes a binary `mask` of shape `(320, 320)`. 
   * `1` indicates valid chip area.
   * `0` indicates artificial padding.

This mask is passed directly into the custom loss functions (e.g., `MaskedHuberLoss`, `MaskedWeightedBCE`) so that the loss is only computed on the active area of the chip.

---

## Dataset Structure

The preprocessed Kaggle dataset consists of 103 `.npz` files (`shard_000.npz` to `shard_102.npz`). Each standard shard contains 100 samples.

Loading a shard yields a dictionary with the following keys:
* `X`: Input feature tensor of shape `(N, 7, 320, 320)`
* `Y`: Output target tensor of shape `(N, 5, 320, 320)`
* `mask`: Binary mask tensor of shape `(N, 320, 320)`
* `names`: String identifiers for the chip designs of shape `(N,)`

---

## Feature Dictionary

### Input Channels (X Tensor)
These 7 channels represent the state of the chip after placement but before routing. These are the independent variables the model uses to make predictions.

| Channel Index | Feature Name | Data Type | Physical Meaning |
| :--- | :--- | :--- | :--- |
| **0** | `macro_region` | Binary (0/1) | Indicates the presence of large, immovable macro blocks (e.g., SRAMs). A value of 1 means the region is blocked, forcing wires to route around it. |
| **1** | `cell_density` | Integer | The count of standard logic cells placed within a specific GCell. High density often correlates with high routing demand. |
| **2** | `RUDY_standard` | Continuous | Rectilinear Undirected Distance Yield. A heuristic that estimates wiring demand based on the bounding box of connected pins. |
| **3** | `RUDY_short` | Continuous | RUDY estimation specifically calculated for short nets (local connections). |
| **4** | `RUDY_long` | Continuous | RUDY estimation specifically calculated for long nets (global connections spanning the chip). |
| **5** | `RUDY_pin` | Continuous | Pin density estimation based on the RUDY framework. |
| **6** | `RUDY_pin_long` | Continuous | Pin density estimation specifically for long nets. |

### Output Channels (Y Tensor)
These 5 channels represent the actual outcomes after the Global Router has finished its job. These serve as the ground truth targets for the models.

| Channel Index | Feature Name | Data Type | Physical Meaning |
| :--- | :--- | :--- | :--- |
| **0** | `GR_h_utilization` | Continuous | The ratio of used horizontal tracks to available horizontal tracks. Values > 1.0 indicate over-capacity. |
| **1** | `GR_v_utilization` | Continuous | The ratio of used vertical tracks to available vertical tracks. Values > 1.0 indicate over-capacity. |
| **2** | `GR_h_overflow` | Continuous | The absolute number of horizontal tracks missing to complete the routing. This matrix is highly sparse. |
| **3** | `GR_v_overflow` | Continuous | The absolute number of vertical tracks missing. This matrix is highly sparse and represents the primary failure targets. |
| **4** | `DRC_violations` | Integer | The count of physical Design Rule Check manufacturing errors resulting from the congestion. |
