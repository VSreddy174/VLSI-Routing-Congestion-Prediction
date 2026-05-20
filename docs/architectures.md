# Architectural Evolution and Ablation Study

This document details the structural evolution of the convolutional neural networks used in this project. All model classes are centrally defined in `core/models.py`. 

The progression through the 6 experiments was systematic, with each architectural change serving as a direct response to a physical limitation observed in the previous model's predictions.

---

## Experiment 1: Baseline U-Net (MSE)

We established the initial mapping of pre-placement features to routing congestion using a standard encoder-decoder U-Net architecture.

* The input layer accepts 4 channels: Macro Region, Cell Density, RUDY Standard, and Congestion Context.
* The hidden layers utilize standard double convolutions with 3x3 kernels, followed by batch normalization and ReLU activation.
* The network uses max-pooling for downsampling and transposed convolutions for upsampling.
* The final output layer uses a single 1x1 convolution mapping to 2 output channels for horizontal and vertical routing utilization.
* Evaluating this baseline revealed a severe "channel bleed" issue. Horizontal and vertical routing occur on completely different metal layers with varying design rules and capacities. By forcing the network to share the final convolutional weights for both horizontal and vertical predictions, the model blurred directional traffic. This resulted in conservative predictions that failed to capture sharp congestion spikes.

---

## Experiment 2: Baseline U-Net (Weighted L1)

Before modifying the architecture, we tested if the failure to predict sharp spikes was a purely mathematical issue.

* The architecture remained identical to the baseline U-Net.
* The loss function was shifted from Mean Squared Error to a Weighted L1 loss to heavily penalize errors in known high-congestion zones.
* The model achieved massive improvements in structural similarity, exceeding an SSIM of 0.96.
* Despite capturing the general shape of routing demand, the model still failed to accurately predict the absolute peak values of the hotspots, confirming that the shared output head was a physical structural bottleneck.

---

## Experiment 3: Decoupled U-Net (Masked Huber)

To solve the channel bleed observed in the first two experiments, we altered the final stage of the network.

* The shared decoder was branched into two completely independent 1x1 convolutional heads.
* One head was dedicated strictly to horizontal routing, and the other to vertical routing.
* This structural isolation allowed the optimizer to learn separate final-stage kernels for horizontal and vertical traffic, preventing the weights from interfering with each other.
* This decoupling immediately improved the model's ability to handle the severe imbalance typically found in vertical routing layers.

---

## Experiment 4: Attention U-Net (CBAM)

Routing congestion is highly localized, often with 90% of a chip remaining perfectly routable while 10% contains critical bottlenecks.

* We integrated the Convolutional Block Attention Module (CBAM) after every encoder block.
* The module applies channel attention to learn which feature maps are most critical, such as heavily weighting the RUDY channel over the macro channel in specific regions.
* The module applies spatial attention to suppress activations in the empty zero-padded regions and amplify them near dense logic clusters.
* This spatial filtering forced the network to focus on high-traffic corridors, pushing the ROC-AUC score above 0.93.

---

## Experiment 5: Decoupled U-Net (Classification Pivot)

Predicting the exact number of missing routing tracks as a regression task proved mathematically hostile due to the extreme sparsity of overflow in the ground truth data.

* We reverted to the physically sound Decoupled U-Net architecture from Experiment 3.
* Instead of outputting continuous regression values, the final heads were configured to output logits.
* The ground truth was thresholded into a binary classification problem, where 1 represented an overflow failure and 0 represented safe routing.
* Treating congestion as an anomaly detection task and applying a heavily weighted Binary Cross-Entropy loss significantly improved the F1-score for detecting actual hotspots.

---

## Experiment 6: Inception U-Net (BCE)

With the classification task proven effective, the architecture was optimized to capture the multi-scale physics of VLSI routing. 

* The standard 3x3 double convolutions in the encoder and decoder were replaced with Inception Modules.
* The Inception module processes the incoming feature map through parallel pathways.
* The 1x1 convolutions capture exact local cell density.
* The 3x3 convolutions capture immediate neighbor routing dynamics.
* The 5x5 convolutions capture broader macroscopic routing corridors.
* Concatenating these multi-scale features at every level of the network allowed this model to achieve the highest combined F1-score and ROC-AUC of the entire study.
