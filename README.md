# Introduction to Graph Neural Networks (GNNs)

A beginner-friendly, fully-explained Jupyter notebook that builds a **Graph
Neural Network (GNN)** from scratch and trains it to classify scientific
papers in the **Cora** citation network.

Every code cell is paired with a plain-English explanation covering **what the
code does**, its **historical context**, and **real-world applications** — so
you can follow along even if GNNs are brand new to you.

## What's inside

The notebook [`Intro_to_GraphNeuralNetworks.ipynb`](./Intro_to_GraphNeuralNetworks.ipynb)
walks through, step by step:

1. Installing and verifying the **Spektral** GNN library
2. Loading the **Cora** dataset (2,708 papers, 1,433 features, 7 topics)
3. Understanding features, the adjacency matrix, and train/val/test masks
4. Writing **masked** loss and accuracy functions for semi-supervised learning
5. Implementing a single GNN layer as **transform → aggregate → activate**
6. Assembling a **two-layer Graph Convolutional Network (GCN)**
7. Training it with TensorFlow 2 and reading the results (~75% accuracy)

## Key concepts explained

- Message passing / neighbourhood aggregation
- Transductive, semi-supervised node classification
- The Kipf & Welling GCN (ICLR 2017) and how this simplified version relates to it
- Automatic differentiation with `tf.GradientTape` and the Adam optimizer

## Requirements

- Python 3.9+
- [Spektral](https://graphneural.network/) 1.2.0
- TensorFlow 2.x, NumPy, SciPy, NetworkX, scikit-learn

Install the main dependency with:
pip install spektral


## How to run

1. Open `Intro_to_GraphNeuralNetworks.ipynb` in Jupyter Notebook, JupyterLab,
   or Google Colab.
2. Run the cells from top to bottom.

## Dataset

**Cora** — the "MNIST of graph learning" — a citation network of machine-learning
papers, loaded automatically via Spektral.

## Acknowledgements

Special thanks to **Petar Veličković**, whose talk
[*Intro to graph neural networks (ML Tech Talks)*](https://www.youtube.com/watch?v=8owQBFAHw7E)
introduced me to Graph Neural Networks — the code in this notebook follows the
live coding demo from that video.

Built with [Spektral](https://github.com/danielegrattarola/spektral) by Daniele
Grattarola. The GCN model follows Kipf & Welling, *"Semi-Supervised
Classification with Graph Convolutional Networks"* (ICLR 2017).
