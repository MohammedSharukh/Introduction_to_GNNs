# The Theory Behind Graph Neural Networks

A companion to [`Intro_to_GraphNeuralNetworks.ipynb`](./Intro_to_GraphNeuralNetworks.ipynb).

The notebook builds a message-passing GCN (the `transform → aggregate → activate`
recipe). This document explains the mathematics *underneath* that recipe: what a
graph convolution really is, what GNNs can and cannot do, why they fail (over-smoothing,
over-squashing, heterophily), and the geometric/topological theory — including
**sheaves** — that repairs those failures.

---

## Table of contents

1. [The five theoretical lenses](#1-the-five-theoretical-lenses)
2. [Sheaves: the geometry of graphs](#2-sheaves-the-geometry-of-graphs)
3. [Worked example — scalar (d = 1)](#3-worked-example--scalar-d--1)
4. [Worked example — vector (d = 2): rotations, parallel transport, holonomy](#4-worked-example--vector-d--2)
5. [Topological Deep Learning: the bigger building](#5-topological-deep-learning-the-bigger-building)
6. [References](#6-references)

---

## 1. The five theoretical lenses

There is no single "theory of GNNs." There are ~five overlapping mathematical
viewpoints, and each explains something the others don't.

### 1.1 Spectral graph theory / graph signal processing
The **graph Laplacian** `L = D − A` is the discrete analogue of the Laplacian operator
from physics. Its eigenvectors define a **graph Fourier transform**, so a graph
convolution is *filtering in the frequency domain*. The GCN in the notebook is a
first-order (low-pass) spectral filter; its `D^{-1/2}(A+I)D^{-1/2}` normalization is that
filter written out.

> **What it explains:** over-smoothing = repeatedly low-pass filtering until only the
> lowest frequency (a constant signal) survives.

- Bruna et al., *Spectral Networks* (2014) — origin of spectral GNNs.
- Defferrard et al., *ChebNet* (2016) — polynomial filters; GCN is its simplification.

### 1.2 Expressive power / the Weisfeiler–Leman hierarchy
The combinatorial theory of *what a GNN can distinguish*. Standard message-passing GNNs
are **exactly as powerful as the 1-WL graph-isomorphism test** — no more. So they
provably cannot count triangles or separate certain non-isomorphic graphs.

> **What it explains:** why we need "beyond-WL" models (higher-order GNNs, subgraph GNNs,
> topological models).

- Xu et al., *How Powerful are GNNs?* (GIN, 2019).
- Morris et al., *Weisfeiler and Leman Go Neural* (2019).
- Maron et al., *Invariant and Equivariant Graph Networks* (2019).

### 1.3 Geometric deep learning / symmetry
A GNN is the most general architecture that is **permutation-equivariant** (relabeling
nodes relabels outputs identically). CNNs (translation symmetry), Transformers
(permutation), and GNNs are the *same principle* on different symmetry groups.

- Bronstein, Bruna, Cohen & Veličković, *Geometric Deep Learning* (2021).

### 1.4 Graphs as PDEs / diffusion
Treat node features as a quantity **diffusing** over continuous time; a GNN layer is one
discretized time-step. This is the **heat equation** on a graph.

> **What it explains:** over-smoothing directly (heat always spreads to a uniform
> temperature) — and it is the doorway to sheaves.

- Chamberlain et al., *GRAND: Graph Neural Diffusion* (2021).

### 1.5 Topology & sheaf theory
The frontier. Replaces the graph Laplacian with a *learned* **sheaf Laplacian**, giving
the network genuine geometric structure (connections, parallel transport, curvature).
Covered in detail below.

---

## 2. Sheaves: the geometry of graphs

### The key observation
A standard GNN implicitly assumes a **trivial sheaf**. In `A · (H·W)`, when node *i*
pulls in neighbour *j*'s features, it copies them **as-is** — assuming *i* and *j* share
the same feature space and coordinate frame. That assumption causes two failures:

1. **Over-smoothing** — diffusion collapses all nodes to one vector.
2. **Heterophily** — on graphs where connected nodes are *dissimilar* (fraud rings,
   protein binding), "average your neighbours" is the wrong prior.

### The fix
A **cellular sheaf** `F` on a graph assigns:

- a vector space (**stalk**) `F(v)` to each **node**,
- a vector space `F(e)` to each **edge**,
- a linear **restriction map** `F_{v◁e} : F(v) → F(e)` to each incident (node, edge) pair.

Before node *i* compares itself to neighbour *j*, each side maps its features into the
**shared edge space** via its restriction map. The **sheaf Dirichlet energy** is

```
E(x) = ½ Σ_(u,v)∈E  ‖ F_{u◁e} x_u − F_{v◁e} x_v ‖²
```

and the operator with `E(x) = ½ xᵀ L_F x` is the **sheaf Laplacian** `L_F`. A standard
GNN is exactly the special case where every restriction map is the identity.

---

## 3. Worked example — scalar (d = 1)

Two nodes `u, v`, one edge, scalar features.

### Standard graph Laplacian

```
L = D − A = [  1  −1 ]
            [ −1   1 ]
```

| eigenvector | meaning     | eigenvalue | fate under diffusion |
|-------------|-------------|:----------:|----------------------|
| `(1,  1)`   | agree (const) | **0**    | **survives forever** |
| `(1, −1)`   | disagree      | 2        | decays               |

If `x_u = 1` (class A) and `x_v = −1` (class B), the informative `(1,−1)` component
decays to `0` and both nodes converge to the average. **Classes become
indistinguishable — over-smoothing**, and the same reason GCNs fail on heterophily.

### Sheaf Laplacian
Let the restriction maps be scalars `a = F_{u◁e}`, `b = F_{v◁e}`:

```
L_F = [  a²   −ab ]
      [ −ab    b² ]
```

**Sanity check (trivial sheaf = ordinary GNN):** `a = b = 1` gives
`L_F = [[1,−1],[−1,1]] = L`. ✓

**Flip one sign** — let the sheaf learn `a = 1`, `b = −1` (an edge that means "these two
should *disagree*"):

```
L_F = [ 1  1 ]
      [ 1  1 ]
```

| eigenvector | meaning  | eigenvalue | fate under sheaf diffusion |
|-------------|----------|:----------:|----------------------------|
| `(1, −1)`   | disagree | **0**      | **survives forever**       |
| `(1,  1)`   | agree    | 2          | decays                     |

The table has **flipped**: now the disagreeing state is preserved, so classes A and B
stay separated. Same diffusion machinery, opposite outcome — because the restriction map
changed which signals count as "smooth."

### Why this generalizes
What survives diffusion is the **kernel** of the Laplacian (sheaf cohomology `H⁰`):

- **Graph Laplacian:** `ker(L)` = constants → only one pattern survives → collapse.
- **Sheaf Laplacian:** `ker(L_F) = { x : a·x_u = b·x_v }` → a *learnable* space.

---

## 4. Worked example — vector (d = 2)

Now each stalk is `ℝ²` (each node carries a 2D **arrow**), and the restriction maps
`a, b` are **2×2 orthogonal matrices** — i.e. **rotations**.

```
E(x) = ½ ‖ a·x_u − b·x_v ‖²        a, b ∈ O(2)
```

With `aᵀa = bᵀb = I` and the **edge transport** `R = aᵀb` (itself a rotation):

```
        [  I    −R  ]
L_F  =  [ −Rᵀ    I  ]
```

The scalar `−1` has been promoted to a **rotation matrix `−R`**. The trivial sheaf is
`R = I`.

### "Agreement up to rotation"
Harmonic signals (preserved by diffusion) satisfy

```
a·x_u = b·x_v      ⟺      x_v = R⁻¹ x_u
```

Nodes agree **not when their arrows are equal, but when one is the other rotated by `R`.**
`R` is the **parallel transport** along the edge — the discrete-geometry *connection*.

### Concrete numbers (transport = 90°)
Take `a = I`, `b = R(90°)`, so `R = [[0,−1],[1,0]]`:

```
        [ 1   0 | 0   1 ]
L_F  =  [ 0   1 |−1   0 ]
        [ 0  −1 | 1   0 ]
        [ 1   0 | 0   1 ]
```

Solving `L_F x = 0` gives `x_v = (x_u₂, −x_u₁)` — node `v`'s arrow is node `u`'s arrow
rotated by −90°:

```
   u: →  (1, 0)        v: ↓  (0, −1)        ← "equal" in the sheaf's eyes
```

Two consequences, both the opposite of over-smoothing:

- The kernel is **2-dimensional** (any `x_u ∈ ℝ²` works) vs. the **1-dimensional**
  constant kernel of the plain Laplacian → more classes stay separable at depth.
- Diffusion drives arrows to a **fixed angular offset**, not a single shared value, so
  information stored in an arrow's *direction* is retained.

### Holonomy: the signature of a real connection
Put three nodes in a triangle `u → v → w → u`, each edge with rotation
`R_uv, R_vw, R_wu`. Transport an arrow around the loop:

```
H = R_wu · R_vw · R_uv        (holonomy)
```

- **`H = I`** → **flat** connection; a consistent global assignment exists.
- **`H ≠ I`** → **nontrivial holonomy = discrete curvature.** No nonzero arrow survives a
  trip around the loop unchanged — the graph *frustrates* global agreement, like carrying
  an arrow around a **Möbius strip**.

This "does parallel transport around a loop return home?" test is the same **curvature**
notion behind over-squashing and Ricci-curvature graph rewiring. So the sheaf picture
unifies **over-smoothing** and **over-squashing** under one object: the **connection on
the graph**.

### The picture to keep
- **Scalar graph Laplacian:** "make neighbours *equal*." Only the constant survives →
  collapse.
- **`d=2` sheaf Laplacian:** "make neighbours equal *after rotating each through a learned
  transport `R`*." Arrows survive; agreement means *aligned up to `R`*; global
  consistency is measured by **holonomy/curvature** around loops.

A sheaf GNN **learns** these rotations from data (Bodnar et al., 2022), giving the network
a genuine notion of geometry rather than the flat "just average" prior of the original
notebook. It provably cures over-smoothing and handles heterophily.

---

## 5. Topological Deep Learning: the bigger building

Sheaves add structure to *edges*. Topological Deep Learning (TDL) goes further, modelling
**higher-order interactions** (a chemical ring, a group of 3+ collaborators, a filled-in
triangle) via simplicial/cellular/combinatorial complexes:

```
graph  →  sheaf (structure on edges)  →  simplicial / cell complex (higher-order faces)
       →  combinatorial complex (all of it)
```

with algebraic topology (Hodge Laplacians, sheaf cohomology) as the unifying language.

- Bodnar et al., *Weisfeiler and Lehman Go Topological* (MPSN, 2021) — simplicial.
- Bodnar et al., *Weisfeiler and Lehman Go Cellular* (CW Networks, 2021) — cell complexes.
- Hajij et al., *Topological Deep Learning: Going Beyond Graph Data* (2023) — survey.
- Papamarkou et al., *Position: TDL is the New Frontier for Relational Learning* (2024).

---

## 6. References

### Foundations (the notebook's model)
- Kipf & Welling. *Semi-Supervised Classification with Graph Convolutional Networks.*
  ICLR 2017. https://arxiv.org/abs/1609.02907
- Gilmer, Schoenholz, Riley, Vinyals & Dahl. *Neural Message Passing for Quantum
  Chemistry.* ICML 2017. https://arxiv.org/abs/1704.01212
- Hamilton, Ying & Leskovec. *Inductive Representation Learning on Large Graphs
  (GraphSAGE).* NeurIPS 2017. https://arxiv.org/abs/1706.02216
- Veličković et al. *Graph Attention Networks.* ICLR 2018. https://arxiv.org/abs/1710.10903

### Spectral theory
- Bruna, Zaremba, Szlam & LeCun. *Spectral Networks and Deep Locally Connected Networks
  on Graphs.* ICLR 2014. https://arxiv.org/abs/1312.6203
- Defferrard, Bresson & Vandergheynst. *ChebNet.* NeurIPS 2016.
  https://arxiv.org/abs/1606.09375

### Expressive power
- Xu, Hu, Leskovec & Jegelka. *How Powerful are Graph Neural Networks? (GIN).* ICLR 2019.
  https://arxiv.org/abs/1810.00826
- Morris et al. *Weisfeiler and Leman Go Neural.* AAAI 2019.
  https://arxiv.org/abs/1810.02244
- Maron, Ben-Hamu, Shamir & Lipman. *Invariant and Equivariant Graph Networks.* ICLR 2019.
  https://arxiv.org/abs/1812.09902

### Geometric deep learning
- Bronstein, Bruna, Cohen & Veličković. *Geometric Deep Learning: Grids, Groups, Graphs,
  Geodesics, and Gauges.* 2021. https://arxiv.org/abs/2104.13478 ·
  book: https://geometricdeeplearning.com/book/

### Diffusion / PDE view
- Chamberlain et al. *GRAND: Graph Neural Diffusion.* ICML 2021.
  https://arxiv.org/abs/2106.10934

### Failure modes
- Li, Han & Wu. *Deeper Insights into Graph Convolutional Networks (over-smoothing).*
  AAAI 2018. https://arxiv.org/abs/1801.07606
- Alon & Yahav. *On the Bottleneck of Graph Neural Networks (over-squashing).* ICLR 2021.
  https://arxiv.org/abs/2006.05205
- Topping, Di Giovanni, Chamberlain, Dong & Bronstein. *Understanding over-squashing and
  bottlenecks on graphs via curvature.* ICLR 2022. https://arxiv.org/abs/2111.14522

### Sheaves & topology
- Hansen & Gebhart. *Sheaf Neural Networks.* 2020. https://arxiv.org/abs/2012.06333
- Bodnar, Di Giovanni, Chamberlain, Liò & Bronstein. *Neural Sheaf Diffusion: A
  Topological Perspective on Heterophily and Oversmoothing in GNNs.* NeurIPS 2022.
  https://arxiv.org/abs/2202.04579
- Bodnar et al. *Weisfeiler and Lehman Go Topological: Message Passing Simplicial
  Networks.* ICML 2021. https://arxiv.org/abs/2103.03212
- Bodnar et al. *Weisfeiler and Lehman Go Cellular: CW Networks.* NeurIPS 2021.
  https://arxiv.org/abs/2106.12575
- Hajij et al. *Topological Deep Learning: Going Beyond Graph Data.* 2023.
  https://arxiv.org/abs/2206.00606
- Papamarkou et al. *Position: Topological Deep Learning is the New Frontier for
  Relational Learning.* ICML 2024. https://arxiv.org/abs/2402.08871

### Video courses
- Petar Veličković — *Intro to graph neural networks (ML Tech Talks).*
  https://www.youtube.com/watch?v=8owQBFAHw7E
- Stanford CS224W — *Machine Learning with Graphs* (Jure Leskovec).
  https://www.youtube.com/playlist?list=PLoROMvodv4rOP-ImU-O1rYRg2RFxomvFp
- Geometric Deep Learning course (Bronstein, Bruna, Cohen, Veličković) — search
  "Geometric Deep Learning course" on YouTube.
