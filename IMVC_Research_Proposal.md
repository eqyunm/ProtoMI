# ProtoMI: Prototype-Level Mutual Information Maximization with Redundancy Reduction for Incomplete Multi-View Clustering

## A Non-Contrastive Framework for Cross-View Alignment Under Arbitrary Missing Patterns

---

## 1. Motivation & Problem Statement

### 1.1 Current Paradigm and Its Limitations

Incomplete Multi-View Clustering (IMVC) aims to partition data into semantically meaningful groups when some views are entirely missing for certain samples. The dominant paradigm relies on **contrastive learning** (e.g., InfoNCE) for cross-view alignment, which pulls representations of the same sample across different views closer (positive pairs) while pushing different samples apart (negative pairs).

However, contrastive learning in the incomplete setting suffers from four fundamental issues:

**P1 — False Negative Contamination.** Without labels, samples from the same cluster may be treated as negatives and pushed apart, causing *class collision*. This is exacerbated in IMVC because missing views prevent reliable pair identification.

**P2 — Paired Data Dependency.** Constructing cross-view positive pairs requires the same sample to be observed in multiple views, but under high missing rates (e.g., >50%), very few such pairs exist, starving the contrastive objective of signal.

**P3 — Task-Alignment Gap.** Instance-level contrastive objectives optimize for sample-level discrimination (distinguishing every sample), which is over-specified relative to the actual clustering goal (distinguishing K clusters). This mismatch wastes representational capacity on within-cluster distinctions.

**P4 — Implicit Uniformity Assumption.** The repulsive force in contrastive loss implicitly assumes a uniform class distribution. Under imbalanced clusters—common in practice—this assumption systematically distorts the learned representation geometry.

### 1.2 Our Insight

We observe that the core purpose of contrastive learning in IMVC is to **enforce cross-view semantic consistency**. This can be achieved more directly and elegantly at the **cluster assignment level** rather than the instance representation level, via:

1. **Prototype-level mutual information maximization** — directly aligning the clustering semantics across views without constructing any positive/negative pairs.
2. **Feature-level redundancy reduction** — preventing representation collapse without relying on negative samples.

This combination yields a **fully non-contrastive** framework that is theoretically grounded, more robust to missing data, and directly aligned with the downstream clustering task.

---

## 2. Proposed Method: ProtoMI

### 2.1 Architecture Overview

```
Input:  x_i^{(v)}  (sample i, view v; may be missing)
                │
    ┌───────────┼───────────┐
    ▼           ▼           ▼
 Encoder₁   Encoder₂   Encoder_V     (view-specific encoders)
    │           │           │
    ▼           ▼           ▼
 h_i^{(1)}  h_i^{(2)}  h_i^{(V)}    (latent representations)
    │           │           │
    ├───────────┼───────────┤
    │    Redundancy Reduction        (L_RR: cross-view correlation)
    │    + Variance Guard            (L_Var: anti-collapse)
    ├───────────┼───────────┤
    │           │           │
    ▼           ▼           ▼
 Cluster₁   Cluster₂   Cluster_V     (shared clustering head)
    │           │           │
    ▼           ▼           ▼
 q_i^{(1)}  q_i^{(2)}  q_i^{(V)}    (soft assignment vectors)
    │           │           │
    ├───────────┼───────────┤
    │  Prototype Mutual Information  (L_PMI: cross-view MI)
    │  + Marginal Entropy Guard      (L_H: uniform regularizer)
    ├───────────┼───────────┤
    │           │           │
    ▼           ▼           ▼
 Proto-Distill for single-view samples (L_PD)
    │
    ▼
 Unified Assignment → Spectral / Hungarian → Final Clusters
```

### 2.2 Module 1: Cross-View Prototype Mutual Information (PMI)

**Goal:** Maximize the mutual information between cluster assignment distributions of different views, ensuring cross-view clustering consistency.

For each pair of views $(a, b)$, consider the set of **paired samples** $\mathcal{P}^{ab} = \{i : x_i^{(a)} \text{ and } x_i^{(b)} \text{ both exist}\}$. The soft assignment vector of sample $i$ in view $v$ is $q_i^{(v)} \in \Delta^{K-1}$ (a K-dimensional probability simplex), produced by a shared clustering head with softmax output.

**Step 1 — Estimate the cross-view joint distribution:**

$$P^{ab}_{jk} = \frac{1}{|\mathcal{P}^{ab}|} \sum_{i \in \mathcal{P}^{ab}} q_{i,j}^{(a)} \cdot q_{i,k}^{(b)}, \quad j,k \in \{1,...,K\}$$

This is a $K \times K$ matrix where entry $(j,k)$ represents the probability mass jointly assigned to cluster $j$ in view $a$ and cluster $k$ in view $b$.

**Step 2 — Compute marginals:**

$$P^{a}_{j} = \sum_k P^{ab}_{jk}, \quad P^{b}_{k} = \sum_j P^{ab}_{jk}$$

**Step 3 — Maximize mutual information:**

$$\mathcal{L}_{PMI} = - \sum_{j,k} P^{ab}_{jk} \log \frac{P^{ab}_{jk}}{P^{a}_{j} \cdot P^{b}_{k}}$$

This is minimized (note the negative sign, since we maximize MI). Intuitively, maximizing $I(Q^{(a)}; Q^{(b)})$ enforces that knowing the cluster assignment in one view gives maximum information about the assignment in the other view, which is precisely the cross-view consistency we need.

**Why this replaces contrastive learning:**

| Property | Contrastive (InfoNCE) | ProtoMI |
|---|---|---|
| Operates on | Instance representations | Cluster assignments |
| Requires negative pairs | Yes | No |
| Sensitive to batch size | Yes (needs many negatives) | No (estimates a K×K matrix) |
| False negative problem | Yes | No (operates on distributions) |
| Alignment granularity | Instance-level | Cluster-level (task-aligned) |
| Paired sample efficiency | Low (each pair = 1 gradient signal) | High (each pair contributes to K² joint estimates) |

### 2.3 Module 2: Marginal Entropy Maximization (MEG)

Maximizing MI alone can be trivially achieved by collapsing all samples into a single cluster (degenerate solution). To prevent this, we maximize the marginal entropy of each view's assignment distribution:

$$\mathcal{L}_{H} = - \sum_v H(\bar{q}^{(v)}) = - \sum_v \left( -\sum_j \bar{q}_j^{(v)} \log \bar{q}_j^{(v)} \right)$$

where $\bar{q}_j^{(v)} = \frac{1}{N_v} \sum_{i: x_i^{(v)} \text{exists}} q_{i,j}^{(v)}$ is the average assignment probability for cluster $j$ across all available samples in view $v$.

Maximizing marginal entropy encourages each cluster to receive a roughly equal share of samples, preventing collapse. Combined with MI maximization, this yields the optimal clustering: **assignments that are maximally informative across views while being maximally spread across clusters**.

**Theoretical connection (Proposition 1):** We can show:
$$I(Q^{(a)}; Q^{(b)}) = H(Q^{(a)}) + H(Q^{(b)}) - H(Q^{(a)}, Q^{(b)})$$

Maximizing MI + maximizing marginal entropies is equivalent to minimizing the joint entropy $H(Q^{(a)}, Q^{(b)})$ under the constraint that marginals are uniform. This means the optimal solution makes the cross-view joint distribution as *concentrated* as possible (i.e., a permutation matrix), which is exactly the desired one-to-one cluster correspondence.

### 2.4 Module 3: Feature-Level Redundancy Reduction (RR)

While PMI operates at the clustering semantic level, we need a complementary objective at the **representation level** to ensure high-quality features. Inspired by Barlow Twins / VICReg, we design a masked variant for the incomplete setting.

**Cross-view cross-correlation matrix:** For each pair of views $(a,b)$, using paired samples $\mathcal{P}^{ab}$:

$$C_{jk}^{ab} = \frac{\sum_{i \in \mathcal{P}^{ab}} \hat{h}_{i,j}^{(a)} \cdot \hat{h}_{i,k}^{(b)}}{\sqrt{\sum_{i \in \mathcal{P}^{ab}} (\hat{h}_{i,j}^{(a)})^2} \cdot \sqrt{\sum_{i \in \mathcal{P}^{ab}} (\hat{h}_{i,k}^{(b)})^2}}$$

where $\hat{h}^{(v)}$ denotes the mean-centered representation.

**Redundancy reduction loss:**

$$\mathcal{L}_{RR} = \underbrace{\sum_j (1 - C_{jj}^{ab})^2}_{\text{invariance term}} + \lambda_{off} \underbrace{\sum_j \sum_{k \neq j} (C_{jk}^{ab})^2}_{\text{redundancy reduction term}}$$

- **Invariance term:** Forces diagonal entries to 1, meaning the $j$-th feature dimension in view $a$ and the $j$-th dimension in view $b$ are perfectly correlated. This aligns representations across views.
- **Redundancy reduction term:** Penalizes off-diagonal entries, decorrelating different feature dimensions to prevent informational collapse.

**Variance guard (from VICReg):** To further prevent collapse, we add a variance regularization term on each view's representations:

$$\mathcal{L}_{Var} = \sum_v \frac{1}{d} \sum_j \max(0, \gamma - \text{Std}(h_{\cdot,j}^{(v)}))$$

where $\text{Std}(h_{\cdot,j}^{(v)})$ is the standard deviation of the $j$-th feature dimension across samples in view $v$, and $\gamma$ is a threshold (typically 1). This hinge loss activates only when a feature dimension's variance drops below $\gamma$, preventing it from becoming constant.

### 2.5 Module 4: Prototype-Guided Distillation for Single-View Samples (PD)

A key challenge in IMVC: samples observed in only one view cannot participate in cross-view objectives. We address this with prototype-guided distillation, exploiting the semantic knowledge embedded in the learned prototypes.

**Step 1 — Learn cluster prototypes from paired data.** From the shared clustering head, define the prototype for cluster $k$ in view $v$ as:

$$\mu_k^{(v)} = \frac{\sum_{i \in \mathcal{P}} q_{i,k}^{(v)} \cdot h_i^{(v)}}{\sum_{i \in \mathcal{P}} q_{i,k}^{(v)}}$$

**Step 2 — Cross-view prototype mapping.** Using the paired data, learn a lightweight affine mapping $\phi_{v \to u}: \mathbb{R}^d \to \mathbb{R}^d$ such that $\phi_{v \to u}(\mu_k^{(v)}) \approx \mu_k^{(u)}$ for all $k$. This captures the systematic transformation between views at the prototype level.

**Step 3 — Distillation for single-view samples.** For sample $i$ observed only in view $v$, generate a pseudo assignment for the missing view $u$ by:

$$\tilde{q}_{i}^{(u)} = \text{softmax}\left(\frac{\phi_{v \to u}(h_i^{(v)}) \cdot [\mu_1^{(u)}, ..., \mu_K^{(u)}]^\top}{\tau_d}\right)$$

The distillation loss minimizes the KL divergence between the actual assignment $q_i^{(v)}$ and the pseudo assignment $\tilde{q}_i^{(u)}$:

$$\mathcal{L}_{PD} = \sum_v \sum_{u \neq v} \sum_{i \in \mathcal{S}_v \setminus \mathcal{P}} D_{KL}(q_i^{(v)} \| \text{sg}(\tilde{q}_i^{(u)}))$$

where $\mathcal{S}_v$ is the set of samples observed in view $v$, and $\text{sg}(\cdot)$ denotes stop-gradient to prevent the pseudo target from being optimized jointly (which would cause trivial solutions).

**Confidence-aware weighting:** Not all pseudo assignments are equally reliable. We weight each distillation term by the confidence of the pseudo assignment:

$$w_i^{(u)} = \max_k \tilde{q}_{i,k}^{(u)} - \frac{1}{K}$$

Low-confidence pseudo assignments (near uniform) contribute little, while high-confidence ones drive the distillation.

### 2.6 Overall Objective

$$\mathcal{L} = \underbrace{\mathcal{L}_{PMI}}_{\text{cross-view MI}} + \alpha \underbrace{\mathcal{L}_{H}}_{\text{entropy guard}} + \beta \underbrace{\mathcal{L}_{RR}}_{\text{redundancy reduction}} + \gamma \underbrace{\mathcal{L}_{Var}}_{\text{variance guard}} + \delta \underbrace{\mathcal{L}_{PD}}_{\text{prototype distillation}}$$

**Training strategy:** Warm-up → Joint optimization → Fine-tuning.

- **Phase 1 (Warm-up, ~30 epochs):** Train only with $\mathcal{L}_{RR} + \mathcal{L}_{Var}$ + reconstruction loss for each auto-encoder, to learn basic view-specific representations.
- **Phase 2 (Joint, ~200 epochs):** Full objective. Enable $\mathcal{L}_{PMI}$ and $\mathcal{L}_{H}$; the clustering head gradually sharpens assignments. $\mathcal{L}_{PD}$ is activated after epoch 50 with a linear ramp-up to ensure prototypes have stabilized.
- **Phase 3 (Fine-tuning, ~20 epochs):** Freeze encoders, fine-tune only the clustering head with $\mathcal{L}_{PMI} + \mathcal{L}_{H}$ to refine cluster boundaries.

---

## 3. Theoretical Analysis

### 3.1 Proposition 1: PMI Achieves Optimal Clustering Alignment

**Claim:** Under the assumption that the marginal cluster distributions are uniform across views, maximizing $I(Q^{(a)}; Q^{(b)})$ drives the joint distribution $P^{ab}$ toward a permutation matrix, i.e., achieving perfect one-to-one cluster correspondence.

**Proof Sketch:** Given $H(Q^{(a)}) = H(Q^{(b)}) = \log K$ (ensured by $\mathcal{L}_H$), maximizing MI is equivalent to minimizing $H(Q^{(a)}|Q^{(b)})$, the conditional entropy. The minimum $H(Q^{(a)}|Q^{(b)}) = 0$ is achieved iff $Q^{(a)}$ is a deterministic function of $Q^{(b)}$, i.e., $P^{ab}$ is a permutation matrix. $\square$

### 3.2 Proposition 2: Sample Efficiency of PMI vs. InfoNCE

**Claim:** Under a missing rate of $\rho$, the PMI loss requires $O(K^2 / \epsilon^2)$ paired samples to estimate the joint distribution within $\epsilon$ total variation distance, whereas InfoNCE requires $O(N / \epsilon^2)$ paired samples for reliable gradient estimation (where $N \gg K^2$ is the total number of samples).

**Intuition:** PMI estimates a $K \times K$ matrix (K typically 2-20), while InfoNCE estimates pairwise similarities across $N$ samples. For a dataset with 10,000 samples and 10 clusters, PMI needs ~100× fewer paired observations for equivalent statistical precision.

### 3.3 Proposition 3: Robustness to Class Imbalance

**Claim:** By replacing the uniform entropy regularizer $\mathcal{L}_H$ with a Rényi entropy variant $H_\alpha(\bar{q}^{(v)})$ (for $\alpha < 1$), the framework naturally accommodates imbalanced cluster sizes without the uniformity bias inherent in contrastive learning's repulsive force.

---

## 4. Experimental Design

### 4.1 Datasets

| Dataset | Samples | Views | Clusters | Type |
|---|---|---|---|---|
| BDGP | 2,500 | 2 | 5 | Biological images |
| Handwritten (HW) | 2,000 | 6 | 10 | Digit features |
| Caltech101-7 | 1,474 | 6 | 7 | Object images |
| Scene-15 | 4,485 | 2 | 15 | Scene images |
| NoisyMNIST | 30,000 | 2 | 10 | Noisy digit images |
| CUB | 11,788 | 2 | 200 | Fine-grained bird images |
| ALOI-100 | 10,800 | 4 | 100 | Object images (illumination) |

Small datasets (BDGP, HW, Caltech) for comparability with existing IMVC literature; large datasets (NoisyMNIST, CUB, ALOI) to test scalability.

### 4.2 Missing Patterns

| Pattern | Description | Rates |
|---|---|---|
| **Uniform Random** | Each view independently missing with probability $\rho$ | $\rho \in \{0.1, 0.3, 0.5, 0.7, 0.9\}$ |
| **Structured Block** | Entire blocks of samples miss specific views | Same rates |
| **Adversarial** | Samples from minority clusters have higher missing rates | Imbalance ratio $R \in \{0.1, 0.3, 0.5\}$ |

Ensure every sample has at least one view available (standard assumption).

### 4.3 Baselines

**Contrastive IMVC methods (primary comparison):**
- COMPLETER (CVPR 2021) — dual contrastive prediction
- DCP (TPAMI 2022) — dual contrastive prediction for representation learning
- DSIMVC (ICML 2022) — deep safe incomplete MVC
- SURE (ICML 2022) — sufficient and robust MVC
- SMILE (NeurIPS 2022) — scaling incomplete MVC
- CVCL (AAAI 2023) — consistency via contrastive learning
- ICMVC (AAAI 2024) — incomplete contrastive MVC
- DCG (AAAI 2025) — diffusion contrastive generation
- HSACC (NeurIPS 2025) — hierarchical semantic alignment

**Non-contrastive / alternative methods:**
- COPER (ICLR 2025) — permutation-based CCA (complete setting; adapt to incomplete)
- GAN-based: GP-MVC, EERIMVC
- Graph-based: AGCL, RGCL (IJCAI 2025)

### 4.4 Evaluation Metrics

- **ACC** (Clustering Accuracy, with Hungarian matching)
- **NMI** (Normalized Mutual Information)
- **ARI** (Adjusted Rand Index)
- **PUR** (Purity)

Report mean ± std over 5 random runs with different missing masks.

### 4.5 Ablation Studies (Critical for ICLR)

| Experiment | Purpose | What to remove/replace |
|---|---|---|
| **A1** | Necessity of PMI | Replace $\mathcal{L}_{PMI}$ with InfoNCE at instance level |
| **A2** | Necessity of RR | Remove $\mathcal{L}_{RR}$ and $\mathcal{L}_{Var}$ |
| **A3** | Necessity of MEG | Remove $\mathcal{L}_{H}$ (observe collapse) |
| **A4** | Necessity of PD | Remove $\mathcal{L}_{PD}$ (single-view samples unused) |
| **A5** | Confidence weighting | Remove $w_i^{(u)}$ in distillation |
| **A6** | Shared vs. separate clustering heads | Use view-specific heads |
| **A7** | Training phases | One-phase vs. three-phase |

### 4.6 Analytical Experiments (For Deeper Insights)

**E1 — Sample efficiency under extreme sparsity:**
Plot ACC vs. fraction of paired samples available (1% to 100%). Hypothesis: ProtoMI degrades gracefully while contrastive methods collapse rapidly below ~10%.

**E2 — Convergence and training stability:**
Learning curves (loss + ACC) compared to contrastive baselines. Show reduced oscillation and faster convergence.

**E3 — Sensitivity to K (number of clusters):**
Vary $K$ from 0.5× to 2× the true number of clusters. PMI on a K×K matrix should be more robust than instance-level contrastive loss.

**E4 — Visualization:**
- t-SNE of learned representations colored by true labels (compare ProtoMI vs. contrastive baseline)
- Heatmap of the joint distribution matrix $P^{ab}$ — should converge to near-permutation matrix
- Evolution of $P^{ab}$ during training (show convergence dynamics)

**E5 — Class imbalance study:**
Create artificially imbalanced versions of datasets ($R \in \{0.1, 0.3, 0.5, 0.9\}$). Compare ProtoMI with Rényi entropy vs. contrastive methods.

**E6 — Prototype quality:**
Measure prototype correspondence error: $\|\phi_{v \to u}(\mu_k^{(v)}) - \mu_k^{(u)}\|$ during training. Show convergence of cross-view prototypes.

---

## 5. Implementation Details

### 5.1 Network Architecture

- **Encoders:** View-specific multi-layer autoencoders. Architecture depends on data type:
  - Feature inputs (HW, Caltech): MLP encoder [d → 1024 → 512 → 256 → d_z], d_z = 128
  - Image inputs (NoisyMNIST, CUB): ResNet-18 backbone pretrained, last FC replaced with [512 → 256 → 128]
- **Decoders:** Symmetric to encoders (for warm-up reconstruction loss only, frozen after Phase 1)
- **Clustering Head:** Shared across views. Two FC layers [d_z → 256 → K] with softmax output. Temperature $\tau$ linearly annealed from 1.0 to 0.1 during training.
- **Prototype Mapper $\phi_{v \to u}$:** Single linear layer [d_z → d_z], trained end-to-end.

### 5.2 Hyperparameters

| Parameter | Value | Selection |
|---|---|---|
| $\alpha$ (entropy weight) | 5.0 | Grid search {1, 2, 5, 10} |
| $\beta$ (RR weight) | 1.0 | Fixed |
| $\gamma$ (variance threshold) | 1.0 | Standard (from VICReg) |
| $\delta$ (distillation weight) | 0.1 → 1.0 (ramp-up) | Linear ramp epochs 50-100 |
| $\lambda_{off}$ (off-diagonal weight) | 0.005 | Grid search {0.001, 0.005, 0.01} |
| $\tau_d$ (distillation temperature) | 0.5 | Grid search {0.1, 0.5, 1.0} |
| Batch size | 256 | Fixed |
| Optimizer | Adam | lr=1e-3, weight decay=1e-5 |
| LR schedule | Cosine annealing | T_max = total epochs |

### 5.3 Critical Implementation Notes

1. **Handling missing views in mini-batch:** Each batch may contain samples with different availability patterns. Use binary mask matrices $M^{(v)} \in \{0,1\}^{B}$ to select available samples for each view and compute losses accordingly.
2. **Joint distribution estimation with momentum:** To stabilize the $K \times K$ joint distribution estimate (which may be noisy in small batches), use exponential moving average: $\hat{P}^{ab} \leftarrow 0.9 \hat{P}^{ab} + 0.1 P^{ab}_{batch}$.
3. **Gradient isolation:** Stop-gradient on pseudo targets in distillation loss; stop-gradient on the joint distribution marginals when computing MI gradient (only differentiate through the joint).

---

## 6. Novelty Justification and Differentiation

### 6.1 What Makes This ICLR-Level

**N1 — Paradigm Shift:** We don't just tweak contrastive learning; we **replace** it entirely with a principled, information-theoretic alternative. This addresses a structural limitation in the field.

**N2 — Theoretical Grounding:** We provide formal analysis showing PMI achieves optimal cross-view alignment (Prop 1), is more sample-efficient than InfoNCE (Prop 2), and is naturally adaptable to imbalanced scenarios (Prop 3).

**N3 — Unified Framework:** The combination of PMI (semantic alignment) + RR (feature quality) + PD (incomplete data handling) addresses all three key challenges in IMVC within a single coherent framework, rather than patching contrastive learning with ad-hoc fixes.

**N4 — Practical Advantages:** No negative samples, no large batch requirement, graceful degradation under extreme missing rates — each of these is independently valuable for practitioners.

### 6.2 Differentiation from Related Work

| Method | Key Difference from ProtoMI |
|---|---|
| COMPLETER / DCP | Instance-level contrastive loss; suffers P1-P4 |
| DCG (AAAI'25) | Uses diffusion for generation + contrastive for alignment; we eliminate contrastive entirely |
| HSACC (NeurIPS'25) | Uses MI in low-level space for alignment; we use MI specifically at the cluster assignment level, which is more task-aligned |
| COPER (ICLR'25) | Permutation-based CCA for complete MVC; does not handle missing views; uses CCA loss not MI |
| SwAV-style | Online clustering with Sinkhorn; does not address cross-view alignment under missing data |
| Barlow Twins | Single-view SSL; our RR module is adapted for cross-view incomplete setting with masking |

### 6.3 Anticipated Reviewer Questions & Responses

**Q: "How is this different from just applying existing MI-based objectives (e.g., IIC) to IMVC?"**

A: IIC (Ji et al., 2019) maximizes MI between augmented views in single-view clustering. Our contributions are: (a) extending MI maximization to the multi-view setting with missing data, which requires the paired-sample estimation strategy and the prototype distillation mechanism for single-view samples; (b) combining MI with Barlow-Twins-style redundancy reduction, providing dual-level alignment; (c) providing the sample efficiency analysis specific to the incomplete setting.

**Q: "The joint distribution P^{ab} is estimated only from paired samples — doesn't this introduce bias?"**

A: Under MCAR (Missing Completely At Random), the paired samples are an unbiased subsample. Under MNAR, bias can exist, but our momentum-based estimation and the prototype distillation mechanism partially compensate by propagating semantic knowledge to non-paired samples. We empirically validate robustness under structured and adversarial missing patterns.

**Q: "Why not use optimal transport instead?"**

A: OT is a strong alternative for cross-view alignment, but it requires solving an optimization subproblem at each iteration (Sinkhorn iterations), adding computational cost. PMI provides a closed-form gradient through the joint distribution, making it simpler and faster. We include an OT-based baseline in ablations.

---

## 7. Expected Results and Story

### 7.1 Main Results Table (Expected Trends)

Under missing rate $\rho = 0.5$:

| Method | BDGP ACC | HW ACC | Caltech ACC | NoisyMNIST ACC |
|---|---|---|---|---|
| COMPLETER | ~85 | ~75 | ~60 | ~80 |
| SURE | ~87 | ~78 | ~63 | ~82 |
| DCG | ~90 | ~80 | ~65 | ~85 |
| HSACC | ~91 | ~82 | ~67 | ~86 |
| **ProtoMI** | **~93** | **~84** | **~70** | **~88** |

Expected improvements of **2-4%** ACC over SOTA, which is meaningful in this saturated field.

### 7.2 Key Selling Points in Results

1. **Extreme missing rate robustness:** At $\rho = 0.9$, ProtoMI maintains reasonable performance (e.g., >70% ACC on BDGP) while contrastive methods degrade catastrophically (<60%).
2. **Training stability:** No mode collapse, no oscillation. Clean convergence curves.
3. **The P^{ab} heatmap:** A beautiful visualization showing the joint distribution converging to a permutation matrix during training — this is both visually compelling and theoretically validating.

### 7.3 Paper Narrative

The paper tells a clear story:

1. **Contrastive learning is the wrong tool for IMVC** (motivated by P1-P4).
2. **The right abstraction level for cross-view alignment is cluster assignments, not instance representations** (motivated by the task-alignment gap).
3. **PMI + RR is a principled, non-contrastive alternative** (supported by theory and experiments).
4. **Prototype distillation extends the framework to single-view samples** (completing the practical framework).

---

## 8. Timeline and Milestones

| Week | Task |
|---|---|
| 1-2 | Implement base framework (encoders, clustering head, PMI loss) |
| 3 | Implement RR module and variance guard |
| 4 | Implement prototype distillation module |
| 5-6 | Experiments on small datasets (BDGP, HW, Caltech) + ablation studies |
| 7-8 | Experiments on large datasets (NoisyMNIST, CUB, ALOI) |
| 9 | Analytical experiments (E1-E6) + visualizations |
| 10 | Writing: introduction, method, experiments |
| 11 | Writing: theory section, related work, revision |
| 12 | Internal review, polish, submit |

---

## 9. Risk Analysis and Mitigation

| Risk | Likelihood | Mitigation |
|---|---|---|
| PMI estimates noisy under extreme sparsity | Medium | Momentum-based estimation; mini-batch accumulation |
| Prototype distillation introduces error propagation | Medium | Confidence-aware weighting; delayed activation |
| RR module insufficient to prevent collapse | Low | VICReg variance guard as backup; monitor feature rank |
| Performance gains marginal on some datasets | Medium | Emphasize robustness analysis (extreme $\rho$, imbalance) as key differentiator |
| Scalability issues on CUB (200 classes) | Low | $K \times K$ matrix is only 200×200; negligible vs. N×N in contrastive |
