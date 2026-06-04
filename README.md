# DualGate
DualGate is a GNN-Based Modeling Approach for AIGs.
# abstract
In Very-Large-Scale Integration (VLSI) design, AND-Inverter Graphs (AIGs) serve as compact representations of Boolean logic and are widely employed in core Electronic Design Automation (EDA) flows such as logic synthesis and formal verification. 
In these flows, accurately encoding node-level functional semantics is crucial for enhancing downstream analysis accuracy and optimization quality.
However, due to dense inverter deployment and complex interactions among two-input \texttt{AND} gates, existing studies remain limited in gate-level semantic modeling, particularly in disentangling and preserving complementary logical semantics during deep propagation, and in adaptively modeling gate-level functional aggregation.
To address these challenges, we propose \emph{DualGate}, which resolves the fidelity preservation of complementary logical signals and the fine-grained modeling of gate-level functionality in deep network propagation by constructing dual semantic channels and an adaptive gating mechanism. 
Specifically, our method first employs a gate-type-aware graph decoupling component to separate \texttt{AND}-input and \texttt{NOT}-input edges according to the target node type, providing a structural foundation for differentiated propagation. 
It then introduces a dual-channel complementary semantic propagation component, leveraging dual initialization and cross-channel \texttt{NOT} propagation to maintain the identifiability of complementary logical semantics. 
In addition, a gated adaptive aggregation component is developed, which enforces binary pairing at the input side and constructs a feature-interaction-aware difference-aware gating mechanism, thereby providing a learnable mechanism to modulate the contributions of \texttt{AND}-gate inputs according to their interaction patterns.
Experimental results demonstrate that DualGate consistently outperforms existing state-of-the-art methods, achieving MAE and MSE improvements of 16.07\% and 33.33\% on Signal Probability Prediction, respectively, as well as 29.62\% and 23.65\% on Truth Table Distance Prediction, respectively.
# Environment
The code is implemented in Python with PyTorch. We recommend using Conda to
create an isolated environment:
```bash
conda env create -f environment.yml
conda activate dualgate
```
If you prefer to create the environment manually:
```bash
conda create -n dualgate python=3.10 -y
conda activate dualgate
conda install dualgate numpy tqdm -c pytorch -c conda-forge -y
```
For CUDA-enabled training, install the PyTorch build that matches your local
CUDA driver. For example:
```bash
conda install pytorch pytorch-cuda=12.1 numpy tqdm -c pytorch -c nvidia -c conda-forge -y
```
Optional dependencies are only needed when using spectral input features:
```bash
conda install scipy scikit-learn -c conda-forge -y
```
