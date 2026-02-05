# Quantum_Cuda_IBM

**Variational Quantum Walks for Distribution Learning (CUDA-Q + Qiskit / IBM Quantum).**  
This repository implements **parameterized (variational) quantum-walk models**—with implementations in **NVIDIA CUDA-Q** and **Qiskit**—to **learn probability distributions** from **synthetic and historical financial returns** and generate **Monte Carlo price scenarios**.

> ⚠️ Research code: provided “as-is” for experimentation.  
> 📌 Not financial advice.

---

## Why this repo exists

Classical scenario generation (e.g., Gaussian assumptions) often fails to capture heavy tails and complex dynamics. This project explores whether **parameterized quantum-walk circuits** can serve as expressive generative models that:
- fit target return distributions,
- sample efficiently once trained,
- run on simulators (CUDA-Q) and optionally IBM Quantum backends (Qiskit).

---

## Key Features

- **CUDA-Q implementation** for fast simulation and prototyping (GPU-accelerated where available)
- **Qiskit implementation** for circuit construction + execution on simulators / IBM Quantum backends
- **Variational training loop** to fit distributions (loss functions like MSE / KL / NLL depending on script)
- **Scenario generation**: sample learned distributions → build Monte Carlo paths
- **HPC-ready workflows** (e.g., SLURM/Ibex-style batch runs)




---

## Requirements

### Option A — CUDA-Q path
- `cuda-quantum` (CUDA-Q)
- Python 3.9+
- `numpy`, `scipy`, `pandas`, `matplotlib`, `tqdm` (typical)

### Option B — Qiskit path
- `qiskit`
- Optional: IBM Quantum account / token (if running on hardware)
- Python 3.9+
- `numpy`, `scipy`, `pandas`, `matplotlib`, `tqdm`

---

## Installation

### 1) Create an environment

**Conda (recommended):**
bash
conda create -n quantum-walk python=3.10 -y
conda activate quantum-walk
pip install -U pip
2A) Install CUDA-Q (CUDA-Q workflow)
Follow NVIDIA CUDA-Q installation for your platform, then verify:

python -c "import cudaq; print('CUDA-Q OK:', cudaq.__version__)"
Install Python deps:

pip install numpy scipy pandas matplotlib tqdm
2B) Install Qiskit (Qiskit workflow)
pip install qiskit numpy scipy pandas matplotlib tqdm
(Optional) If running IBM Quantum:

Set your token via environment variable (recommended):

export IBM_QUANTUM_TOKEN="YOUR_TOKEN_HERE"
Quick Start
A) Run a CUDA-Q experiment
python cudaq/run_cudaq.py \
  --data data/synthetic/returns.csv \
  --steps 2000 \
  --lr 0.05 \
  --seed 7 \
  --out results/cudaq_run/
B) Run a Qiskit experiment (simulator)
python qiskit/run_qiskit.py \
  --data data/synthetic/returns.csv \
  --steps 2000 \
  --optimizer cobyla \
  --seed 7 \
  --out results/qiskit_sim/
C) Run on IBM Quantum (optional)
If your script supports it:

python qiskit/run_qiskit.py \
  --data data/historical/returns.csv \
  --backend ibm_backend_name \
  --steps 300 \
  --shots 4000 \
  --out results/ibm_hw/
Tip: start on simulators first, then move to hardware once the pipeline is stable.


Model Overview (high-level)
A typical “quantum walk” generative model here looks like:

Initialize coin + position registers

Apply a sequence of parameterized walk steps:

parameterized coin operators

conditional shift operators

optional split-step / multi-step variants

Measure the position register to obtain a discrete probability distribution

Map measured outcomes → returns (bin centers / quantiles)

Train parameters to minimize divergence from the target distribution

Data Format
Recommended input: a CSV file with a single column of returns:

return
-0.0123
0.0041
...
If you store prices instead of returns, include a preprocessing step:

compute log-returns: r_t = log(p_t/p_{t-1})

normalize/clamp as needed

discretize into bins if your model outputs a discrete distribution

Outputs
Typical outputs saved under results/...:

training loss curves (loss.csv, loss.png)

learned parameters (params.json)

sampled returns (samples.csv)

scenario paths (mc_paths.npy / plots)

Running on HPC (SLURM / Ibex-style)
Example job script (slurm/run_cudaq.slurm):

#!/bin/bash
#SBATCH --job-name=qw_cudaq
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
# (optional GPU)
#SBATCΗ --gres=gpu:1

module load python
source activate quantum-walk

python cudaq/run_cudaq.py --data data/historical/returns.csv --steps 3000 --out results/hpc_run/
Submit:

sbatch slurm/run_cudaq.slurm
Reproducibility
Fix random seeds (--seed)

Log hyperparameters (params.json)

Save versions (pip freeze > results/requirements.txt)

Keep raw data immutable (data/raw/), store processed outputs separately (data/processed/)

Notes / Limitations
Hardware runs (IBM Quantum) introduce noise; expect:

slower training loops (shot-based sampling)

stochastic gradients / noisy loss curves

Discretization matters: binning / scaling can dominate results

This repo is best used as an experimental research pipeline

Citation
If you build on this work, cite the repository:

@misc{ghori_quantum_cuda_ibm,
  author = {Salman Ghori},
  title  = {Quantum\_Cuda\_IBM: Variational Quantum Walks in CUDA-Q and Qiskit},
  year   = {2026},
  howpublished = {GitHub repository}
}
License
Choose a license (MIT / BSD-3 / Apache-2.0) and add LICENSE to the repo.

Contact
Open an issue or reach out via GitHub for questions or collaboration.


---

If you paste your repo’s **file list** (top-level filenames) or a screenshot of the repo tree, I can tailor the README to match your *exact* scripts/commands (so it’s copy-paste runnable).
::contentReference[oaicite:0]{index=0}
