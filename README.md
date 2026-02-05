# Quantum_Cuda_IBM

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![CUDA-Q](https://img.shields.io/badge/Provider-NVIDIA%20CUDA--Q-green)
![Qiskit](https://img.shields.io/badge/Provider-IBM%20Qiskit-blue)

**Variational Quantum Walks for Distribution Learning (CUDA-Q + Qiskit / IBM Quantum)**

This repository implements **parameterized (variational) quantum-walk models** to learn probability distributions from synthetic and historical financial returns. By leveraging both **NVIDIA CUDA-Q** for high-performance simulation and **Qiskit** for IBM Quantum hardware, the project generates Monte Carlo price scenarios with quantum-enhanced expressivity.

> [!CAUTION]
> **Research code:** Provided “as-is” for experimentation. This is **not** financial advice.

---

## 🔬 Why This Repo Exists

Classical scenario generation (e.g., Gaussian assumptions) often fails to capture "fat tails" and complex volatility clustering. This project explores whether **parameterized quantum-walk circuits** can:
* Fit non-Gaussian target return distributions more accurately.
* Sample efficiently once the variational parameters are optimized.
* Seamlessly bridge the gap between GPU-accelerated simulation and real quantum backends.

---

## ✨ Key Features

* **CUDA-Q Integration**: Fast GPU-accelerated prototyping for high-qubit simulations.
* **Qiskit Backend**: Full support for IBM Quantum hardware and Aer simulators.
* **Variational Pipeline**: Flexible training loops using MSE, KL-Divergence, or NLL loss.
* **Financial Toolkit**: Map discrete quantum states to binned return distributions and generate multi-step Monte Carlo paths.
* **HPC Ready**: Includes templates for SLURM/Ibex batch processing.
