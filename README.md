# VG-MPCC for AUV Path Following

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)

This repository provides a Python implementation of the **Virtual-Guide Model Predictive Contouring Control (VG-MPCC)** algorithm for Autonomous Underwater Vehicles (AUVs). 

The code is a Python conversion of the original MATLAB implementation, designed to maintain the core logic, model structure, and algorithmic integrity of the paper authored by **Liang Zhao**.

## 📑 Reference Paper
**Title:** *VG-MPCC: A Virtual-Guide Model Predictive Contouring Control Formulation for Path Following of Autonomous Underwater Vehicles* **Author:** Liang Zhao  

If you use this code in your academic research, please consider citing the original paper.

## ✨ Key Features
* **3-DOF AUV Dynamics:** Implements a highly accurate 3-Degree-Of-Freedom AUV mathematical model.
* **4th-Order Runge-Kutta (RK4):** Uses RK4 integration for stable and precise discrete-time system dynamics.
* **Advanced MPCC Formulation:** Optimizes contouring error, lag error, heading error, energy consumption, and control input smoothness simultaneously.
* **Virtual Rabbit Mechanism:** Dynamically aligns the reference point on the spatial path using a virtual guide speed ($v_\theta$).
* **Constant Forward Speed Constraint:** Includes customized cost weights to maintain a stable true forward speed ($u_{ref}$).
* **Modular Design:** Cleanly separated Python files for easy reading, modification, and integration into other projects.
* **Simulation & Logging:** Automatically logs trajectory data to a CSV file and generates comprehensive plots.

## 📂 Repository Structure

```text
├── auv_model.py       # AUV kinematics, dynamics, and RK4 integration
├── mpcc.py            # MPCC controller optimization using SciPy (SLSQP)
├── utils.py           # Helper classes including 1D clamped cubic spline interpolation
├── main.py            # Simulation loop, virtual rabbit updates, and data plotting
├── OptRoute_Data.csv  # Required input: Spatial reference path waypoints (x, y)
└── README.md          # Project documentation
