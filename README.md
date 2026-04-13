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
* **Virtual Rabbit Mechanism:** Dynamically aligns the reference point on the spatial path using a virtual guide speed (`v_theta`).
* **Constant Forward Speed Constraint:** Includes customized cost weights to maintain a stable true forward speed (`u_ref`).
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
```

## ⚙️ Dependencies

To run this simulation, you need a standard Python environment with the following scientific computing libraries installed:

```bash
pip install numpy scipy matplotlib
```

## 🚀 How to Run

1. Clone the repository to your local machine:
   ```bash
   git clone [https://github.com/yourusername/VG-MPCC-AUV.git](https://github.com/yourusername/VG-MPCC-AUV.git)
   cd VG-MPCC-AUV
   ```
2. Ensure that the `OptRoute_Data.csv` file (containing the `[x, y]` waypoints) is placed in the root directory.
3. Run the main simulation script:
   ```bash
   python main.py
   ```

## 📊 Output
Upon successful execution, the script will:
1. Print the optimization progress in the console step-by-step.
2. Save the simulation state logs (positions, speeds, control forces, errors) to `auv_newsimulation_log.csv`.
3. Open a series of Matplotlib figures showcasing:
   * The 2D spatial reference path vs. AUV trajectory.
   * Control inputs (`F_u`, `F_v`, `F_r`) over time.
   * The virtual rabbit speed (`v_theta`).
   * The true AUV forward speed (`u`) tracking the reference.

## 🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/yourusername/VG-MPCC-AUV/issues).

## 📄 License
This project is open-sourced under the [MIT License](LICENSE).
