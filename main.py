# ==============================================================================
# Author: Liang Zhao
# Paper: VG-MPCC: A Virtual-Guide Model Predictive Contouring Control Formulation 
#        for Path Following of Autonomous Underwater Vehicles
# ==============================================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
import csv  # Added: Used to save data as a CSV file

from utils import ClampedSpline1D
from auv_model import AUV
from mpcc import MPCCController2

# =========================================================
# Main program
# =========================================================
def main():
    # -----------------------------------------------------
    # AUV system parameters
    # -----------------------------------------------------
    m = 116.0
    Iz = 13.1
    X_udot = -167.6
    Y_vdot = -477.2
    N_rdot = -15.9
    Xu = 26.9
    Yv = 35.8
    Nr = 3.5
    Du = 241.3
    Dv = 503.8
    Dr = 76.9

    coef = np.array([m, Iz, X_udot, Y_vdot, N_rdot, Xu, Yv, Nr, Du, Dv, Dr], dtype=float)
    ndof = 3

    # -----------------------------------------------------
    # MPC parameters
    # -----------------------------------------------------
    N = 15
    T = 0.2  # Modified sampling time; using the RK4 integrator will no longer cause divergence
    Tstep = 50
    X0 = np.array([0, 0, 0, 0, 0, 0], dtype=float)
    nx = len(X0)
    nu = 4  # [Fu, Fv, Fr, v_theta]

    Xall2 = np.zeros((nx, Tstep + 1), dtype=float)
    Uall2 = np.zeros((nu, Tstep), dtype=float)
    theta_all = np.zeros(Tstep + 1, dtype=float)

    Xall2[:, 0] = X0
    U0_mpc = np.zeros((nu, N), dtype=float)

    # Control bounds
    Fu_max = 2000.0
    Fv_max = 2000.0
    Fr_max = 1000.0
    v_theta_max = 2.0

    Umax0 = np.array([Fu_max, Fv_max, Fr_max, v_theta_max], dtype=float)
    Umin0 = np.array([-Fu_max, -Fv_max, -Fr_max, 0.0], dtype=float)

    # MPCC weights (smoother and more stable)
    mpcc_weights = np.array([1200, 30, 200, 5], dtype=float)

    # Increased control input penalty
    R_weights = 1e-4 * np.eye(3)

    # Increased energy consumption term
    lambda_u = 1e-5
    energy_weights = np.array([1.0, 1.0, 1.0], dtype=float)
    v_target = 1.5

    # Added: Parameters for constant true forward speed term
    speed_weight = 200.0
    u_ref = 1.5

    # -----------------------------------------------------
    # Initialize AUV and MPC
    # -----------------------------------------------------
    auv = AUV(coef, ndof, X0, U0_mpc[0:3, :])
    mpcc = MPCCController2(
        N, auv, mpcc_weights, R_weights, Umax0, Umin0,
        lambda_u, energy_weights, speed_weight, u_ref
    )

    # -----------------------------------------------------
    # Waypoints & Spatial Path Generation
    # -----------------------------------------------------
    # Read CSV waypoint file
    waypoints = np.loadtxt('OptRoute_Data.csv', delimiter=',', dtype=float)

    # If the file contains only one row of data, ensure it remains a 2D array
    waypoints = np.atleast_2d(waypoints)

    if waypoints.shape[1] != 2:
        raise ValueError(
            f"OptRoute_Data.csv format error: it should contain two columns of coordinates [x, y], current shape = {waypoints.shape}"
        )

    ds = np.sqrt(np.sum(np.diff(waypoints, axis=0)**2, axis=1))
    s_wp = np.concatenate(([0.0], np.cumsum(ds)))

    s_dense = np.arange(0.0, s_wp[-1] + 10.0 + 1e-12, 0.05)

    pchip_x = PchipInterpolator(s_wp, waypoints[:, 0])
    pchip_y = PchipInterpolator(s_wp, waypoints[:, 1])
    x_dense = pchip_x(s_dense)
    y_dense = pchip_y(s_dense)

    dx = np.gradient(x_dense, 0.05)
    dy = np.gradient(y_dense, 0.05)
    psi_dense = np.unwrap(np.arctan2(dy, dx))

    PathData = {
        "s": s_dense,
        "Fx": ClampedSpline1D(s_dense, x_dense),
        "Fy": ClampedSpline1D(s_dense, y_dense),
        "Fpsi": ClampedSpline1D(s_dense, psi_dense)
    }

    # -----------------------------------------------------
    # Initial theta alignment
    # -----------------------------------------------------
    print('Projecting AUV position to path for initial theta alignment...')
    X_init = Xall2[0, 0]
    Y_init = Xall2[1, 0]
    dist_sq = (x_dense - X_init)**2 + (y_dense - Y_init)**2
    min_idx = np.argmin(dist_sq)
    theta_all[0] = s_dense[min_idx]
    print(f'Initial theta_all[0] aligned to: {theta_all[0]:.4f}')

    u0_guess = np.tile(np.array([0, 0, 0, v_target], dtype=float), N)

    # -----------------------------------------------------
    # Simulation Loop
    # -----------------------------------------------------
    actual_steps = Tstep
    log_data = []

    for i in range(Tstep):
        if (i + 1) % 10 == 0:
            print(f'Step {i + 1} / {Tstep}...')

        # MPCC optimization
        u_opt = mpcc.calc_control(u0_guess, Xall2[:, i], theta_all[i], PathData, v_target, T)

        u_actual = u_opt[0:3, 0]
        v_theta_actual = u_opt[3, 0]
        Uall2[:, i] = u_opt[:, 0]

        xr_curr = PathData["Fx"](theta_all[i])
        yr_curr = PathData["Fy"](theta_all[i])
        psi_r_curr = PathData["Fpsi"](theta_all[i])
        dx_err = Xall2[0, i] - xr_curr
        dy_err = Xall2[1, i] - yr_curr
        ec = -np.sin(psi_r_curr) * dx_err + np.cos(psi_r_curr) * dy_err
        el = np.cos(psi_r_curr) * dx_err + np.sin(psi_r_curr) * dy_err
        epsi = Xall2[2, i] - psi_r_curr
        epsi = np.arctan2(np.sin(epsi), np.cos(epsi))

        log_data.append({
            'step': i,
            'x': Xall2[0, i], 'y': Xall2[1, i], 'psi': Xall2[2, i],
            'u': Xall2[3, i], 'v': Xall2[4, i], 'r': Xall2[5, i],
            'Fu': u_actual[0], 'Fv': u_actual[1], 'Fr': u_actual[2],
            'v_theta': v_theta_actual, 'theta': theta_all[i],
            'ec': ec, 'el': el, 'epsi': epsi
        })

        # Physical AUV state update
        auv.advance(u_actual, np.array([0, 0, 0], dtype=float), T)
        Xall2[:, i + 1] = auv.X

        # Virtual rabbit update
        theta_all[i + 1] = theta_all[i] + v_theta_actual * T

        # Warm start update
        u0_guess = np.concatenate([
            u_opt[:, 1:].reshape(-1, order='F'),
            np.array([0, 0, 0, v_target], dtype=float)
        ])

        if theta_all[i + 1] >= PathData["s"][-1] - 1.0:
            print('Reached the end of the path (virtual rabbit condition)!')
            actual_steps = i + 1
            break

        dist_to_last_wp = np.sqrt(
            (Xall2[0, i + 1] - waypoints[-1, 0])**2 +
            (Xall2[1, i + 1] - waypoints[-1, 1])**2
        )
        progress_ratio = theta_all[i + 1] / PathData["s"][-1] if PathData["s"][-1] > 0 else 0.0

        if (progress_ratio >= 0.9) and (dist_to_last_wp <= 0.5):
            print(f'AUV is within 0.5m of the last waypoint at step {i+1}. Simulation terminated!')
            actual_steps = i + 1
            break

    # -----------------------------------------------------
    # Save Data to CSV
    # -----------------------------------------------------
    csv_filename = 'auv_newsimulation_log.csv'
    if log_data:
        with open(csv_filename, mode='w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=log_data[0].keys())
            writer.writeheader()
            writer.writerows(log_data)
        print(f"Simulation data successfully saved to {csv_filename}")

    # -----------------------------------------------------
    # Plot
    # -----------------------------------------------------
    plt.figure(1)
    plt.plot(x_dense, y_dense, 'k--', linewidth=1.5, label='Spatial Reference Path')
    plt.plot(waypoints[:, 0], waypoints[:, 1], 'bo', markerfacecolor='b', label='Waypoints')
    plt.plot(Xall2[0, :actual_steps], Xall2[1, :actual_steps], 'r', linewidth=2, label='AUV Trajectory')
    plt.grid(True)
    plt.legend()
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.title('Fast MPCC Path Following')

    plt.figure(2)
    plt.plot(Uall2[0, :actual_steps], 'r')
    plt.ylabel('F_u')
    plt.grid(True)

    plt.figure(3)
    plt.plot(Uall2[1, :actual_steps], 'r')
    plt.ylabel('F_v')
    plt.grid(True)

    plt.figure(4)
    plt.plot(Uall2[2, :actual_steps], 'r')
    plt.ylabel('F_r')
    plt.grid(True)

    plt.figure(5)
    plt.plot(Uall2[3, :actual_steps], 'b', linewidth=1.5)
    plt.ylabel('v_theta')
    plt.xlabel('Time step')
    plt.grid(True)
    plt.title('Virtual Rabbit Speed')

    plt.figure(6)
    plt.plot(Xall2[3, :actual_steps + 1], linewidth=2, label='u')
    plt.axhline(y=u_ref, linestyle='--', linewidth=1.2, label='u_ref')
    plt.xlabel('Time step')
    plt.ylabel('Forward speed u (m/s)')
    plt.title('AUV forward speed')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()