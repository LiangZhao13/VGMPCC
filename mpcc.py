# ==============================================================================
# Author: Liang Zhao
# Paper: VG-MPCC: A Virtual-Guide Model Predictive Contouring Control Formulation 
#        for Path Following of Autonomous Underwater Vehicles
# ==============================================================================

import numpy as np
from scipy.optimize import minimize, Bounds

# =========================================================
# MPCC Controller class
# =========================================================
class MPCCController2:
    def __init__(self, N, model, weights, R_weights, upperbound, lowerbound,
                 lambda_u, energy_weights, speed_weight, u_ref):
        self.N = int(N)
        self.model = model
        self.weights = np.asarray(weights, dtype=float).reshape(-1)   # [Qc, Ql, Qpsi, Qv]
        self.R_weights = np.asarray(R_weights, dtype=float)

        self.upperbound = None
        self.lowerbound = None
        self.calc_upperbound(upperbound)
        self.calc_lowerbound(lowerbound)

        self.lambda_u = float(lambda_u)
        self.energy_weights = np.asarray(energy_weights, dtype=float).reshape(-1)

        # Added: Constant true forward speed term
        self.speed_weight = float(speed_weight)
        self.u_ref = float(u_ref)

    def calc_upperbound(self, upperbound):
        upperbound = np.asarray(upperbound, dtype=float).reshape(-1)
        m = len(upperbound)
        thrust_max = np.zeros(m * self.N, dtype=float)
        for i in range(self.N):
            thrust_max[i * m:(i + 1) * m] = upperbound
        self.upperbound = thrust_max

    def calc_lowerbound(self, lowerbound):
        lowerbound = np.asarray(lowerbound, dtype=float).reshape(-1)
        m = len(lowerbound)
        thrust_min = np.zeros(m * self.N, dtype=float)
        for i in range(self.N):
            thrust_min[i * m:(i + 1) * m] = lowerbound
        self.lowerbound = thrust_min

    def mpcc_cost(self, u, X0_auv, theta0, PathData, v_target, dt):
        Hp = self.N
        nu = 4  # [Fu, Fv, Fr, v_theta]

        Qc = self.weights[0]
        Ql = self.weights[1]
        Qpsi = self.weights[2]
        Qv = self.weights[3]
        R = self.R_weights

        u = np.asarray(u, dtype=float).reshape(-1)
        X0_auv = np.asarray(X0_auv, dtype=float).reshape(-1)

        U = u.reshape(Hp, nu).T   # shape: (4, Hp)
        X_auv = np.zeros((6, Hp), dtype=float)
        theta_pred = np.zeros(Hp, dtype=float)

        # State prediction using RK4
        Xplus = self.model.dynamics_discrete(X0_auv, U[0:3, 0], dt)
        X_auv[:, 0] = Xplus
        theta_pred[0] = theta0 + U[3, 0] * dt

        for i in range(1, Hp):
            Xplus = self.model.dynamics_discrete(Xplus, U[0:3, i], dt)
            X_auv[:, i] = Xplus
            theta_pred[i] = theta_pred[i - 1] + U[3, i] * dt

        cost = 0.0
        w1, w2, w3 = self.energy_weights

        for i in range(Hp):
            x_auv = X_auv[0, i]
            y_auv = X_auv[1, i]
            psi_auv = X_auv[2, i]
            u_auv = X_auv[3, i]
            v_theta = U[3, i]

            xr = PathData["Fx"](theta_pred[i])
            yr = PathData["Fy"](theta_pred[i])
            psi_r = PathData["Fpsi"](theta_pred[i])

            dx = x_auv - xr
            dy = y_auv - yr

            ec = -np.sin(psi_r) * dx + np.cos(psi_r) * dy
            el =  np.cos(psi_r) * dx + np.sin(psi_r) * dy

            epsi = psi_auv - psi_r
            epsi = np.arctan2(np.sin(epsi), np.cos(epsi))

            J_contour = Qc * ec**2 + Ql * el**2 + Qpsi * epsi**2
            J_speed = Qv * (v_theta - v_target)**2
            J_u_const = self.speed_weight * (u_auv - self.u_ref)**2

            J_energy = self.lambda_u * (
                w1 * U[0, i]**2 + w2 * U[1, i]**2 + w3 * U[2, i]**2
            ) * dt

            Te = U[0:3, i]
            J_control = Te.T @ R @ Te

            cost += J_contour + J_speed + J_u_const + J_energy + J_control

        return float(cost)

    def calc_control(self, u0, X0_auv, theta0, PathData, v_target, dt):
        u0 = np.asarray(u0, dtype=float).reshape(-1)

        bounds = Bounds(self.lowerbound, self.upperbound)

        result = minimize(
            fun=lambda u: self.mpcc_cost(u, X0_auv, theta0, PathData, v_target, dt),
            x0=u0,
            method='SLSQP',
            bounds=bounds,
            options={
                'disp': False,
                'ftol': 1e-3,
                'maxiter': 2000
            }
        )

        u = result.x
        return u.reshape(self.N, 4).T  # shape: (4, N)