# ==============================================================================
# Author: Liang Zhao
# Paper: VG-MPCC: A Virtual-Guide Model Predictive Contouring Control Formulation 
#        for Path Following of Autonomous Underwater Vehicles
# ==============================================================================

import numpy as np
from abc import ABC, abstractmethod

# =========================================================
# Abstract base class: Model
# =========================================================
class Model(ABC):
    def __init__(self):
        self.X = None
        self.U = None

    @abstractmethod
    def dynamics_discrete(self, X, U, dt):
        pass

    @abstractmethod
    def ErrorState(self, X, Xr):
        pass

    @abstractmethod
    def F(self, X, Xe, P):
        pass

    def setState(self, X):
        self.X = np.asarray(X, dtype=float).reshape(-1)

    def setControl(self, U):
        self.U = np.asarray(U, dtype=float)

    def getState(self):
        print('System State:')
        print(self.X)

    def getControl(self):
        print('Control Input:')
        print(self.U)


# =========================================================
# AUV class
# =========================================================
class AUV(Model):
    def __init__(self, coef, ndof, X0, U0):
        super().__init__()

        if coef is None or ndof is None or X0 is None or U0 is None:
            raise ValueError("Four input arguments (coef, ndof, X0, U0) should be provided.")

        if ndof == 3:
            X0 = np.asarray(X0, dtype=float).reshape(-1)
            coef = np.asarray(coef, dtype=float).reshape(-1)

            assert len(X0) == 6, 'For 3 DOF AUV model X = [x;y;psi;u;v;r].'
            assert len(coef) == 11, 'coef = [m; Iz; X_udot; Y_vdot; N_rdot; Xu; Yv; Nr; Du; Dv; Dr]'

        self.Coef = np.asarray(coef, dtype=float).reshape(-1)
        self.Ndof = ndof
        self.DeducedCoef = np.array([283.6, 593.2, 29.0], dtype=float)
        self.calc_deduced_coef()
        self.X = np.asarray(X0, dtype=float).reshape(-1)
        self.U = np.asarray(U0, dtype=float)

    def calc_deduced_coef(self):
        if self.Ndof == 3:
            assert len(self.Coef) == 11, 'Number of model coefficients is incorrect: For 3 DOF AUV model we need 11 parameters.'
            m = self.Coef[0]
            Iz = self.Coef[1]
            X_udot = self.Coef[2]
            Y_vdot = self.Coef[3]
            N_rdot = self.Coef[4]
            Mx = m - X_udot
            My = m - Y_vdot
            Mpsi = Iz - N_rdot
            self.DeducedCoef = np.array([Mx, My, Mpsi], dtype=float)
        else:
            raise ValueError('The Ndof is currently not supported')

    def dynamics_continuous(self, X, U):
        """
        Calculates the continuous time derivatives X_dot = f(X, U)
        """
        m = 116.0
        Iz = 13.1
        X_udot = -100.6
        Y_vdot = -477.2
        N_rdot = -15.9
        Xu = 26.9
        Yv = 35.8
        Nr = 3.5
        Du = 100.3
        Dv = 503.8
        Dr = 76.9

        Mx = m - X_udot
        My = m - Y_vdot
        Mpsi = Iz - N_rdot

        Fu = U[0]
        Fv = U[1]
        Fr = U[2]

        psi = X[2]
        u = X[3]
        v = X[4]
        r = X[5]

        x_dot = u * np.cos(psi) - v * np.sin(psi)
        y_dot = u * np.sin(psi) + v * np.cos(psi)
        psi_dot = r
        u_dot = (My / Mx) * v * r - (Xu / Mx) * u - (Du / Mx) * u * abs(u) + Fu / Mx
        v_dot = -(Mx / My) * u * r - (Yv / My) * v - (Dv / My) * v * abs(v) + Fv / My
        r_dot = ((Mx - My) / Mpsi) * u * v - (Nr / Mpsi) * r - (Dr / Mpsi) * r * abs(r) + Fr / Mpsi

        return np.array([x_dot, y_dot, psi_dot, u_dot, v_dot, r_dot], dtype=float)

    def dynamics_discrete(self, X, U, dt):
        """
        Discrete dynamics using 4th-Order Runge-Kutta (RK4) Integration.
        """
        X = np.asarray(X, dtype=float).reshape(-1)
        U = np.asarray(U, dtype=float).reshape(-1)

        # RK4 Steps
        k1 = self.dynamics_continuous(X, U)
        k2 = self.dynamics_continuous(X + 0.5 * dt * k1, U)
        k3 = self.dynamics_continuous(X + 0.5 * dt * k2, U)
        k4 = self.dynamics_continuous(X + dt * k3, U)

        Xplus = X + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return Xplus

    def advance(self, U, W, dt):
        if self.Ndof == 3:
            U = np.asarray(U, dtype=float).reshape(-1)
            W = np.asarray(W, dtype=float).reshape(-1)
            disturbed_U = U + W
            self.X = self.dynamics_discrete(self.X, disturbed_U, dt)
            self.U = U
        else:
            raise ValueError('The Ndof is currently not supported')

    def ErrorState(self, X, Xr):
        if self.Ndof == 3:
            X = np.asarray(X, dtype=float).reshape(-1)
            Xr = np.asarray(Xr, dtype=float).reshape(-1)

            x, y, psi, u, v, r = X
            xr, yr, psi_r, ur, vr, rr = Xr

            xe = (xr - x) * np.cos(psi) + (yr - y) * np.sin(psi)
            ye = -(xr - x) * np.sin(psi) + (yr - y) * np.cos(psi)
            psi_e = psi_r - psi
            ue = u - ur * np.cos(psi_e) + vr * np.sin(psi_e)
            ve = v - ur * np.sin(psi_e) - vr * np.cos(psi_e)
            re = r - rr

            Xe = np.array([xe, ye, psi_e, ue, ve, re], dtype=float)
            return Xe
        else:
            raise ValueError('The Ndof is currently not supported')

    def F(self, X, Xe, P):
        """
        P = [xR;yR;psiR;uR;vR;rR;uRdot;vRdot;rRdot]
        """
        Mx = self.DeducedCoef[0]
        My = self.DeducedCoef[1]
        Mpsi = self.DeducedCoef[2]
        Xu = self.Coef[5]
        Yv = self.Coef[6]
        Nr = self.Coef[7]
        Du = self.Coef[8]
        Dv = self.Coef[9]
        Dr = self.Coef[10]

        X = np.asarray(X, dtype=float).reshape(-1)
        Xe = np.asarray(Xe, dtype=float).reshape(-1)
        P = np.asarray(P, dtype=float).reshape(-1)

        x, y, psi, u, v, r = X
        xe, ye, psi_e, ue, ve, re = Xe
        xr, yr, psi_r, ur, vr, rr, ur_dot, vr_dot, rr_dot = P

        f1 = (
            -My * v * r
            + Xu * u
            + Du * u * abs(u)
            + Mx * (
                ur_dot * np.cos(psi_e)
                + ur * np.sin(psi_e) * re
                - vr_dot * np.sin(psi_e)
                + vr * np.cos(psi_e) * re
            )
        )

        f2 = (
            Mx * u * r
            + Yv * v
            + Dv * v * abs(v)
            + My * (
                ur_dot * np.sin(psi_e)
                - ur * np.cos(psi_e) * re
                + vr_dot * np.cos(psi_e)
                + vr * np.sin(psi_e) * re
            )
        )

        f3 = (My - Mx) * u * v + Nr * r + Dr * r * abs(r) + Mpsi * rr_dot

        return f1, f2, f3