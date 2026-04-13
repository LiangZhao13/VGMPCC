# ==============================================================================
# Author: Liang Zhao
# Paper: VG-MPCC: A Virtual-Guide Model Predictive Contouring Control Formulation 
#        for Path Following of Autonomous Underwater Vehicles
# ==============================================================================

import numpy as np
from scipy.interpolate import CubicSpline

# =========================================================
# Helper interpolator to mimic griddedInterpolant(...,'spline','nearest')
# Inside range: cubic spline interpolation
# Outside range: clamp to nearest boundary
# =========================================================
class ClampedSpline1D:
    def __init__(self, x, y):
        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self.spline = CubicSpline(self.x, self.y, extrapolate=True)
        self.xmin = self.x[0]
        self.xmax = self.x[-1]

    def __call__(self, xq):
        xq_arr = np.asarray(xq, dtype=float)
        xq_clip = np.clip(xq_arr, self.xmin, self.xmax)
        yq = self.spline(xq_clip)
        if np.isscalar(xq):
            return float(yq)
        return yq