import time
import numpy as np
import cvxpy as cp
from scipy.interpolate import BSpline


class ExactBsplineSolver:
    def __init__(self, n_cps, degree, dim, T, limit, knot_type='uniform'):
        """
        Args:
            n_cps (int): Total number of control points (e.g., 40).
            degree (int): Degree of the B-spline (typically 3).
            dim (int): Spatial dimension (e.g., 2).
            T (float): Total duration.
            limit (float): Upper bound for the acceleration L2 norm.
            knot_type (str): 'uniform' (default) or 'clamped'.
                'uniform': A B-spline with floating ends (the curve does not necessarily pass through the first/last control points).
                'clamped': A B-spline constrained to pass through (interpolate) the first and last control points.
        """
        self.n_cps = n_cps
        self.degree = degree
        self.dim = dim
        self.dt = T
        self.limit = limit
        self.knot_type = knot_type.lower()

        # print(f"🔧 Init Solver ({self.knot_type.capitalize()} Full-Constraint): N={n_cps}, Limit={limit}")
        t0 = time.time()

        self.M = self._get_derivative_matrix(n_cps, degree, T, order=2, knot_type=self.knot_type)

        self.W = cp.Variable((n_cps, dim))
        self.init_cps_param = cp.Parameter((n_cps, dim))

        objective = cp.Minimize(cp.sum_squares(self.W - self.init_cps_param))

        Acc_CPs = self.M @ self.W
        constraints = []

        for i in range(Acc_CPs.shape[0]):
            constraints.append(cp.norm(Acc_CPs[i], 2) <= self.limit)

        self.prob = cp.Problem(objective, constraints)

        self.init_cps_param.value = np.zeros((n_cps, dim))
        try:
            self.prob.solve(solver=cp.CLARABEL, verbose=False)
        except Exception as e:
            print(f"⚠️ Warm-up failed: {e}")


    def _get_derivative_matrix(self, n, p, T, order=2, knot_type='uniform'):

        if knot_type == 'uniform':

            n_knots = n + p + 1
            knots = np.linspace(0, 1, n_knots)
        elif knot_type == 'clamped':
            #  [0,0,0,0, t1, t2, ..., 1,1,1,1]
            n_inner = n - (p + 1)
            knots = np.concatenate([
                np.zeros(p + 1),
                np.linspace(0, 1, n_inner + 2)[1:-1],
                np.ones(p + 1)
            ])
        else:
            raise ValueError(f"Unknown knot_type: {knot_type}. Use 'uniform' or 'clamped'.")

        target_n = n - order
        M = np.zeros((target_n, n))
        I = np.eye(n)

        for i in range(n):
            c = I[:, i]
            spl = BSpline(knots, c, k=p)
            deriv = spl.derivative(order)
            M[:, i] = deriv.c[:target_n]

        M *= (1.0 / T ** order)
        return M

    def solve(self, init_cps):
        """
        Args:
            init_cps: ControlPoint (N, dim)
        Returns:
            Projected CPs (N, dim) or None
        """
        self.init_cps_param.value = init_cps

        try:
            self.prob.solve(solver=cp.CLARABEL, verbose=False, warm_start=True)
        except Exception as e:
            print(f"Solver Error: {e}")
            return None

        if self.prob.status not in ["optimal", "optimal_inaccurate"]:
            return None

        return self.W.value
