import os
import time
import pickle
import numpy as np
import cvxpy as cp
import plotly.graph_objects as go
from scipy.interpolate import BSpline

# config
SOURCE_DIR = "../dataset_source"
OUTPUT_DIR = "../dataset_training"

SAMPLES_PER_MODE = 1024  
NOISE_SCALE = 0.2  
ACC_LIMIT = 40.0  

# Solver
NUM_PTS = 32
DEGREE = 5
DIM = 3
TOTAL_TIME = 1.0



class BsplineProjector:
    def __init__(self, n_cps, degree, dim, T, limit):
        """
        B-spline Projector with Boundary Constraints.
        """
        self.limit = limit
        print(f"🔧 Init Projector: N={n_cps}, Limit={limit}, Boundary=Fixed")
        t0 = time.time()

        self.M = self._get_derivative_matrix(n_cps, degree, T, order=2)

        self.W = cp.Variable((n_cps, dim))
        self.init_cps_param = cp.Parameter((n_cps, dim))

        objective = cp.Minimize(cp.sum_squares(self.W - self.init_cps_param))

        constraints = []

        Acc_CPs = self.M @ self.W
        for i in range(Acc_CPs.shape[0]):
            constraints.append(cp.norm(Acc_CPs[i], 2) <= self.limit)

        #(Fixed Start/Goal)
        constraints.append(self.W[0] == self.init_cps_param[0])
        constraints.append(self.W[-1] == self.init_cps_param[-1])

        self.prob = cp.Problem(objective, constraints)

        #  Warm-up
        self.init_cps_param.value = np.zeros((n_cps, dim))
        try:
            self.prob.solve(solver=cp.CLARABEL, verbose=False)
        except:
            try:
                self.prob.solve(solver=cp.ECOS, verbose=False)
            except:
                pass

        print(f"Projector Ready! ({time.time() - t0:.4f}s)")

    def _get_derivative_matrix(self, n, p, T, order=2):
        n_inner = n - (p + 1)
        knots = np.concatenate([
            np.zeros(p + 1),
            np.linspace(0, 1, n_inner + 2)[1:-1],
            np.ones(p + 1)
        ])
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

    def solve(self, noisy_cps):
        self.init_cps_param.value = noisy_cps
        try:
            self.prob.solve(solver=cp.CLARABEL, verbose=False)
        except:
            self.prob.solve(solver=cp.ECOS, verbose=False)

        if self.prob.status not in ["optimal", "optimal_inaccurate"]:
            return None
        return self.W.value



def check_collision(cps, obstacles, degree=5):
    k = degree
    n = len(cps)
    knots = np.concatenate(([0] * k, np.linspace(0, 1, n - k + 1), [1] * k))
    spl = BSpline(knots, cps, k)

    t_eval = np.linspace(0, 1, 50)
    pts = spl(t_eval)
    margin = 0.05

    # Check Cube
    c_min, c_max = obstacles["cube"]["min"], obstacles["cube"]["max"]
    in_cube = np.all((pts >= c_min + margin) & (pts <= c_max - margin), axis=1)
    if np.any(in_cube): return False

    # Check Sphere
    s_c, s_r = obstacles["sphere"]["center"], obstacles["sphere"]["radius"]
    dists = np.linalg.norm(pts - s_c, axis=1)
    if np.any(dists < s_r + margin): return False

    return True



def main():
    print(f"Loading: {SOURCE_DIR} ...")
    seeds = np.load(f"{SOURCE_DIR}/expert_seeds_pts{NUM_PTS}.npy", allow_pickle=True).item()

    with open(f"{SOURCE_DIR}/env_config.pkl", "rb") as f:
         env_config = pickle.load(f)
         obstacles = env_config["obstacles"]

    print("Success!")


    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)


    projector = BsplineProjector(
        n_cps=NUM_PTS, degree=DEGREE, dim=DIM, T=TOTAL_TIME, limit=ACC_LIMIT
    )

    all_trajs = []
    all_labels = []

    for mode_name, data in seeds.items():
        seed_cps = data["cps"]
        label_id = data["label"]

        print(f"\n Processing Mode: [{mode_name}] (Label: {label_id})")

        valid_trajs_batch = []
        stats = {"ok": 0, "fail_solve": 0, "fail_coll": 0}

        # original_data
        valid_trajs_batch.append(seed_cps)
        #add noise
        while len(valid_trajs_batch) < SAMPLES_PER_MODE:
            noise = np.random.normal(0, NOISE_SCALE, size=seed_cps.shape)
            noise[0] = 0
            noise[-1] = 0
            noisy_inputs = seed_cps + noise

            # projectiom
            clean_cps = projector.solve(noisy_inputs)

            if clean_cps is None:
                stats["fail_solve"] += 1
                continue

            # check
            if check_collision(clean_cps, obstacles, degree=DEGREE):
                valid_trajs_batch.append(clean_cps)
                stats["ok"] += 1
            else:
                stats["fail_coll"] += 1

            print(f"   -> Progress: {len(valid_trajs_batch)}/{SAMPLES_PER_MODE} "
                  f"| Fail(QP):{stats['fail_solve']} | Fail(Coll):{stats['fail_coll']}", end='\r')

        print(f"   Done.")

        all_trajs.append(np.array(valid_trajs_batch))

        labels_batch = np.full(len(valid_trajs_batch), label_id, dtype=np.int64)
        all_labels.append(labels_batch)

    # Save
    final_trajs = np.concatenate(all_trajs, axis=0)  # (Total, 32, 3)
    final_labels = np.concatenate(all_labels, axis=0)  # (Total, )

    save_path = f"{OUTPUT_DIR}/dataset_final_pts{NUM_PTS}.npz"
    np.savez(save_path, trajs=final_trajs, labels=final_labels)

    print(f"\nSave path: {save_path}")
    print(f"Trajs Shape:  {final_trajs.shape}")
    print(f"Labels Shape: {final_labels.shape}")



if __name__ == "__main__":

    main()
