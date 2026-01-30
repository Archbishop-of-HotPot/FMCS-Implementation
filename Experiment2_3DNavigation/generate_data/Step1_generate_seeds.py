import os
import pickle
import numpy as np
import cvxpy as cp
from scipy.interpolate import BSpline

#config
SAVE_DIR ="../dataset_source"
MAX_ACC = 40.0
NUM_PTS = 32
DEGREE = 5

LABEL_MAP = {
    "top": 0,  
    "side": 1,  
    "bottom": 2  
}


def get_basis(num_pts, degree, num_samples=60):
    k = degree
    knots = np.concatenate(([0] * k, np.linspace(0, 1, num_pts - k + 1), [1] * k))
    t = np.linspace(0, 1, num_samples)
    eye = np.eye(num_pts)
    H_acc = np.array([BSpline(knots, eye[i], k).derivative(2)(t) for i in range(num_pts)]).T
    return H_acc


def solve_qp(start, end, strategy, obstacles, bounds):
    H_acc = get_basis(NUM_PTS, DEGREE)
    P = cp.Variable((NUM_PTS, 3))
    cost = cp.sum_squares(cp.diff(P, k=3, axis=0))  # Min Jerk

    constraints = [P[0] == start, P[-1] == end]

    mid = slice(int(NUM_PTS * 0.35), int(NUM_PTS * 0.65))
    margin = 0.8

    if strategy == 'top':
        constraints.append(P[mid, 2] >= bounds["z_max"] + margin)
    elif strategy == 'front': 
        constraints.append(P[mid, 1] <= bounds["y_min"] - margin)
    elif strategy == 'bottom':
        constraints.append(P[mid, 2] <= 1.5) 

    # acceleration
    constraints.append(cp.abs(H_acc @ P) <= MAX_ACC)

    prob = cp.Problem(cp.Minimize(cost), constraints)
    try:
        prob.solve(solver=cp.ECOS)
    except:
        try:
            prob.solve()
        except:
            return None

    if prob.status in ["optimal", "optimal_inaccurate"]:
        return P.value
    return None


def main():
    if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)

    obstacles = {
        "cube": {"min": np.array([2, 2, 2]), "max": np.array([4, 4, 4])},
        "sphere": {"center": np.array([5, 3, 3]), "radius": 1.2}
    }
    bounds = {"y_min": 2.0, "z_max": 4.2}

    start = np.array([0, 3, 3])
    end_A = np.array([8, 3, 3])
    end_B = np.array([8, 3, 0])

    tasks = [
        {"name": "top", "s": start, "e": end_A, "strat": "top"},
        {"name": "side", "s": start, "e": end_A, "strat": "front"},
        {"name": "bottom", "s": start, "e": end_B, "strat": "bottom"},
    ]

    # generate_seed_data
    seed_data = {}
    print("Calculating Seeds...")

    for t in tasks:
        cps = solve_qp(t["s"], t["e"], t["strat"], obstacles, bounds)

        if cps is not None:
            label_id = LABEL_MAP[t["name"]]

            seed_data[t["name"]] = {
                "cps": cps,  # (32, 3) 
                "label": label_id  # int     label (0, 1, 2)
            }
            print(f" -> [{t['name']}] success! | Label: {label_id}")
        else:
            print(f" -> [{t['name']}] failed!")

    # save
    seed_filename = f"{SAVE_DIR}/expert_seeds_pts{NUM_PTS}.npy"
    np.save(seed_filename, seed_data)

    config_data = {
        "obstacles": obstacles,
        "label_map": LABEL_MAP
    }
    with open(f"{SAVE_DIR}/env_config.pkl", "wb") as f:
        pickle.dump(config_data, f)

    print(f"\nSeeds_and_data Label save to  {SAVE_DIR}/expert_seeds_pts{NUM_PTS}.npy")


if __name__ == "__main__":
    main()