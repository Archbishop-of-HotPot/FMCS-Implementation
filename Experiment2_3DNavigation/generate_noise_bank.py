import os
import argparse
import numpy as np
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import time

# Solver
from bspline_qp_solver import ExactBsplineSolver


solver_instance = None
data_mean = None
data_std = None

def init_worker(mean, std, n_cps, degree, dim, total_time, limit, factor):
    global solver_instance, data_mean, data_std, std_factor
    data_mean = mean
    data_std = std
    std_factor = factor
    solver_instance = ExactBsplineSolver(
        n_cps=n_cps, degree=degree, dim=dim, T=total_time, limit=limit, knot_type='clamped'
    )

def process_sample(seed):
    global solver_instance, data_mean, data_std, std_factor
    rng = np.random.RandomState(seed)
    raw_noise = rng.randn(solver_instance.n_cps, solver_instance.dim) * data_std * std_factor + data_mean
    proj_noise = solver_instance.solve(raw_noise)
    if proj_noise is None: return None
    return raw_noise.astype(np.float32), proj_noise.astype(np.float32)

#main
def generate_bank(args):
    print(f"[Noise Bank Generator] Start...")
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Data file not found: {args.data_path}")

    try:
        if args.data_path.endswith('.npz'):
            data_pack = np.load(args.data_path)
            if 'trajs' in data_pack:
                raw_data = data_pack['trajs']
            else:
                first_key = data_pack.files[0]
                raw_data = data_pack[first_key]
        else:
            raw_data = np.load(args.data_path)
        
        #check
        if len(raw_data.shape) == 3:
            if raw_data.shape[-1] != args.dim and raw_data.shape[1] == args.dim:
                raw_data = raw_data.transpose(0, 2, 1)
            
        mean = raw_data.mean(axis=0) 
        std = raw_data.std(axis=0)
        print(f"Statistics loaded. Shape: {raw_data.shape}")
        
    except Exception as e:
        print(f" Error loading data: {e}")
        return

    n_cores = args.cores if args.cores > 0 else max(1, cpu_count() - 2)
    print(f" Using {n_cores} CPU cores.")
    
    raw_list, proj_list = [], []
    init_args = (mean, std, args.n_cps, args.degree, args.dim, args.time, args.limit, args.std_factor)

    with Pool(processes=n_cores, initializer=init_worker, initargs=init_args) as pool:
        iterator = pool.imap(process_sample, range(args.n_samples), chunksize=1000)
        for res in tqdm(iterator, total=args.n_samples, unit="traj", desc="Generating"):
            if res is not None:
                r, p = res
                raw_list.append(r)
                proj_list.append(p)

    # numpy 
    raw_np = np.array(raw_list)
    proj_np = np.array(proj_list)


    print("\n" + "="*50)
    print("DATA STATISTICS REPORT")
    print("-"*50)
    print(f"1. Original Data ({raw_data.shape}):")
    print(f"   - Mean: {np.mean(raw_data):.6f}")
    print(f"   - Std:  {np.std(raw_data):.6f}")
    
    print(f"2. Raw Noise (Factor={args.std_factor}):")
    print(f"   - Mean: {np.mean(raw_np):.6f}")
    print(f"   - Std:  {np.std(raw_np):.6f}")
    
    print(f"3. Projected Noise:")
    print(f"   - Mean: {np.mean(proj_np):.6f}")
    print(f"   - Std:  {np.std(proj_np):.6f}")
    print("="*50 + "\n")

    print("Packing data...")
    save_dir = os.path.dirname(args.save_path)
    if save_dir and not os.path.exists(save_dir): os.makedirs(save_dir)

    np.savez_compressed(
        args.save_path,
        raw=raw_np,
        projected=proj_np,
        data_mean=mean,
        data_std=std,
        config=vars(args)
    )
    print(f"Saved to: {args.save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--n_samples", type=int, default=500000)
    parser.add_argument("--limit", type=float, default=40.0)
    parser.add_argument("--n_cps", type=int, default=32)
    parser.add_argument("--degree", type=int, default=5)
    parser.add_argument("--dim", type=int, default=3)
    parser.add_argument("--time", type=float, default=1.0)
    parser.add_argument("--cores", type=int, default=-1)
    parser.add_argument("--std_factor", type=float, default=1)
    
    args = parser.parse_args()
    generate_bank(args)
