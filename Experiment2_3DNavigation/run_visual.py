import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline
from bspline_qp_solver import ExactBsplineSolver
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.gridspec import GridSpec 
import matplotlib.colors as mcolors 
from matplotlib.lines import Line2D  
import os
from types import SimpleNamespace
from model import Unet1D
from dataset import FlowDataset
from bspline_qp_solver import ExactBsplineSolver


time_val = [0,0.5,0.8,0.9,1.0]
DENSITY_STRIDE = 2 #should be dived by steps
CONFIG = {
    "SAMPLE_IDX": 222,
    "STEPS": 30,
    "ACC_LIMIT": 50.0,
    "SLICE_INDICES": [1, 15, 24, 28, 30],

    "SAVE_DIR": "saved_models",
    "NOISE_BANK_PATH": "dataset_training/noise_bank_1000k.npz",
    "DATASET_PATH": "dataset_training/dataset_final.npz",
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu"
}


##inference
# 1.inferenceEngine
class InferenceEngine:
    def __init__(self, device):
        self.device = device

    @torch.no_grad()
    def infer_history(self, model, x0, cond, steps, method, normalizer):
        history = []
        x_t = x0.clone()
        def record(xt):
            phys = normalizer.unnormalize(xt).squeeze(0).permute(1, 0).cpu().numpy()
            history.append(phys)
        record(x_t)  # Step 0
        dt = 1.0 / steps
        for i in range(steps):
            t_in = torch.tensor([i / steps], device=self.device)
            v = model(x_t, t_in, cond=cond)
            x_t = x_t + v * dt
            record(x_t)
        return history

# 2.Loading models
def load_model(key, save_dir, device):
    prefix = {"fm_baseline": "fm_baseline", "ours": "fm_ours_qp"}.get(key, key)
    import glob
    files = glob.glob(os.path.join(save_dir, f"{prefix}_*.pth"))
    if not files:
        print(f"⚠️ Warning: Model {key} not found in {save_dir}")
        return None, None
    path = max(files, key=os.path.getmtime)
    print(f"   [Model] Load: {os.path.basename(path)}")
    ckpt = torch.load(path, map_location=device)
    if 'config' in ckpt:
        cfg = SimpleNamespace(**ckpt['config'])
    else:
        cfg = SimpleNamespace(MODEL_DIM=32, MODEL_CHANNELS=32, MODEL_MULTS=[1, 2, 4], NUM_CLASSES=3)

    model = Unet1D(
        input_dim=getattr(cfg, 'MODEL_DIM', 32),
        base_channels=getattr(cfg, 'MODEL_CHANNELS', 32),
        dim_mults=getattr(cfg, 'MODEL_MULTS', [1, 2, 4]),
        num_classes=getattr(cfg, 'NUM_CLASSES', 3)
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model, cfg

# Generate function
def generate_viz_data(config):
    print(f"⚡ [Backend] Generating Data (Sample={config['SAMPLE_IDX']})...")
    DEVICE = config['DEVICE']

    # A. Dataset
    dpath = config['DATASET_PATH']
    if not os.path.exists(dpath):
        if os.path.exists("dataset_final.npz"):
            dpath = "dataset_final.npz"
        else:
            raise FileNotFoundError(f" Failed to find dataset: {dpath}")

    dataset = FlowDataset(dpath)
    normalizer = dataset.normalizer

    # B. NoiseBank
    if not os.path.exists(config['NOISE_BANK_PATH']):
        raise FileNotFoundError(f"Failed to find NoiseBank: {config['NOISE_BANK_PATH']}")

    bank = np.load(config['NOISE_BANK_PATH'])

    # Baseline Input
    raw_np = bank['raw'][config['SAMPLE_IDX']]
    if raw_np.shape[-1] == 3: raw_np = raw_np.T
    x0_base = normalizer.normalize(torch.tensor(raw_np).float().to(DEVICE).unsqueeze(0))

    # Ours Input
    proj_np = bank['projected'][config['SAMPLE_IDX']]
    if proj_np.shape[-1] == 3: proj_np = proj_np.T
    x0_ours = normalizer.normalize(torch.tensor(proj_np).float().to(DEVICE).unsqueeze(0))

    # C. Infer
    engine = InferenceEngine(DEVICE)
    model_base, cfg = load_model("fm_baseline", config['SAVE_DIR'], DEVICE)
    model_ours, _ = load_model("ours", config['SAVE_DIR'], DEVICE)

    if not model_base or not model_ours:
        raise RuntimeError("Failed to load models，check paths of saved_models please")

    hist_base, hist_ours = {}, {}
    steps = config['STEPS']

    for lid in [0, 1, 2]:
        cond = torch.tensor([lid], device=DEVICE).long()
        hist_base[lid] = engine.infer_history(model_base, x0_base, cond, steps, 'fm', normalizer)
        hist_ours[lid] = engine.infer_history(model_ours, x0_ours, cond, steps, 'fm', normalizer)

    return {
        "h_base": hist_base,
        "h_ours": hist_ours,
        "model_cfg": cfg
    }
# ==========================================
# 2. 绘图辅助
# ==========================================
STYLES = {
    0: {"c": "#1f77b4", "lw": 2.5, "z": 3},  # Top (Blue)
    1: {"c": "#ff7f0e", "lw": 2.5, "z": 2},  # Side (Orange)
    2: {"c": "#2ca02c", "lw": 2.5, "z": 1}  # Bottom (Green)
}

ENV = {
    "start": [0, 3, 3], "end_A": [8, 3, 3], "end_B": [8, 3, 0],
    "cube_min": [2, 2, 2], "cube_max": [4, 4, 4], "sphere_c": [5, 3, 3], "sphere_r": 1.2
}


def calc_acc(traj, cfg):
    k, n = getattr(cfg, 'DEGREE', 5), getattr(cfg, 'NUM_PTS', 32)
    t = np.linspace(0, 1, 100)
    knots = np.concatenate(([0] * k, np.linspace(0, 1, n - k + 1), [1] * k))
    H = np.array([BSpline(knots, np.eye(n)[i], k).derivative(2)(t) for i in range(n)]).T
    acc_vec = (H @ traj) / (getattr(cfg, 'TOTAL_TIME', 1.0) ** 2)
    acc_norm = np.linalg.norm(acc_vec, axis=1)
    return t[3:-3], acc_norm[3:-3]


def draw_env(ax):
    ax.scatter(*ENV['start'], c='k', s=60, label='Start', zorder=500, edgecolors='white')
    ax.scatter(*ENV['end_A'], c='gray', marker='D', s=40, label='End A', zorder=500)
    ax.scatter(*ENV['end_B'], c='purple', marker='D', s=40, label='End B', zorder=500)

    # Cube (zorder=350)
    m, M = ENV['cube_min'], ENV['cube_max']
    corners = [[m[0], m[1], m[2]], [M[0], m[1], m[2]], [M[0], M[1], m[2]], [m[0], M[1], m[2]], [m[0], m[1], M[2]],
               [M[0], m[1], M[2]], [M[0], M[1], M[2]], [m[0], M[1], M[2]]]
    faces = [[corners[0], corners[1], corners[2], corners[3]], [corners[4], corners[5], corners[6], corners[7]],
             [corners[0], corners[1], corners[5], corners[4]], [corners[2], corners[3], corners[7], corners[6]],
             [corners[1], corners[2], corners[6], corners[5]], [corners[4], corners[7], corners[3], corners[0]]]

    cube_poly = Poly3DCollection(faces, alpha=0.15, facecolor='#36454F', edgecolor='#1C2833', linewidths=0.5)
    cube_poly.set_zorder(350)
    ax.add_collection3d(cube_poly)

    # Sphere (zorder=350)
    u, v = np.mgrid[0:2 * np.pi:30j, 0:np.pi:15j]
    x = ENV['sphere_c'][0] + ENV['sphere_r'] * np.cos(u) * np.sin(v)
    y = ENV['sphere_c'][1] + ENV['sphere_r'] * np.sin(u) * np.sin(v)
    z = ENV['sphere_c'][2] + ENV['sphere_r'] * np.cos(v)
    ax.plot_surface(x, y, z, color='#8B0000', alpha=0.15, linewidth=0, rstride=1, cstride=1, zorder=350)

    ax.set_xlim(0, 8);
    ax.set_ylim(0, 6);
    ax.set_zlim(0, 5)
    ax.set_box_aspect((8, 6, 5))


# ==========================================
# Baseline Flow
def draw_particle_flow(ax, hist_steps, color_hex, base_z):
    traj_stack = np.stack(hist_steps, axis=0)  # (T, 32, 3)
    particle_trajs = np.transpose(traj_stack, (1, 0, 2))  # (32, T, 3)
    n_particles, n_time, _ = particle_trajs.shape
    cmap = mcolors.LinearSegmentedColormap.from_list("flow_p", ["#d0d0d0", color_hex])

    for t in range(n_time - 1):
        progress = t / (n_time - 1)
        seg_alpha = 0.2 + 0.7 * np.sin(progress * np.pi)
        seg_color = cmap(progress)
        p_start = particle_trajs[:, t, :]
        p_end = particle_trajs[:, t + 1, :]
        for i in range(n_particles):
            ax.plot([p_start[i, 0], p_end[i, 0]], [p_start[i, 1], p_end[i, 1]], [p_start[i, 2], p_end[i, 2]],
                    color=seg_color, alpha=seg_alpha, lw=1.8,
                    zorder=base_z + t * 0.1)

# =======================================================
# Ours flow
def draw_trajectory_flow(ax, hist_steps, color_hex, base_z):
    total_frames = len(hist_steps)
    cmap = mcolors.LinearSegmentedColormap.from_list("flow_t", ["#ffffff", color_hex])
    for i in range(1, total_frames - 1):
        if i % DENSITY_STRIDE != 0:
            continue
        traj = hist_steps[i]
        progress = i / (total_frames - 1)
        alpha = 0.2 + 0.7 * (progress ** 2)
        color = cmap(progress)
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2],
                c=color, lw=2.0, alpha=alpha, zorder=base_z + i * 0.1)




# ==========================================
def draw_custom_plot(ax, data, method_name):
    history_dict = data['h_base'] if method_name == 'Baseline' else data['h_ours']
    draw_env(ax)  

    ax.set_title(f"{method_name} Flow Overview", fontsize=14, fontweight='bold')
    # ax.set_xticks([]);
    # ax.set_yticks([]);
    # ax.set_zticks([])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.grid(True) 

    solver = ExactBsplineSolver(32, 5, 3, 1.0, 40.0, 'clamped')
    draw_order = [2, 1, 0]  
    for lid in draw_order:
        hist_steps = history_dict[lid]
        target_color = STYLES[lid]['c']
        base_z = (4 - lid) * 100
        
        if method_name == 'Baseline':
            draw_particle_flow(ax, hist_steps, target_color, base_z)
        else:
            draw_trajectory_flow(ax, hist_steps, target_color, base_z)
            
        traj_0 = hist_steps[0]
        if method_name == 'Baseline':
            ax.scatter(traj_0[:, 0], traj_0[:, 1], traj_0[:, 2],
                       c='lightgray', s=30, alpha=0.8,
                       edgecolors='black', linewidths=0.8,
                       zorder= 10)
        else:
            ax.plot(traj_0[:, 0], traj_0[:, 1], traj_0[:, 2],
                    c='black', lw=4.5, alpha=0.7, zorder= 10)
            ax.plot(traj_0[:, 0], traj_0[:, 1], traj_0[:, 2],
                    c='#d0d0d0', lw=2.5, alpha=1.0, zorder= 11)

        traj_final = hist_steps[-1]

        ax.plot(traj_final[:, 0], traj_final[:, 1], traj_final[:, 2],
                c=target_color, lw=3.5, alpha=1.0, zorder=base_z + 100)

        ax.scatter(traj_final[:, 0], traj_final[:, 1], traj_final[:, 2],
                   c=target_color, s=15, alpha=1.0,
                   edgecolors='white', linewidths=0.8,
                   zorder=base_z + 101)


# ==========================================
def main(user_config=None): 
    if user_config is not None:
        CONFIG.update(user_config)
    data = generate_viz_data(CONFIG)
    h_base = data['h_base']
    h_ours = data['h_ours']

    slices = CONFIG['SLICE_INDICES']
    n_slice_cols = len(slices)

    # width_ratios: The width of the large image on the left is 1.5 times the width of a single slice on the right (you can adjust this 1.5).
    fig = plt.figure(figsize=(3.5 * n_slice_cols + 5, 10))
    gs = GridSpec(4, n_slice_cols + 1, figure=fig, width_ratios=[2] + [1] * n_slice_cols,
                  height_ratios=[2, 1, 2, 1])

    solver = ExactBsplineSolver(32, 5, 3, 1.0, 40.0, 'clamped')

    # ==========================================
    # A. Draw the two large images on the left (placeholder)
    legend_elements = [
        Line2D([0], [0], color=STYLES[0]['c'], lw=3, label='Traj A'),
        Line2D([0], [0], color=STYLES[1]['c'], lw=3, label='Traj B'),
        Line2D([0], [0], color=STYLES[2]['c'], lw=3, label='Traj C')
    ]
    leg_kws = {'handles': legend_elements, 'loc': 'lower left',
               'bbox_to_anchor': (0,0), 'frameon': False, 'fontsize': 10}
    ax_big_base = fig.add_subplot(gs[0:2, 0], projection='3d')
    draw_custom_plot(ax_big_base, data, "Baseline")
    ax_big_base.legend(**leg_kws)

    # 2. Ours 
    ax_big_ours = fig.add_subplot(gs[2:4, 0], projection='3d')
    draw_custom_plot(ax_big_ours, data, "Ours")
    ax_big_ours.legend(**leg_kws)

    for i, step in enumerate(slices):
        safe_step = min(step, CONFIG['STEPS'])
        col_idx = i + 1  

        # --- Row 1: Base 3D ---
        ax0 = fig.add_subplot(gs[0, col_idx], projection='3d')
        # ax0.set_title(f"Baseline (Step {step})", fontsize=11, fontweight='bold')
        ax0.set_title(f"t = {time_val[i]}s", fontsize=12, fontweight='bold')
        draw_env(ax0)

        for lid in [0, 1, 2]:
            t = h_base[lid][safe_step]
            s = STYLES[lid]
            if step == CONFIG['STEPS']:
                ax0.plot(t[:, 0], t[:, 1], t[:, 2], c=s['c'], lw=s['lw'], zorder=s['z'])
            else:
                ax0.scatter(t[:, 0], t[:, 1], t[:, 2], c=s['c'], s=15, alpha=0.6, zorder=s['z'])

        # --- Row 2: Base Acc ---
        ax1 = fig.add_subplot(gs[1, col_idx])
        # ax1.set_title(f"Base Acc (Step {step})", fontsize=10)
        for lid in [0, 1, 2]:
            t_plot, acc = calc_acc(h_base[lid][safe_step], data['model_cfg'])
            s = STYLES[lid]
            ax1.plot(t_plot, acc, c=s['c'], lw=s['lw'], zorder=s['z'], alpha=0.9)
        ax1.axhline(CONFIG['ACC_LIMIT'], c='r', ls='--')
        ax1.grid(True, alpha=0.3)

        # --- Row 3: Ours 3D ---
        ax2 = fig.add_subplot(gs[2, col_idx], projection='3d')
        # ax2.set_title(f"Ours (Step {step})", fontsize=11, fontweight='bold')
        draw_env(ax2)
        for lid in [0, 1, 2]:
            t = h_ours[lid][safe_step]
            s = STYLES[lid]
            ax2.plot(t[:, 0], t[:, 1], t[:, 2], c=s['c'], lw=s['lw'], zorder=s['z'])
            if step == CONFIG['STEPS']:
                ax2.scatter(t[0, 0], t[0, 1], t[0, 2], c=s['c'], marker='o', s=20, zorder=s['z'] + 5)
                ax2.scatter(t[-1, 0], t[-1, 1], t[-1, 2], c=s['c'], marker='x', s=20, zorder=s['z'] + 5)

        # --- Row 4: Ours Acc ---
        ax3 = fig.add_subplot(gs[3, col_idx])
        # ax3.set_title(f"Ours Acc (Step {step})", fontsize=10)
        for lid in [0, 1, 2]:
            t_traj = h_ours[lid][safe_step]
            t_plot, acc = calc_acc(t_traj, data['model_cfg'])
            s = STYLES[lid]
            ax3.plot(t_plot, acc, c=s['c'], lw=s['lw'], zorder=s['z'], alpha=0.9)
        ax3.axhline(CONFIG['ACC_LIMIT'], c='r', ls='--')
        ax3.set_ylim(0, CONFIG['ACC_LIMIT'] * 1.6)
        ax3.grid(True, alpha=0.3)

        if i == 0:
            ax0.set_zlabel('Z');
            ax2.set_zlabel('Z')
            ax1.set_ylabel('Acceleration');
            ax3.set_ylabel('Acceleration')

    plt.tight_layout()
    plt.savefig(f"viz_final_sample{CONFIG['SAMPLE_IDX']}.png", dpi=200)
    print(f"Completed! Save to  viz_final_sample{CONFIG['SAMPLE_IDX']}.png")
    plt.show()


if __name__ == "__main__":
    main()