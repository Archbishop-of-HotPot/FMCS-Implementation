# FMCS-Implementation
## Installation
**Prerequisites:** Python 3.12
### 1. Install Dependencies
First, install the core libraries using the provided requirements file:
```bash
pip install -r requirements.txt
```

### 2. Install PyTorch
PyTorch is **not** included in `requirements.txt` to prevent OS/CUDA conflicts. Please install the version compatible with your hardware manually.

**For CUDA 12.1 (Tested Environment):**
```bash
pip install torch torchvision --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
```

**For other versions (CPU / Mac / Different CUDA):**
Please install the correct version compatible with your system.
*(Recommended version: torch>=2.3.0)*

## Experiments

### Experiment 1: Checkerboard
Run the following notebook directly to reproduce the checkerboard experiment results:
- **Notebook:** `Experiment1_checkerbord.ipynb`


### Experiment 2: 3D Navigation

**Step 1: Data Generation (Optional)**
> **Note:** Pre-generated example data is already included in the directory. You can skip this step and proceed directly to the main experiment.

If you wish to regenerate the dataset from scratch, run the following commands in order:

```bash
python Experiment2_3DNavigation/generate_data/Step1_generate_seeds.py
python Experiment2_3DNavigation/generate_data/Step2_Noise_Add.py
```

**Step 2: Main Experiment**
Run the main notebook to train and evaluate:
- **Notebook:** `Experiment2_3DNavigation/experiment_main.ipynb`
