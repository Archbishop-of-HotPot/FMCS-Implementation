# FMCS-Implementation
## Installation
### 1. Install Dependencies
First, install the core libraries using the provided requirements file:
```bash
pip install -r requirements.txt
```

### 2. Install PyTorch
PyTorch is **not** included in `requirements.txt` to prevent OS/CUDA conflicts.

**Tested Environment:** PyTorch 2.3.0 + CUDA 12.1

While other versions may work, we recommend aligning with this version if you encounter dependency conflicts.

**For CUDA 12.1 (Recommended):**
```bash
pip install torch torchvision --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
```
## Experiments

### Experiment 1: Checkerboard
Run the following notebook directly to reproduce the checkerboard experiment results:
- **Notebook:** `Experiment1_checkerbord.ipynb`


### Experiment 2: 3D Navigation

**Step 1: Data Generation (Optional)**
> **Note:** Pre-generated example data is already included in the directory. You can skip this step and proceed directly to the main experiment.

If you wish to regenerate the dataset from scratch, run the following commands in order:

```bash
cd FMCS-Implementation/Experiment2_3DNavigation/generate_data
python Step1_generate_seeds.py
python Step2_Noise_Add.py
```

**Step 2: Main Experiment**

Run the main notebook to train and evaluate:
- **Notebook:** `Experiment2_3DNavigation/experiment_main.ipynb`
