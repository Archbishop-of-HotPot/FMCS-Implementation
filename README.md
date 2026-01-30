# FMCS-Implementation
## Installation

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
