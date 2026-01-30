import torch
import numpy as np
from torch.utils.data import Dataset


class TrajectoryNormalizer:
    def __init__(self, data_tensor):

        self.mean = data_tensor.mean(dim=(0, 2), keepdim=True)
        self.std = data_tensor.std(dim=(0, 2), keepdim=True) + 1e-3 #epsilon = 1e-3

    def normalize(self, x):
        return (x - self.mean.to(x.device)) / self.std.to(x.device)

    def unnormalize(self, x):
        return x * self.std.to(x.device) + self.mean.to(x.device)


class FlowDataset(Dataset):
    def __init__(self, data_path, p_uncond=0.1, num_classes=3):
        self.p_uncond = p_uncond
        self.null_label = num_classes
        
        # A. loading
        data = np.load(data_path)
        # (N, 32, 3) -> (N, 3, 32)
        x1_physical = torch.FloatTensor(data['trajs']).permute(0, 2, 1)
        self.labels = torch.LongTensor(data['labels'])

        # B. Normalize
        self.normalizer = TrajectoryNormalizer(x1_physical)
        
        # C. save
        self.x1 = self.normalizer.normalize(x1_physical)

        print(f"   FlowDataset initialized.")
        print(f"   Stats Mean: {self.normalizer.mean.view(-1).numpy()}")
        print(f"   Stats Std:  {self.normalizer.std.view(-1).numpy()}")

    def __len__(self):
        return len(self.x1)

    def __getitem__(self, idx):
        x1 = self.x1[idx]
        label = self.labels[idx].item()
        if np.random.rand() < self.p_uncond:
            label = self.null_label
        return x1, label


class NoiseBank:
    def __init__(self, noise_path, normalizer, mode='baseline', device='cuda'):

        bank = np.load(noise_path)
        key = 'raw' if mode == 'baseline' else 'projected'
        
        data_physical = torch.FloatTensor(bank[key]).permute(0, 2, 1)
        
        self.data = normalizer.normalize(data_physical).to(device)
        
        print(f" NoiseBank ({mode}) loaded & normalized. Shape: {self.data.shape}")

    def sample(self, batch_size):
        idx = torch.randint(0, len(self.data), (batch_size,), device=self.data.device)
        return self.data[idx]