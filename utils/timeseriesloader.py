from torch.utils.data import Dataset
import numpy as np
import random 
import torch
from utils.features import getFeatures
import pandas as pd

class TimeSeriesDataset(Dataset):
    def __init__(self, path, augment=False):
        
        self.augment = augment
        self.path = path

    def __len__(self):
        return 1
    
    def __getitem__(self, idx):

        df = pd.read_parquet(self.path)
        x, y = df["x"].values, df["y"].values
        k_series, alpha_series, state_series = np.log10(df['D'].values + 1), df["alpha"].values, df["state"].values

        if self.augment:
            # AUGMENTATION 1: Random Truncation of time series based on start-end indices
            if np.random.rand() < 0.3 and len(x) >= 24: # challenge constraint - minimum length for time series

                start_idx = random.randint(0, len(x) - 24) 
                end_idx = start_idx + random.randint(24, len(x) - start_idx)
                # challenge constraint - a state needs to have at least 3 unique alpha values
                while not (alpha_series[start_idx] == alpha_series[start_idx+1] and alpha_series[start_idx+1] == alpha_series[start_idx+2]):
                    start_idx += 1 

                while not (alpha_series[end_idx - 1] == alpha_series[end_idx-2] and alpha_series[end_idx-2] == alpha_series[end_idx-3]):
                    end_idx -= 1
            
            else:
                start_idx = 0
                end_idx = len(x)
            
            x = x[start_idx:end_idx]
            y = y[start_idx:end_idx]
                
            # AUGMENTATION 2: Random Rotation of coordinates from [0,2*pi]
            if np.random.rand() < 0.3:
                # rotate the coordinates 
                angle = np.random.rand() * 2 * np.pi
                x_rot = x * np.cos(angle) - y * np.sin(angle)
                y_rot = x * np.sin(angle) + y * np.cos(angle)

                x = x_rot
                y = y_rot
            
            # AUGMENTATION 3: Gaussian noise of std 0.1 mean 0 
            if np.random.rand() < 0.3:
                x += np.random.normal(0, 0.1, len(x))
                y += np.random.normal(0, 0.1, len(y))

            # AUGMENTATION 4: Flip along the x-axis
            if np.random.rand() < 0.3:
                x = -x            
            # AUGMENTATION 5: Flip along the y-axis
            if np.random.rand() < 0.3:
                y = -y

            features = torch.tensor(np.nan_to_num(getFeatures(x, y), nan=0.0, posinf=0.0, neginf=0.0), dtype=torch.float32)            
            alpha_series = torch.tensor(alpha_series[start_idx:end_idx], dtype=torch.float32)
            k_series = torch.tensor(k_series[start_idx:end_idx], dtype=torch.float32)
            state_series = torch.tensor(state_series[start_idx:end_idx], dtype=torch.float32)
            return features, alpha_series, k_series, state_series   

        else:
            features = torch.tensor(np.nan_to_num(getFeatures(x, y), nan=0.0, posinf=0.0, neginf=0.0), dtype=torch.float32)            
            alpha_series = torch.tensor(alpha_series, dtype=torch.float32)
            k_series = torch.tensor(k_series, dtype=torch.float32)
            state_series = torch.tensor(state_series, dtype=torch.float32)
            return features, alpha_series, k_series, state_series