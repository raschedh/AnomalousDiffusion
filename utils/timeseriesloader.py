from torch.utils.data import Dataset
import numpy as np
import random 
import torch
from utils.features import getFeatures

class TimeSeriesDataset(Dataset):
    def __init__(self, df, augment=False):
        
        self.augment = augment

        self.x = df["x"].values
        self.y = df["y"].values

        self.D_values = np.log10(df['D'].values + 1)
        self.alpha_values = df["alpha"].values
        self.state_values = df["state"].values

    def __len__(self):
        return 1
    
    def __getitem__(self, idx):

        if self.augment:

            # AUGMENTATION 1: Random Truncation of time series based on start-end indices
            series_length = len(self.x)
            if np.random.rand() < 0.3 and series_length >= 24: # challenge constraint - minimum length for time series

                start_idx = random.randint(0, series_length - 24) 
                end_idx = start_idx + random.randint(24, series_length - start_idx)

                # challenge constraint - a state needs to have at least 3 unique alpha values
                while not (self.alpha_values[start_idx] == self.alpha_values[start_idx+1] and self.alpha_values[start_idx+1] == self.alpha_values[start_idx+2]):
                    start_idx += 1 

                while not (self.alpha_values[end_idx - 1] == self.alpha_values[end_idx-2] and self.alpha_values[end_idx-2] == self.alpha_values[end_idx-3]):
                    end_idx -= 1
            
            else:
                start_idx = 0
                end_idx = series_length
                
            # AUGMENTATION 2: Random Rotation of coordinates from [0,2*pi]
            if np.random.rand() < 0.3:
                x_copy = self.x.copy()
                y_copy = self.y.copy()
                # rotate the coordinates 
                angle = np.random.rand() * 2 * np.pi
                x_rot = x_copy * np.cos(angle) - y_copy * np.sin(angle)
                y_rot = x_copy * np.sin(angle) + y_copy * np.cos(angle)

                x_rot = x_rot[start_idx:end_idx]
                y_rot = y_rot[start_idx:end_idx]
            
            else:
                x_rot = self.x.copy()[start_idx:end_idx]
                y_rot = self.y.copy()[start_idx:end_idx]
            
            # AUGMENTATION 3: Gaussian noise of std 0.1 mean 0 
            if np.random.rand() < 0.3:
                x_rot += np.random.normal(0, 0.1, len(x_rot))
                y_rot += np.random.normal(0, 0.1, len(y_rot))

            # AUGMENTATION 4: Flip along the x-axis
            if np.random.rand() < 0.3:
                x_rot = -x_rot
                y_rot = y_rot
            
            # AUGMENTATION 5: Flip along the y-axis
            if np.random.rand() < 0.3:
                x_rot = x_rot
                y_rot = -y_rot
            

            features = np.nan_to_num(getFeatures(x_rot, y_rot), nan=0.0, posinf=0.0, neginf=0.0)
            features = torch.tensor(features, dtype=torch.float32)
            
            alpha_label = torch.tensor(self.alpha_values[start_idx:end_idx], dtype=torch.float32)
            D_label = torch.tensor(self.D_values[start_idx:end_idx], dtype=torch.float32)
            state_label = torch.tensor(self.state_values[start_idx:end_idx], dtype=torch.float32)

            return features, alpha_label, D_label, state_label   

        else:

            features = np.nan_to_num(getFeatures(self.x, self.y), nan=0.0, posinf=0.0, neginf=0.0)
            alpha_label = torch.tensor(self.alpha_values, dtype=torch.float32)
            D_label = torch.tensor(self.D_values, dtype=torch.float32)
            state_label = torch.tensor(self.state_values, dtype=torch.float32)
            features = torch.tensor(features, dtype=torch.float32)

            return features, alpha_label, D_label, state_label 