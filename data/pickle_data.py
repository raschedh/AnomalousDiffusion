import random 
import pickle
from sklearn.model_selection import train_test_split
import pandas as pd 
from multiprocessing import Pool, Manager
import numpy as np
from torch.utils.data import Dataset
import os

# This file creates a pickle file from the parquet files for faster training. It also uses multiprocessing to speed up the process.
# The __getitem__ method is not implemented here as it is not needed for this purpose.

class TimeSeriesDataset(Dataset):
    """
    Time Series Dataset class. Full Implementation in src/utils/timeseriesdataset.py
    """
    def __init__(self, df, augment=False):
        
        self.augment = augment

        self.x = df["x"].values
        self.y = df["y"].values

        self.k_series = np.log10(df["D"].values + 1)
        self.alpha_series = df["alpha"].values
        self.state_series = df["state"].values

    def __len__(self):
        return 1
    
    def __getitem__(self, idx):
        # not implemented here as not needed for this purpose, see utils for full implementation
        return  
        

def read_groups(data_paths):
    """
    Reads data from multiple parquet files and groups the data based on the 'traj_idx' column.
    
    Args:
        data_paths (list): A list of file paths to the parquet files.
        
    Returns:
        list: A list of grouped dataframes, where each dataframe represents a group of data.
    """
    
    groups = []
    counter = 0

    for path in data_paths:
        df = pd.read_parquet(path)

        for _, group in df.groupby("traj_idx"):
            group = group.drop(columns=["traj_idx"]).reset_index(drop=True)
            groups.append(group)
            counter += 1
            print("Groups: ", counter, end="\r")

    return groups

def split_groups(groups):
    """
    Splits a list of groups into training, validation, and test sets.

    Args:
        groups (list): A list of groups to be split.

    Returns:
        tuple: A tuple containing the training, validation, and test sets.

    """
    # splits in ratio 70:15:15
    random.shuffle(groups)
    train_data, temp_data = train_test_split(groups, test_size=0.3, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
    return train_data, val_data, test_data


def create_instance(item, augment, shared_counter, lock):
    """
    Creates an instance of the TimeSeriesDataset class for a given item.

    Args:
        item (pd.DataFrame): The dataframe representing a group of data.
        augment (bool): Flag indicating whether to apply data augmentation.
        shared_counter (multiprocessing.Value): Shared counter for tracking the number of instances processed.
        lock (multiprocessing.Lock): Lock for synchronizing access to the shared counter.

    Returns:
        TimeSeriesDataset: An instance of the TimeSeriesDataset class.
    """
    with lock:
        shared_counter.value += 1
        print(f"Instances processed: {shared_counter.value}", end='\r')
    return TimeSeriesDataset(df=item, augment=augment)


def create_instances_parallel(group_list, augment, save_path, workers=30):
    """
    Create instances in parallel using multiple workers.

    Args:
        group_list (list): A list of items to create instances from.
        augment (bool): Flag indicating whether to apply augmentation.
        save_path (str): The path to save the instances.
        workers (int, optional): The number of worker processes to use. Defaults to 30.

    Returns:
        None
    """
    manager = Manager()
    shared_counter = manager.Value('i', 0)
    lock = manager.Lock()

    with Pool(workers) as pool:
        instances = pool.starmap(create_instance, [(item, augment, shared_counter, lock) for item in group_list])

    with open(save_path, "wb") as file:
        pickle.dump(instances, file)

    print("\n")
    return None



if __name__ =="__main__":

    ROOT_PATH = "./simulated_tracks/simulation_alpha_K_one_fixed_more_tracks"
    data_paths = [
                    os.path.join(ROOT_PATH, "confinement.parquet"),
                    os.path.join(ROOT_PATH, "dimerization.parquet"),
                    # os.path.join(ROOT_PATH, "single_state.parquet"),
                    os.path.join(ROOT_PATH, "multi_state.parquet"),
                    # os.path.join(ROOT_PATH, "immobile_traps.parquet")
                ]

    # read the groups
    groups = read_groups(data_paths)

    # split the groups
    train_data, val_data, test_data = split_groups(groups)
    print(len(train_data), len(val_data), len(test_data))

    create_instances_parallel(train_data, augment=True, save_path=os.path.join(ROOT_PATH, "train_instances.pkl"))
    create_instances_parallel(val_data, augment=False, save_path=os.path.join(ROOT_PATH, "val_instances.pkl"))
    create_instances_parallel(test_data, augment=False, save_path=os.path.join(ROOT_PATH, "test_instances.pkl"))

    




