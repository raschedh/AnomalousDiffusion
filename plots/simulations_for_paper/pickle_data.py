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


def create_instances_parallel(group_list, augment, save_path, workers=1):
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

    # PATH = "/home/haidiri/Desktop/AnDiChallenge2024/plots/results_for_plotting/jaccard_simulations_for_alpha_fixedL_threeCP_results/jaccard_simulations_for_alpha_fixedL_threeCP.parquet"
    # PATH = "/home/haidiri/Desktop/AnDiChallenge2024/plots/results_for_plotting/fixed_states_over_length_results/fixed_states_over_length.parquet"
    PATH = "/home/haidiri/Desktop/AnDiChallenge2024/plots/results_for_plotting/K_jaccard_single_CP_more_sampling_results/K_jaccard_single_CP_more_sampling.parquet"
    
    directory, filename = os.path.split(PATH)
    name, _ = os.path.splitext(filename)
    new_filename = name + ".pkl"
    SAVE_PATH = os.path.join(directory, new_filename)
    print("Saving to:", SAVE_PATH)

    groups = []
    counter = 0
    df = pd.read_parquet(PATH)

    for _, group in df.groupby("traj_idx"):
        group = group.drop(columns=["traj_idx"]).reset_index(drop=True)
        groups.append(group)
        counter += 1
        print("Groups: ", counter, end="\r")

    create_instances_parallel(groups, augment=False, save_path=SAVE_PATH)

    




