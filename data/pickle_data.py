import random 
import pickle
from sklearn.model_selection import train_test_split
import pandas as pd 
from multiprocessing import Pool, Manager
# from features import getAllFeatures
import numpy as np
import torch
from torch.utils.data import Dataset

# This file creates a pickle file from the parquet files for faster training. It also uses multiprocessing to speed up the process.
# The getitem method is not implemented here as it is not needed for this purpose.

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
        # not implemented here as not needed for this purpose, see utils for full implementation
        return  
        

def read_groups(data_paths):
    groups = []
    counter = 0

    for path in data_paths:
        df = pd.read_parquet(path)
        # for j in range(4): 
        #     class_counts[f"class{j}"] += len(df[df["state"] == j])

        for _, group in df.groupby("traj_idx"):
            group = group.drop(columns=["traj_idx"]).reset_index(drop=True)
            groups.append(group)
            counter += 1
            print("Groups: ", counter, end="\r")

    # print("\n")
    # print("Class counts:", class_counts)
    # print("\n")

    return groups

def split_groups(groups):
    # splits in ratio 70:15:15
    random.shuffle(groups)
    train_data, temp_data = train_test_split(groups, test_size=0.3, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
    return train_data, val_data, test_data


def create_instance(item, is_train, shared_counter, lock):
    with lock:
        shared_counter.value += 1
        print(f"Instances processed: {shared_counter.value}", end='\r')

    return TimeSeriesDataset(df=item, is_train=is_train)


def create_instances_parallel(group_list, is_train, workers=30, save_path=None):
    manager = Manager()
    shared_counter = manager.Value('i', 0)
    lock = manager.Lock()

    with Pool(workers) as pool:
        instances = pool.starmap(create_instance, [(item, is_train, shared_counter, lock) for item in group_list])

    if save_path:
        with open(save_path, "wb") as file:
            pickle.dump(instances, file)
    print("\n")
    return None



if __name__ =="__main__":
    # read the paths 

    data_paths = [
        "/home/haidiri/Desktop/andi/data_from_andi/from_andi_default_t200/confinement.parquet",
        "/home/haidiri/Desktop/andi/data_from_andi/from_andi_default_t200/immobile_traps.parquet",
        "/home/haidiri/Desktop/andi/data_from_andi/from_andi_default_t200/multi_state.parquet",
        "/home/haidiri/Desktop/andi/data_from_andi/from_andi_default_t200/single_state.parquet",
        "/home/haidiri/Desktop/andi/data_from_andi/from_andi_default_t200/dimerization.parquet"
    ]


    # read the groups
    groups = read_groups(data_paths)
    # groups = read_groups(pilot_path)
    # create_instances_parallel(groups, is_train=False, save_path="pilot_instances.pkl")

    # split the groups
    train_data, val_data, test_data = split_groups(groups)

    print(len(train_data), len(val_data), len(test_data))

    create_instances_parallel(train_data, is_train=True, save_path="/home/haidiri/Desktop/andi/data_from_andi/from_andi_default_t200/default_params_max_200_train_instances.pkl")
    create_instances_parallel(val_data, is_train=False, save_path="/home/haidiri/Desktop/andi/data_from_andi/from_andi_default_t200/default_params_max_200_val_instances.pkl")
    create_instances_parallel(test_data, is_train=False, save_path="/home/haidiri/Desktop/andi/data_from_andi/from_andi_default_t200/default_params_max_200_test_instances.pkl")

    




