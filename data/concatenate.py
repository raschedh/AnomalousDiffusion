import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
from multiprocessing import Pool, Manager
import numpy as np 
import os

# This file combines the small parquet files from each directory into a single parquet file. Once this is done feel free to delete the original smaller files (or keep them if you have use for it).

def process_files(files, shared_counter, lock):
    
    df_list = []
    
    for file in files:
    
        df = pd.read_parquet(file)
        
        with lock:
            idx = shared_counter.value
            shared_counter.value += 1
            print(f"Files processed: {shared_counter.value}", end='\r')
            
            df['traj_idx'] = idx
            df.reset_index(drop=True, inplace=True)
        
            df_list.append(df)

    return pd.concat(df_list, ignore_index=True)

if __name__ == '__main__':

    ROOT_PATH = "data/simulated_tracks"
    N_WORKERS = 30  # Number of parallel processes to use

    dirs = [os.path.join(ROOT_PATH, "confinement"),
            os.path.join(ROOT_PATH, "dimerization"),
            os.path.join(ROOT_PATH, "single_state"),
            os.path.join(ROOT_PATH, "multi_state"),
            os.path.join(ROOT_PATH, "immobile_traps")]

    for dir_ in dirs:

        print(f"Reading files from {dir_}")

        all_files = list(Path(dir_).rglob('*.parquet'))
        print(len(all_files), "files found")
        
        chunk_size = (len(all_files) + N_WORKERS - 1) // N_WORKERS  # Ensure no files are missed
        file_chunks = [all_files[i:i + chunk_size] for i in range(0, len(all_files), chunk_size)]

        manager = Manager()
        shared_counter = manager.Value('i', 0)
        lock = manager.Lock()

        # Create a pool of worker processes
        with Pool(N_WORKERS) as pool:
            results = pool.starmap(process_files, [(chunk, shared_counter, lock) for chunk in file_chunks])

        combined_df = pd.concat(results, ignore_index=True)
        save_path = os.path.join(ROOT_PATH, os.path.basename(dir_) + ".parquet")
        combined_df.to_parquet(save_path)
        print("Number of tracks:", combined_df['traj_idx'].nunique(), "\n")