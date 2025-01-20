import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
from multiprocessing import Pool, Manager
import numpy as np 
import os

def process_files(files, shared_counter, lock):
    """
    Process a list of files and concatenate their dataframes into a single dataframe.

    Args:
        files (list): A list of file paths.
        shared_counter (Value): A shared counter object for tracking the number of files processed.
        lock (Lock): A lock object for thread synchronization.

    Returns:
        pandas.DataFrame: A concatenated dataframe containing the data from all the input files.
    """
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

    ROOT_PATH = "K_jaccard_single_CP_more_sampling"
    SAVE_PATH = "../results_for_plotting/"+ROOT_PATH+"_results/"+ROOT_PATH+".parquet"
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    N_WORKERS = 10  # Number of parallel processes to use
    print(f"Reading files from {ROOT_PATH}")

    all_files = list(Path(ROOT_PATH).rglob('*.parquet'))
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
    combined_df.to_parquet(SAVE_PATH)
    print("Number of tracks:", combined_df['traj_idx'].nunique(), "\n")

