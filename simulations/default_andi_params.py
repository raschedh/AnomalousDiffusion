import os
os.environ["TQDM_DISABLE"] = "1" # Suppresses the tqdm progress bar from challenge_phenom_dataset function for clearer output
from andi_datasets.datasets_challenge import challenge_phenom_dataset, _get_dic_andi2
import numpy as np 
from multiprocessing import Pool, Manager
import random 
import contextlib
import io
import numpy as np


# Worker function for multiprocessing
def process_dic(dic, counters, lock, limit=400000):

    """
    Process a dictionary and save simulated protein tracks (challenge_phenom_dataset) to parquet files. Printing from the function suppressed for cleaner output.
    
    Args:
        dic (dict): The dictionary containing the model information.
        counters (dict): A dictionary of counters for each model type.
        lock (Lock): A lock object for thread synchronization.
        limit (int, optional): The maximum number of tracks to process. Defaults to 100000.
    """
    model_type = dic['model']
    SAVE_PATH = os.path.join(SAVE_DIR, model_type)

    with lock:
        if counters[model_type].value >= limit:
            return
        
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            dfs_traj, _, _ = challenge_phenom_dataset(save_data=False, 
                                                    dics=[dic], 
                                                    return_timestep_labs=True, 
                                                    get_video=False)
    except:
        return

    if dfs_traj != 0:
        for _, group in dfs_traj[0].groupby('traj_idx'):
            with lock:
                if counters[model_type].value >= limit:
                    return
                counters[model_type].value += 1
                counter_values = {key: counters[key].value for key in counters}
                print(f"Counters: {counter_values}", end='\r')
                group.to_parquet(os.path.join(SAVE_PATH, f"track_{counters[model_type].value}.parquet"))



if __name__ == "__main__":

    MODELS = np.arange(5)
    REPEATS = 400000
    SAVE_DIR = "/home/haidiri/Desktop/AnDiChallenge2024/data/simulated_tracks/simulations_default_new_bug_fixed"
    os.makedirs(SAVE_DIR, exist_ok=True)

    for m in MODELS:
        dic = _get_dic_andi2(m+1)
        SAVE_PATH = os.path.join(SAVE_DIR, dic['model'])
        os.makedirs(SAVE_PATH, exist_ok=True)

    dics = []
    for m in MODELS:        
        for _ in range(REPEATS):  
            dic = _get_dic_andi2(m+1)
            dic['T'] = 200 
            dic['N'] = 100
            dics.append(dic)

    print("Done generating dics", len(dics))
    random.shuffle(dics)

    manager = Manager()
    counters = {
        'single_state': manager.Value('i', 0),
        'multi_state': manager.Value('i', 0),
        'immobile_traps': manager.Value('i', 0),
        'dimerization': manager.Value('i', 0),
        'confinement': manager.Value('i', 0)
    }

    lock = manager.Lock()

    with Pool(30) as p:
        p.starmap(process_dic, [(dic, counters, lock) for dic in dics])

  