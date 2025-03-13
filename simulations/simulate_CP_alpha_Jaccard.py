import os
os.environ["TQDM_DISABLE"] = "1"
from andi_datasets.datasets_challenge import challenge_phenom_dataset, _get_dic_andi2
import numpy as np 
from multiprocessing import Pool, Lock, Manager
import contextlib
import io
import sys 
import collections
import random 

def generate_sample(alpha_value1, alpha_value2):
    state = 1 # multi
    dic = _get_dic_andi2(state + 1)    
    dic['T'] = 200 
    dic['N'] = 100
    dic['alphas']= np.array([[alpha_value1, 0], [alpha_value2, 0]])
    dic['Ds'] = np.array([[1, 0],[1, 0]])
    return dic

def process_alpha(args):

    delta_alpha_pair, counter, lock = args
    tracks_generated = 0
    delta_alpha, alpha_value1, alpha_value2 = delta_alpha_pair

    while tracks_generated < 100:

        dic = generate_sample(alpha_value1, alpha_value2)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                dfs_traj, _, _ = challenge_phenom_dataset(save_data=False, 
                                                        dics=[dic], 
                                                        return_timestep_labs=True, 
                                                        get_video=False)
        except:
            continue

        if dfs_traj != 0:
            for _, group in dfs_traj[0].groupby('traj_idx'):
                if int(len(group)) == 200:
                    group.to_parquet(os.path.join(SAVE_DIR, f"delta_alpha_{delta_alpha:.2f}_track_{tracks_generated}.parquet"))
                    tracks_generated += 1
                    if tracks_generated >= 100:
                        break
    
    with lock:
        counter.value += 1
        print(f"\rProcessed: {counter.value} out of {TOTAL_POSSIBILITIES}", end="\r")

    # print(f"\nCompleted delta_alpha_{delta_alpha:.2f}, Total tracks: {tracks_generated}")
    return delta_alpha, tracks_generated

if __name__ == "__main__":
    NUM_WORKERS = 30

    alpha_value_list1 = [1,1.95,0.05]
    alpha_value_list2 = [round(0.05 * i, 2) for i in range(1, 40)]
    l = []
    delta_values = set()

    for a1 in alpha_value_list1:
        for a2 in alpha_value_list2:
            delta = round(a1 - a2, 2)
            t = (delta, a1, a2)
            if delta not in delta_values:
                delta_values.add(delta)
                l.append(t)
    
    TOTAL_POSSIBILITIES = len(l)

    manager = Manager()
    counter = manager.Value('i', 0)
    lock = manager.Lock()

    SAVE_DIR = "jaccard_new_single_CP"
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    with Pool(NUM_WORKERS) as p:
        # results = p.map(process_alpha, delta_alpha_items)
        results = p.map(process_alpha, [(item, counter, lock) for item in l])

    # for alpha_pairs, tracks in results:
    #     print(f"Alpha: {alpha_pairs:.2f}, Tracks generated: {tracks}")
    print("\nAll alpha values processed.")