import os
os.environ["TQDM_DISABLE"] = "1"
from andi_datasets.datasets_challenge import challenge_phenom_dataset, _get_dic_andi2
import numpy as np 
from multiprocessing import Pool
import contextlib
import io

def generate_sample(log1p_k_value):

    k_value = 10**log1p_k_value - 1
    state = 0
    dic = _get_dic_andi2(state + 1)    
    dic['T'] = 200 
    dic['N'] = 100
    dic['alphas'], dic['Ds'] = np.array([1, 0]), np.array([k_value, 0])
    return dic

def process_k(log1p_k_value):
    tracks_generated = 0
    attempts = 0
    points_generated = 0    
    # while tracks_generated < 100:
    while points_generated < 20000:
        dic = generate_sample(log1p_k_value=log1p_k_value)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                dfs_traj, _, _ = challenge_phenom_dataset(save_data=False, 
                                                        dics=[dic], 
                                                        return_timestep_labs=True, 
                                                        get_video=False)
        except:
            attempts += 1
            if attempts == 3:
                break
            
            continue

        if dfs_traj != 0:
            attempts = 0
            for _, group in dfs_traj[0].groupby('traj_idx'):
                # if int(len(group)) == 200:
                    group.to_parquet(os.path.join(SAVE_DIR, f"k_{log1p_k_value:.2f}_track_{tracks_generated}.parquet"))
                    tracks_generated += 1
                    points_generated += len(group)
                    # if tracks_generated >= 100:
                    #     break
                    if points_generated >= 20000:
                        break
    if attempts == 3:
        print(f"\nSkipping k = {log1p_k_value:.2f}")
        return log1p_k_value, points_generated
    
    print(f"\nCompleted k = {log1p_k_value:.2f}, Total tracks: {points_generated}")
    return log1p_k_value, points_generated

if __name__ == "__main__":
    NUM_WORKERS = 30
    # log1p_k_value_list = [round(0.01 * i, 2) for i in range(0, 601, 5)][1:]
    log1p_k_value_list = [round(0.01 * i, 2) for i in range(0, 601, 1)][1:]
    SAVE_DIR = "single_state_k_values_all_point_based"
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"Processing {len(log1p_k_value_list)} k values using {NUM_WORKERS} workers")
    
    with Pool(NUM_WORKERS) as p:
        results = p.map(process_k, log1p_k_value_list)

    for k, tracks in results:
        print(f"k: {k:.2f}, Tracks generated: {tracks}")

    print("\nAll k values processed.")