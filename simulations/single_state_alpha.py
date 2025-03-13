import os
os.environ["TQDM_DISABLE"] = "1"
from andi_datasets.datasets_challenge import challenge_phenom_dataset, _get_dic_andi2
import numpy as np 
from multiprocessing import Pool
import contextlib
import io
import sys 

def generate_sample(alpha_value):
    state = 0
    dic = _get_dic_andi2(state + 1)    
    dic['T'] = 200 
    dic['N'] = 100
    dic['alphas'], dic['Ds'] = np.array([alpha_value, 0]), np.array([1, 0])
    return dic

def process_alpha(alpha_value):
    tracks_generated = 0
    
    while tracks_generated < 1000:
        dic = generate_sample(alpha_value=alpha_value)
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
                    group.to_parquet(os.path.join(SAVE_DIR, f"alpha_{alpha_value:.2f}_track_{tracks_generated}.parquet"))
                    tracks_generated += 1
                    if tracks_generated >= 1000:
                        break

    print(f"\nCompleted alpha = {alpha_value:.2f}, Total tracks: {tracks_generated}")
    return alpha_value, tracks_generated

if __name__ == "__main__":
    NUM_WORKERS = 30
    alpha_value_list = [round(0.01 * i, 2) for i in range(1, 200)]
    SAVE_DIR = "single_state_alpha_values"
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"Processing {len(alpha_value_list)} alpha values using {NUM_WORKERS} workers")
    
    with Pool(NUM_WORKERS) as p:
        results = p.map(process_alpha, alpha_value_list)

    for alpha, tracks in results:
        print(f"Alpha: {alpha:.2f}, Tracks generated: {tracks}")

    print("\nAll alpha values processed.")