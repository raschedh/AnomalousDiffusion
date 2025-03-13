import os
os.environ["TQDM_DISABLE"] = "1" # Suppresses the tqdm progress bar from challenge_phenom_dataset function for clearer output
from andi_datasets.datasets_challenge import challenge_phenom_dataset, _get_dic_andi2
import numpy as np 
from multiprocessing import Pool, Manager
import random 
import contextlib
import io
import numpy as np


def sample_alpha_K(range_alpha, range_K):
    """
    Sample alpha and K values.

    Args:
        range_alpha (array-like): The range of alpha values to choose from.
        range_K (array-like): The range of K values to choose from.

    Returns:
        tuple: A tuple containing the sampled alpha and K values.
    """
    log_K_plus1, alpha = np.random.choice(range_K), np.random.choice(range_alpha)
    K = np.round(10**log_K_plus1 - 1, 2)
    return alpha, K


def generate_random_transition_matrix(n):
    """
    Generate a random transition matrix of size n x n.

    The function generates a random transition matrix where each element represents the probability of transitioning
    from one state to another. The diagonal elements are set between 0.5 and 0.9, and the off-diagonal elements are
    filled to ensure that each row sums to 1.

    Args:
        n (int): The size of the transition matrix.

    Returns:
        transition_matrix (numpy.ndarray): The generated random transition matrix.

    Example usage:
    >>> generate_random_transition_matrix(3)
    array([[0.7, 0.3, 0. ],
           [0.2, 0.6, 0.2],
           [0. , 0.4, 0.6]])
    """
    transition_matrix = np.zeros((n, n))

    # Set diagonal values between 0.5 and 0.9
    for i in range(n):
        transition_matrix[i, i] = np.random.uniform(0.5, 0.9)
    
    # Fill the off-diagonal values to ensure each row sums to 1
    for i in range(n):
        remaining_probability = 1 - transition_matrix[i, i]
        off_diagonal_values = np.random.dirichlet(np.ones(n - 1)) * remaining_probability
        transition_matrix[i, :i] = off_diagonal_values[:i]
        transition_matrix[i, i + 1:] = off_diagonal_values[i:]
    
    return transition_matrix


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

    SAVE_DIR = "/home/haidiri/Desktop/AnDiChallenge2024/data/simulated_tracks/simulation_alpha_K_one_fixed_more_tracks"
    os.makedirs(SAVE_DIR, exist_ok=True)

    MODELS = [1,3,4]
    REPEATS = 400000
    NUM_WORKERS = 30
    STD = 0.05

    range_k = np.linspace(0, 6, 601)[1:] #K range from 10e-12 to 10e6 which we transform to log10(K + 1) becoming [0,6]
    range_alpha = np.linspace(0, 2, 201)[1:-1] #alpha ranges from [0,2]
    
    print("Start K:", range_k[0], "End K:", range_k[-1])
    print("Start Alpha:",range_alpha[0], "End Alpha:", range_alpha[-1])

    dics = []

    for m in MODELS:
        for _ in range(REPEATS):  

            dic = _get_dic_andi2(m+1)
            dic['T'] = 200 
            dic['N'] = 100

            ### MULTI STATE ####
            if m == 1:

                base_states = 2 
                if np.random.rand() < 0.5:
                    base_states += 1

                # generate transition matrix 
                D_array = np.zeros((base_states, 2))
                alpha_array = np.zeros((base_states, 2))

                # fix either alpha or K for all states 
                if np.random.rand() < 0.5:
                    # fix K
                    _, K_fix = sample_alpha_K(range_alpha, range_k)

                    for i in range(base_states):
                        alpha, _ = sample_alpha_K(range_alpha, range_k)
                        D_array[i] = np.array([K_fix, 0])
                        alpha_array[i] = np.array([alpha, STD*alpha])
                
                else:
                    # fix alpha 
                    alpha_fix, _ = sample_alpha_K(range_alpha, range_k)

                    for i in range(base_states):
                        _, K = sample_alpha_K(range_alpha, range_k)
                        D_array[i] = np.array([K, STD*K])
                        alpha_array[i] = np.array([alpha_fix, 0])

                dic['M'] = generate_random_transition_matrix(base_states)
                dic['Ds'] = D_array
                dic['alphas'] = alpha_array

                SAVE_PATH = os.path.join(SAVE_DIR, dic['model'])
                os.makedirs(SAVE_PATH, exist_ok=True)
            
            ### DIMERIZATION ####
            if m == 3:
                
                Ds_array = np.zeros((2, 2))
                alphas_array = np.zeros((2, 2))

                if np.random.rand() < 0.5:
                    # fix K for  dimers
                    alpha_1, K_fix = sample_alpha_K(range_alpha, range_k)
                    alpha_2, _ = sample_alpha_K(range_alpha, range_k)

                    Ds_array = np.array([[K_fix, 0], [K_fix, 0]])
                    alphas_array = np.array([[alpha_1, STD*alpha_1], [alpha_2, STD*alpha_2]])

                else:
                    # fix alpha for dimers
                    alpha_fix, K1 = sample_alpha_K(range_alpha, range_k)
                    _, K2 = sample_alpha_K(range_alpha, range_k)

                    Ds_array = np.array([[K1, STD*K1], [K2, STD*K2]])
                    alphas_array = np.array([[alpha_fix, 0], [alpha_fix, 0]])

                dic['Ds'] = Ds_array
                dic['alphas'] = alphas_array
                dic['Pu'] = random.uniform(0, 0.1)
                dic['Pb'] = 1
                dic['r'] = random.uniform(0.5, 5)

                SAVE_PATH = os.path.join(SAVE_DIR, dic['model'])
                os.makedirs(SAVE_PATH, exist_ok=True)
                
            ### CONFINEMENT ####
            if m == 4:

                Ds_array = np.zeros((2, 2))
                alphas_array = np.zeros((2, 2))

                if np.random.rand() < 0.5:
                    # fix K 
                    alpha_1, K_fix = sample_alpha_K(range_alpha, range_k)
                    alpha_2, _ = sample_alpha_K(range_alpha, range_k)

                    Ds_array = np.array([[K_fix, 0], [K_fix, 0]])
                    alphas_array = np.array([[alpha_1, STD*alpha_1], [alpha_2, STD*alpha_2]])

                else:
                    # fix alpha
                    alpha_fix, K1 = sample_alpha_K(range_alpha, range_k)
                    _, K2 = sample_alpha_K(range_alpha, range_k)

                    Ds_array = np.array([[K1, STD*K1], [K2, STD*K2]])
                    alphas_array = np.array([[alpha_fix, 0], [alpha_fix, 0]])

                dic['Ds'] = Ds_array
                dic['alphas'] = alphas_array
                dic['trans'] = random.uniform(0, 0.3)
                dic['Nc'] = random.randint(30,50)
                dic['r'] = random.randint(5, 10)

                SAVE_PATH = os.path.join(SAVE_DIR, dic['model'])
                os.makedirs(SAVE_PATH, exist_ok=True)

            dics.append(dic)

    print("Done generating dics", len(dics))
    random.shuffle(dics)
    # save it to correct folders 
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

  