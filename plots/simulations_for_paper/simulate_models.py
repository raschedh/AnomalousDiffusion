import os
os.environ["TQDM_DISABLE"] = "1" # Suppresses the tqdm progress bar from challenge_phenom_dataset function for clearer output
from andi_datasets.datasets_challenge import challenge_phenom_dataset, _get_dic_andi2
import numpy as np 
from multiprocessing import Pool, Manager
import random 
import contextlib
import io
import numpy as np
import sys 
import random
import matplotlib.pyplot as plt 

def sample_value(choices):
    return random.choice(choices)


# def sample_alpha(choices):
#     return random.choice(choices)

# def sample_k(choices):
#     # return np.round(10**np.random.choice(choices) - 1, 2)
#     return random.choice(choices)

# def sample_cps(choices):
#     return random.choice(choices)

# def sample_state(choices):
#     return random.choice(choices)

# def sample_length(choices):
#     return random.choice(choices)

# def generate_random_transition_matrix(n):
#     """
#     Generate a random transition matrix of size n x n.

#     The function generates a random transition matrix where each element represents the probability of transitioning
#     from one state to another. The diagonal elements are set between 0.5 and 0.9, and the off-diagonal elements are
#     filled to ensure that each row sums to 1.

#     Args:
#         n (int): The size of the transition matrix.

#     Returns:
#         transition_matrix (numpy.ndarray): The generated random transition matrix.

#     Example usage:
#     >>> generate_random_transition_matrix(3)
#     array([[0.7, 0.3, 0. ],
#            [0.2, 0.6, 0.2],
#            [0. , 0.4, 0.6]])
#     """
#     transition_matrix = np.zeros((n, n))

#     # Set diagonal values between 0.5 and 0.9
#     for i in range(n):
#         transition_matrix[i, i] = np.random.uniform(0.5, 0.5)
    
#     # Fill the off-diagonal values to ensure each row sums to 1
#     for i in range(n):
#         remaining_probability = 1 - transition_matrix[i, i]
#         off_diagonal_values = np.random.dirichlet(np.ones(n - 1)) * remaining_probability
#         transition_matrix[i, :i] = off_diagonal_values[:i]
#         transition_matrix[i, i + 1:] = off_diagonal_values[i:]
    
#     return transition_matrix


def generate_sample(alpha_range):

    state = 1
    # state = sample_value(state_range)
    dic = _get_dic_andi2(state + 1)
    # length = sample_value(length_range)
    
    dic['T'] = 200 
    dic['N'] = 100 # number of molecules for this setting

    ### SINGLE STATE ####
    # if state == 0:  
    #     dic['alphas'], dic['Ds'] = np.array([sample_value(alpha_range), 0]), np.array([sample_value(k_range), 0])

    ### MULTI STATE ####
    # if state == 1:
        # generate transition matrix 
    D_array, alpha_array = np.zeros((2, 2)), np.zeros((2, 2))

        # D_array = np.array([1,0],[1,0])
        # alpha_array = np.array([sample_value(alpha_range), 0], [sample_value(alpha_range), 0])

    for i in range(2):
        D_array[i], alpha_array[i] = np.array([1, 0]), np.array([sample_value(alpha_range), 0])

        # dic['M'] = generate_random_transition_matrix(no_of_multi_states)
    dic['Ds'], dic['alphas'] = D_array, alpha_array

    # #### IMMOBILE TRAPS ####
    # if state == 2:

    #     dic['Ds'], dic['alphas'] = np.array([sample_value(k_range), 0]), np.array([sample_value(alpha_range), 0])
    #     dic['Pu'] = random.uniform(0, 0.1)
    #     dic['Pb'] = 1
    #     dic['r'] = random.uniform(0.5, 2)
    #     dic['Nt'] = random.randint(100, 300)
    
    # ### DIMERIZATION ####
    # if state == 3:
        
    #     Ds_array, alphas_array = np.zeros((2, 2)), np.zeros((2, 2))
    #     Ds_array[0], alphas_array[0] = np.array([sample_value(k_range), 0]),  np.array([sample_value(alpha_range), 0])
    #     Ds_array[1], alphas_array[1]  = np.array([sample_value(k_range), 0]), np.array([sample_value(alpha_range), 0])
    #     dic['Ds'] = Ds_array
    #     dic['alphas'] = alphas_array
    #     dic['Pu'] = random.uniform(0, 0.1)
    #     dic['Pb'] = 1
    #     dic['r'] = random.uniform(0.5, 5)
        
    # ### CONFINEMENT ####
    # if state == 4:

    #     Ds_array, alphas_array = np.zeros((2, 2)), np.zeros((2, 2))
    #     Ds_array[0], alphas_array[0] = np.array([sample_value(k_range), 0]), np.array([sample_value(alpha_range), 0])
    #     Ds_array[1], alphas_array[1] = np.array([sample_value(k_range), 0]), np.array([sample_value(alpha_range), 0])
    #     dic['Ds'] = Ds_array
    #     dic['alphas'] = alphas_array
    #     dic['trans'] = random.uniform(0, 0.3)
    #     dic['Nc'] = random.randint(30,50)
    #     dic['r'] = random.randint(5, 10)

    return dic


# Worker function for multiprocessing
def process_dic(counters, lock, params, limit=100000):

    """
    Process a dictionary and save simulated protein tracks (challenge_phenom_dataset) to parquet files. Printing from the function suppressed for cleaner output.
    
    Args:
        dic (dict): The dictionary containing the model information.
        counters (dict): A dictionary of counters for each model type.
        lock (Lock): A lock object for thread synchronization.
        limit (int, optional): The maximum number of tracks to process. Defaults to 100000.
    """

    with lock:
        if counters['tracks'].value >= limit:
            sys.exit(0)

    save_dir = params["save_dir"]
    alpha_range = params["alpha_range"]
    # k_range = params["k_range"]
    # state_range = params["state_range"]
    cps_range = params["cps_range"]
    # length_range = params["length_range"]
    # no_of_multi_states = params["no_of_multi_states"]

    dic = generate_sample(alpha_range=alpha_range)

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

            # changes = group["alpha"].diff() != 0 
            # number_of_changes = int(changes.sum())

            if int(len(group)) == 200:
                with lock:
                    counters['tracks'].value += 1
                    current_track = counters['tracks'].value
                    counter_values = {key: counters[key].value for key in counters}
                print(f"Counters: {counter_values}", end='\r')
                group.to_parquet(os.path.join(save_dir, f"track_{current_track}.parquet"))

    return None


def worker_process_dic(counters, lock, params):
    while True:
        process_dic(counters, lock, params)
    
if __name__ == "__main__":

    NUM_WORKERS = 30
    # alpha simulations
    params = {
        "save_dir": "./vary_alpha_fix_rest", 
        "alpha_range": [round(0.01 * i, 2) for i in range(1, 200)], 
        "k_range": [1], 
        "state_range": [1],
        "cps_range": [1],
        "length_range": [200],
        "no_of_multi_states": 2
    }

    manager = Manager()
    lock = manager.Lock()
    counters = {
        'tracks': manager.Value('i', 0)
    }

    os.makedirs(params["save_dir"], exist_ok=True)
    with Pool(NUM_WORKERS) as p:
        p.starmap(worker_process_dic, [(counters, lock, params) for _ in range(NUM_WORKERS)])

    print("\n")

    # ---------------------------------------------------------------------------------------------

    # # vary k
    # dics_k = get_dicts(save_dir="./vary_k_fix_rest", 
    #                     alpha_range=np.array([1]), 
    #                     k_range=np.linspace(0, 6, 601)[1:], 
    #                     state_range=np.array([2]),
    #                     cps_range=np.array([1]),
    #                     length_range=np.array([200]),
    #                     no_of_multi_states=2)

    # # vary length
    # dics_len = get_dicts(save_dir="./vary_len_fix_rest", 
    #                     alpha_range=np.array([1]), 
    #                     k_range=np.array([1]), 
    #                     state_range=np.array([2]),
    #                     cps_range=np.array([1]),
    #                     length_range=np.array([x for x in range(20,201)]),
    #                     no_of_multi_states=2)

    # # vary model
    # dics_model = get_dicts(save_dir="./vary_model_fix_rest", 
    #                     alpha_range=np.linspace(0, 2, 201)[1:-1], 
    #                     k_range=np.array([1]), 
    #                     state_range=np.array([2]),
    #                     cps_range=np.array([1]),
    #                     length_range=np.array([200]),
    #                     no_of_multi_states=2)

    # # vary cps
    # dics_cps = get_dicts(save_dir="./vary_cps_fix_rest", 
    #                     alpha_range=np.array([1]), 
    #                     k_range=np.array([1]), 
    #                     state_range=np.array([2]),
    #                     cps_range=np.array([0,1,2,3,4]),
    #                     length_range=np.array([200]),
    #                     no_of_multi_states=2)


