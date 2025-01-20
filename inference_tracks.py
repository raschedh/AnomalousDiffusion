import torch
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm
from utils.pad_batch import pad_batch, LABEL_PADDING_VALUE
from models.RegressionModel import RegressionModel
from models.ClassificationModel import ClassificationModel
import pickle 
from pathlib import Path
import numpy as np
from utils.timeseriesdataset import TimeSeriesDataset
import os 

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32

AlphaModel = RegressionModel().to(DEVICE)
KModel = RegressionModel().to(DEVICE)
StateModel = ClassificationModel().to(DEVICE)

AlphaModel.load_state_dict(torch.load("models/optimal_weights/alpha_weights"))
KModel.load_state_dict(torch.load("models/optimal_weights/k_weights"))
StateModel.load_state_dict(torch.load("models/optimal_weights/state_weights"))

AlphaModel.eval()
KModel.eval()
StateModel.eval()

# def pad_tensor(tensor, target_length):
#     current_length = tensor.shape[1]
#     if current_length < target_length:
#         padding = torch.zeros(tensor.shape[0], target_length - current_length, device=tensor.device)
#         return torch.cat([tensor, padding], dim=1)
#     return tensor

if __name__ == "__main__":

    SAVE_DIR = "../plots/results_for_plotting/test_general_simulations"
    ROOT_DATA_DIR = "../data/simulated_tracks"

    test_instances = []
    test_files = list(Path(ROOT_DATA_DIR).glob("*/test_instances.pkl"))

    for file in test_files:
        with open(file, "rb") as f:
            test_instances += pickle.load(f)

    conc_test = ConcatDataset(test_instances)
    test_loader = DataLoader(conc_test, batch_size=BATCH_SIZE, shuffle=True, collate_fn=pad_batch)
    print("Test data: ", len(test_instances), "DataLoader Sizes:", len(test_loader))

    pred_a, pred_k, pred_state = [], [], []
    gt_a, gt_k, gt_state = [], [], []

    progress_bar = tqdm(total=len(test_loader), desc='Testing', position=0)

    with torch.no_grad():
        for inputs, alpha_labels, k_labels, state_labels in test_loader:
            inputs = inputs.to(DEVICE)
            alpha_labels = alpha_labels.to(DEVICE)
            k_labels = k_labels.to(DEVICE)
            state_labels = state_labels.to(DEVICE)

            pred_alpha_per_track = AlphaModel(inputs).squeeze(-1)
            pred_k_per_track = KModel(inputs).squeeze(-1)
            pred_state_per_track = torch.argmax(StateModel(inputs), dim=-1)

            pred_a.append(pred_alpha_per_track)
            pred_k.append(pred_k_per_track)
            pred_state.append(pred_state_per_track)

            gt_a.append(alpha_labels)
            gt_k.append(k_labels)
            gt_state.append(state_labels)

            progress_bar.update()

    progress_bar.close()

    # Concatenate all tensors
    pred_a = torch.cat(pred_a, dim=0)
    pred_k = torch.cat(pred_k, dim=0)
    pred_state = torch.cat(pred_state, dim=0)
    gt_a = torch.cat(gt_a, dim=0)
    gt_k = torch.cat(gt_k, dim=0)
    gt_state = torch.cat(gt_state, dim=0)

    # Pad tensors to 200 length
    # pred_a = pad_tensor(pred_a, 200)
    # pred_k = pad_tensor(pred_k, 200)
    # pred_state = pad_tensor(pred_state, 200)
    # gt_a = pad_tensor(gt_a, 200)
    # gt_k = pad_tensor(gt_k, 200)
    # gt_state = pad_tensor(gt_state, 200)

    print(pred_a.shape)
    print(pred_k.shape)
    print(pred_state.shape)

    # Move to CPU and convert to numpy only when saving
    np.save(os.path.join(SAVE_DIR, "pred_a.npy"), pred_a.cpu().numpy())
    np.save(os.path.join(SAVE_DIR, "pred_k.npy"), pred_k.cpu().numpy())
    np.save(os.path.join(SAVE_DIR, "pred_state.npy"), pred_state.cpu().numpy())
    np.save(os.path.join(SAVE_DIR, "gt_a.npy"), gt_a.cpu().numpy())
    np.save(os.path.join(SAVE_DIR, "gt_k.npy"), gt_k.cpu().numpy())
    np.save(os.path.join(SAVE_DIR, "gt_state.npy"), gt_state.cpu().numpy())