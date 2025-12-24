import pickle
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import torch
import sys
import torch.nn as nn


path = '/home/data/zsy/SSSD-main/SSSD-main/src'
sys.path.append(path)


from dataloader import *


def PM25_Dataset(eval_length=24, target_dim=24, batch_size = 16, mode="train", validindex=0):

    eval_length = eval_length
    target_dim = target_dim
    path = "/home/data/lrc/TSI/PriSTI-main/data/pm25/pemsbay_meanstd.pk"
    with open(path, "rb") as f:
        train_mean, train_std = pickle.load(f)


    df = pd.read_hdf("/home/data/lrc/TSI/PriSTI-main/pems-bay.h5")

    sample = df.values
    ob_mask = (df.values != 0.).astype('uint8')
    eval_mask = sample_mask(shape=(52116, 325), p=0.05, p_noise=0.05, min_seq=12, max_seq=12 * 4, rng=rng)
    gt_mask = (1 - (eval_mask | (1 - ob_mask))).astype('uint8')


    c_data = (
                     (df.fillna(0).values - train_mean) / train_std
             ) * ob_mask
    # observed_values, observed_masks, gt_masks = mask_missing_train_rm(sample, 0.5)

    df_gt = pd.read_hdf("/home/data/lrc/TSI/PriSTI-main/pems-bay.h5")


    data_len = sample.shape[0]
    test_len = int(0.2 * data_len)
    val_len = int(0.1 * (data_len - test_len))
    test_start = data_len - test_len
    val_start = test_start - val_len


    c_data = torch.from_numpy(c_data)
    c_mask = torch.from_numpy(ob_mask)
    c_gt_mask = torch.from_numpy(gt_mask)
    # c_data, c_mask, c_gt_mask = torch.from_numpy(c_data), torch.from_numpy(ob_mask), torch.from_numpy(gt_mask)
    path1 = "/home/data/zsy/spin/spin/CSDI-copy22/data/pm25/time_emb.pk"
    with open(path1, "rb") as f:
        time_emb = pickle.load(f)
        time_emb = torch.from_numpy(time_emb)
    train_loader, valid_loader, test_loader = get_dataloader1(False ,batch_size, 0.1, 0.2, 24,
                                                             c_data, c_mask, c_gt_mask,time_emb, "")



    return train_loader, valid_loader, test_loader, train_mean, train_std

def get_dataloader(batch_size, device, val_len=0.1, test_len=0.2, missing_pattern='block',
                   is_interpolate=False, num_workers=4, target_strategy='random', validindex=0):
    train_loader, valid_loader, test_loader, train_mean, train_std = PM25_Dataset(batch_size=batch_size, mode="train", validindex=validindex)


    scaler = torch.from_numpy(train_std).to(device).float()
    mean_scaler = torch.from_numpy(train_mean).to(device).float()

    return train_loader, valid_loader, test_loader, scaler, mean_scaler



def mask_missing_train_rm(data, missing_ratio=0.0):
    observed_values = np.array(data)
    # observed_masks = ~np.isnan(observed_values)
    observed_masks = (observed_values != 0).astype('uint8')

    # seed = np.random.randint(1e9)
    seed = np.random.randint(1e9)
        # Fix seed for random mask generation
    seed = 479346624
    # seed = 9101112

    np.random.seed(seed)
    print(seed)
    # random = np.random.default_rng(seed)
    masks = observed_values.reshape(-1).copy()
    obs_indices = np.where(masks)[0].tolist()
    # miss_indices = random_choice_without_nan(masks,int(len(obs_indices) * missing_ratio))
    miss_indices = np.random.choice(obs_indices, int(len(obs_indices) * missing_ratio), replace=False)
    if len(obs_indices) == 0:
        pass
    else:
        masks[miss_indices] = np.nan
    gt_masks = masks.reshape(observed_masks.shape)
    observed_values = masks.reshape(observed_masks.shape)
    # observed_values = np.nan_to_num(observed_values)
    observed_masks = observed_masks.astype("float32")
    gt_masks = gt_masks.astype("float32")
    # SEED = 9101112
    # random = np.random.default_rng(SEED)
    # ob_mask = (data != 0.).astype('uint8')
    # eval_mask = sample_mask(shape=(34272, 207), p=0., p_noise=0.25, max_seq=12, min_seq=12 * 4, rng=random)
    # gt_masks = (1 - (eval_mask | (1 - ob_mask))).astype('uint8')

    return observed_values, observed_masks, gt_masks


import numpy as np


def random_choice_without_nan(arr, num_samples):
    # Create a mask to identify NaN elements in the array
    nan_mask = np.isnan(arr)

    # Calculate the number of non-NaN elements
    num_non_nan = np.sum(~nan_mask)

    if num_non_nan < num_samples:
        raise ValueError("Number of non-NaN elements is less than the desired number of samples.")

    # Create a probability array with non-NaN elements having equal probability
    prob_non_nan = np.ones_like(arr) / num_non_nan
    # Set the probability of selecting a NaN element to 0
    prob_non_nan[nan_mask] = 0

    # Use random choice with probabilities to select indices
    selected_indices = np.random.choice(len(arr), num_samples, replace=False, p=prob_non_nan)

    return selected_indices


def splitter(self, dataset, val_len=0, test_len=0, window=0):
    idx = np.arange(len(dataset))
    if test_len < 1:
        test_len = int(test_len * len(idx))
    if val_len < 1:
        val_len = int(val_len * (len(idx) - test_len))
    test_start = len(idx) - test_len
    val_start = test_start - val_len
    return [idx[:val_start - window], idx[val_start:test_start - window], idx[test_start:]]




def sample_mask(shape, p=0.0015, p_noise=0.05, max_seq=1, min_seq=1, rng=None):
    if rng is None:
        rand = np.random.random
        randint = np.random.randint
    else:
        rand = rng.random
        randint = rng.integers
    mask = rand(shape) < p
    for col in range(mask.shape[1]):
        idxs = np.flatnonzero(mask[:, col])
        if not len(idxs):
            continue
        fault_len = min_seq
        if max_seq > min_seq:
            fault_len = fault_len + int(randint(max_seq - min_seq))
        idxs_ext = np.concatenate([np.arange(i, i + fault_len) for i in idxs])
        idxs = np.unique(idxs_ext)
        idxs = np.clip(idxs, 0, shape[0] - 1)
        mask[idxs, col] = True
    mask = mask | (rand(mask.shape) < p_noise)
    return mask.astype('uint8')



def get_test_randmask(observed_mask, missing_ratio):
    rand_for_mask = torch.rand_like(observed_mask) * observed_mask
    rand_for_mask = rand_for_mask.reshape(-1)
    sample_ratio = missing_ratio  # missing ratio
    num_observed = observed_mask.sum().item()
    num_masked = round(num_observed * sample_ratio)
    rand_for_mask[rand_for_mask.topk(num_masked).indices] = -1

    cond_mask = (rand_for_mask > 0).reshape(observed_mask.shape).float()
    return cond_mask


