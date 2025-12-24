import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, Dataset
import os
import torch.distributed as dist
import torch.multiprocessing as mp


def split_ratio(data, val_ratio, test_ratio, window=24):#区分出测试集和训练集
    data_len = data.shape[0]
    if test_ratio < 1:
        test_len = int(test_ratio * data_len)
    if val_ratio < 1:
        val_len = int(val_ratio * (data_len - test_len))
    test_start = data_len - test_len
    val_start = test_start - val_len

    return data[:val_start - window], data[val_start:test_start - window], data[test_start:]


def read_data(data_path):
    flow_data = np.load(data_path, allow_pickle=True).astype(np.float32)
    flow_data = flow_data[:5000]
    return flow_data


# generate data, groundtruth and mask
def get_X_Y(impdata, data, mask, time, wea, step, imput = True):
    # impdata = full_data , data = observed_mask, mask = gt_mask
    length = len(impdata)

    end_index = length - step - 12
    gt_index = int(step / 2)    # 6
    X, Y, M, W, T = [], [], [], [], []
    index = 0

    i = 0
    while index <= end_index:
        if imput is True:
            mask[index+12:index + step] = 0

            X.append(impdata[index:index + step])   # 13
            # Y.append(data[index + gt_index])    # 1 (the sixth of 13 number)
            # M.append(mask[index + gt_index])    # 1 (the sixth of 13 number)
            Y.append(data[index:index + step])
            M.append(mask[index:index + step])
            T.append(time[index:index + step])
            i = i + 1
        else:
            X.append(impdata[index:index + step])  # 13
            # Y.append(data[index + gt_index])    # 1 (the sixth of 13 number)
            # M.append(mask[index + gt_index])    # 1 (the sixth of 13 number)
            Y.append(data[index:index + step])
            M.append(mask[index:index + step])
            T.append(time[index:index + step])
            i = i + 1
        if wea is not None:
            W.append(wea[index:index + step])       # 13
        index += 24

    # 判断 weather feature 是否存在
    if wea is not None:
        W = np.array(W)
    else:
        W = None


    k = int(i/2)
    # T = torch.pad(T, (0, 24 - T.shape[0]), 'constant', constant_values=(0, -1))
    T = T[0:50]
    T1 = T + T + T + T
    T1 = T1 + T1 + T1
    T1 = T1 + T1 +T1
    T1 = T1 + T1 + T1

    X, Y, M, T = torch.stack(X,dim=0), torch.stack(Y,dim=0), torch.stack(M,dim=0), torch.stack(T1,dim=0)
    return X, Y, M, T



class MyDataset(Dataset):
    def __init__(self,observed_values,observed_masks,gt_masks, T, eval_length):
        self.observed_values = observed_values.squeeze(-1)
        self.observed_masks = observed_masks.squeeze(-1)
        self.gt_masks = gt_masks.squeeze(-1)
        self.t = eval_length
        self.l= self.observed_values.size(0)
        self.te = T.squeeze(-1)


    def __getitem__(self, index):

        s = {
            "observed_data": self.observed_values[index],
            "observed_mask": self.observed_masks[index],
            "gt_mask": self.gt_masks[index],
            "timepoints": np.arange(self.t),
            'time_emb': self.te[index % 1000]
        }

        return s

    def __len__(self):
        return self.l


def data_loader(X, Y, M, T, step,batch_size, shuffle=True, drop_last=True):

    data = MyDataset(X, Y, M, T, step)
    # sampler = torch.utils.data.distributed.DistributedSampler(data)
    #
    # dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size,shuffle=shuffle, sampler=sampler)
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    return dataloader


def get_dataloader1(imput, batch_size, val_ratio, test_ratio, step, impdata, data, mask, time_emb, weather_path):

    impdata_train, impdata_val, impdata_test = split_ratio(impdata, val_ratio, test_ratio)
    data_train, data_val, data_test = split_ratio(data, val_ratio, test_ratio)
    mask_train, mask_val, mask_test = split_ratio(mask, val_ratio, test_ratio)
    time_train, time_val, time_test = split_ratio(time_emb, val_ratio, test_ratio)

    # 判断feature是否存在time
    if os.path.exists(weather_path):
        weather = read_data(weather_path)
        wea_train, wea_val, wea_test = split_ratio(weather, val_ratio, test_ratio)
    else:
        wea_train, wea_val, wea_test = None, None, None

    x_train, y_train, m_train, time_train = get_X_Y(impdata_train, data_train, mask_train, time_train, wea_train, step, imput=imput)
    x_val, y_val, m_val, time_val = get_X_Y(impdata_val, data_val, mask_val,time_val, wea_val, step, imput=imput)
    x_test, y_test, m_test, time_test = get_X_Y(impdata_test, data_test, mask_test, time_test,wea_test, step, imput=imput)

    train_dataloader = data_loader(x_train, y_train, m_train, time_train, step,batch_size=batch_size, shuffle=True, drop_last=True)
    val_dataloader = data_loader(x_val, y_val, m_val, time_val, step,batch_size=batch_size, shuffle=False, drop_last=True)
    test_dataloader = data_loader(x_test, y_test, m_test, time_test, step,batch_size=batch_size, shuffle=False, drop_last=True)

    return train_dataloader, val_dataloader, test_dataloader


