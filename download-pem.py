import tarfile
import zipfile
import sys
import os

import requests
import pandas as pd
import pickle
import torch
import matplotlib.pyplot as plt
import numpy as np



def download():

    os.makedirs("data/", exist_ok=True)

    def create_normalizer_pm25():

        df = pd.read_hdf("/home/duanlei/Score-CDM/pems-bay.h5")

        mean = df.describe().loc["mean"].values
        std = df.describe().loc["std"].values

        path = "./data/pm25/pemsbay_meanstd.pk"
        with open(path, "wb") as f:
            pickle.dump([mean, std], f)

    create_normalizer_pm25()

if __name__ == "__main__":
    download()