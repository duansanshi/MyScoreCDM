import argparse
import logging

import datetime
import json

import os
import sys

import numpy as np
import torch
import yaml
from dataset_pems08 import get_dataloader
from main_model_aqi36 import Score_Pems08
from utils import train, evaluate
from download1 import download


def main(args):

    path = "config/" + args.config
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    config["model"]["is_unconditional"] = args.unconditional
    config["model"]["target_strategy"] = args.targetstrategy
    config["diffusion"]["adj_file"] = 'pems-08'


    print(json.dumps(config, indent=4))

    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    foldername = (
        "./save/pems08_" + args.missing_pattern + '_' + current_time + "/"
    )

    print('model folder:', foldername)
    os.makedirs(foldername, exist_ok=True)
    with open(foldername + "config.json", "w") as f:
        json.dump(config, f, indent=4)

    # 载入数据
    train_loader, valid_loader, test_loader, scaler, mean_scaler = get_dataloader(
            config["train"]["batch_size"],
            device=args.device,
            val_len=0.1,
            is_interpolate=True,
            num_workers=args.num_workers,
            target_strategy=args.targetstrategy,
        )

    model = Score_Pems08(config, args.device).to(args.device)

    # if args.modelfolder == "":
    #     train(
    #         model,
    #         config["train"],
    #         train_loader,
    #         valid_loader=valid_loader,
    #         foldername=foldername,
    #     )
    # else:
    #     model.load_state_dict(torch.load("./save/" + args.modelfolder + "/model.pth"))

    model.load_state_dict(torch.load("/home/duanlei/Score-CDM/save/pems08_point_20251210_173143/model.pth")) #11.5 use guide false
    #model.load_state_dict(torch.load("/home/duanlei/Score-CDM/save/pems08_point_20251211_014150/model.pth")) #1e-4 lr
    logging.basicConfig(filename=foldername + '/test_model.log', level=logging.DEBUG)
    logging.info("model_name={}".format(args.modelfolder))
    evaluate(
        model,
        test_loader,
        nsample=args.nsample,
        scaler=scaler,
        mean_scaler=mean_scaler,
        foldername=foldername,
    )


if __name__ == '__main__':
    download()
    parser = argparse.ArgumentParser(description="Score-CDM")
    parser.add_argument("--config", type=str, default="traffic1.yaml")
    parser.add_argument('--device', default='cuda:1', help='Device for Attack')
    parser.add_argument('--num_workers', type=int, default=16, help='Device for Attack')
    parser.add_argument("--modelfolder", type=str, default="")  #pemsbay_point_20231114_125929  pemsbay_point_20231211_162248  pemsbay_point_20231213_172916 pemsbay_point_20231213_172916
    parser.add_argument(
        "--targetstrategy", type=str, default="block", choices=["mix", "random", "block"]
    )
    parser.add_argument("--nsample", type=int, default=10)
    parser.add_argument("--seed", type=int, default=479346624)
    parser.add_argument("--unconditional", action="store_true")
    parser.add_argument("--missing_pattern", type=str, default="point")     # block|point

    args = parser.parse_args()
    print(args)

    main(args)