import argparse
import os

import torch

import Data_Container_OD
import Model_Trainer


def build_parser():
    parser = argparse.ArgumentParser(description="Run OD prediction.")

    parser.add_argument("-GPU", "--GPU", type=str, default="auto")
    parser.add_argument("-in", "--input_dir", type=str, default="../data")
    parser.add_argument("-out", "--output_dir", type=str, default="./output")

    parser.add_argument("--od_file", type=str, default="Chicago_od_matrix_2013.npz")
    parser.add_argument("--od_key", type=str, default="OD_matrix")
    parser.add_argument("--distance_file", type=str, default="distance_matrix_Chi.csv")
    parser.add_argument("--adj_file", type=str, default="adj_matrix_district.csv")
    parser.add_argument("--poi_file", type=str, default="CHI poi.csv")

    parser.add_argument("-model", "--model", type=str, choices=["GINDL"], default="GINDL")
    parser.add_argument("-t", "--time_slice", type=int, default=24)
    parser.add_argument("-obs", "--obs_len", type=int, default=7)
    parser.add_argument("-pred", "--pred_len", type=int, default=1)
    parser.add_argument("-norm", "--norm", type=str, choices=["auto", "none", "minmax", "std"], default="auto")
    parser.add_argument(
        "-split",
        "--split_ratio",
        type=int,
        nargs="+",
        help="Relative data split ratio in train : validate : test.",
        default=None,
    )
    parser.add_argument(
        "--data_source",
        type=str,
        choices=["auto", "full", "keyframes"],
        default="auto",
        help="auto uses keyframes for pretrain and full OD data for train/test.",
    )
    parser.add_argument("--keyframe_quantile", type=float, default=0.4)

    parser.add_argument("-batch", "--batch_size", type=int, default=16)
    parser.add_argument("-hidden", "--hidden_dim", type=int, default=64)
    parser.add_argument("-K", "--cheby_order", type=int, default=2)
    parser.add_argument("-loss", "--loss", type=str, choices=["MSE", "MAE", "Huber", "SSIM"], default="MSE")
    parser.add_argument("--mask_threshold", type=float, default=1.0)
    parser.add_argument("-optim", "--optimizer", type=str, default="Adam")
    parser.add_argument("-lr", "--learn_rate", type=float, default=0.0001)
    parser.add_argument("-dr", "--decay_rate", type=float, default=0)
    parser.add_argument("-epoch", "--num_epochs", type=int, default=1000)
    parser.add_argument("-mode", "--mode", type=str, choices=["pretrain", "train", "test"], default="test")
    return parser


def apply_mode_defaults(params):
    if params["split_ratio"] is None:
        if params["mode"] == "pretrain":
            params["split_ratio"] = [6, 2, 2]
        else:
            params["split_ratio"] = [8, 0, 2]

    if len(params["split_ratio"]) != 3:
        raise ValueError("--split_ratio must contain exactly three integers: train validate test.")

    if params["norm"] == "auto":
        params["norm"] = "std"

    if params["data_source"] == "auto":
        params["use_keyframes"] = params["mode"] == "pretrain"
    else:
        params["use_keyframes"] = params["data_source"] == "keyframes"

    if params["mode"] in ["train", "pretrain"]:
        params["pred_len"] = 1

    return params


def resolve_device(requested_device):
    requested_device = str(requested_device).lower()
    if requested_device == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if requested_device.startswith("cuda"):
        if not torch.cuda.is_available():
            print("CUDA was requested but is not available. Falling back to CPU.")
            return torch.device("cpu")
        if ":" in requested_device:
            device_index = int(requested_device.split(":", 1)[1])
            if device_index >= torch.cuda.device_count():
                print(f"{requested_device} is not available. Falling back to cuda:0.")
                return torch.device("cuda:0")
        return torch.device(requested_device)

    return torch.device(requested_device)


def main():
    params = build_parser().parse_args().__dict__
    params = apply_mode_defaults(params)

    params["GPU"] = resolve_device(params["GPU"])
    print(f"Using device: {params['GPU']}")
    print(
        "Mode defaults: "
        f"mode={params['mode']}, "
        f"data_source={'keyframes' if params['use_keyframes'] else 'full'}, "
        f"split_ratio={params['split_ratio']}, "
        f"norm={params['norm']}"
    )

    os.makedirs(params["output_dir"], exist_ok=True)

    data_input = Data_Container_OD.DataInput(params=params)
    data = data_input.load_data()
    params["N"] = data["OD"].shape[1]

    data_generator = Data_Container_OD.DataGenerator(
        obs_len=params["obs_len"],
        pred_len=params["pred_len"],
        data_split_ratio=params["split_ratio"],
    )
    data_loader = data_generator.get_data_loader(data=data, params=params)

    mode_len = data_generator.split2len(data_len=len(data_loader["train"].dataset) + len(data_loader["validate"].dataset) + len(data_loader["test"].dataset))
    print(f"Train set length: {mode_len['train']}")
    print(f"Validate set length: {mode_len['validate']}")
    print(f"Test set length: {mode_len['test']}")

    trainer = Model_Trainer.ModelTrainer(
        params=params,
        data=data,
        data_container=data_input,
    )

    if params["mode"] == "pretrain":
        trainer.pretrain(data_loader=data_loader, modes=["train", "validate"])
    elif params["mode"] == "train":
        trainer.train(data_loader=data_loader)
    else:
        trainer.test(data_loader=data_loader, modes=["train", "test"])


if __name__ == "__main__":
    main()
