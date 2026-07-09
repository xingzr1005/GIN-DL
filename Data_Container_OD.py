from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from keyframe_extraction import KeyFrameExtractor


class DataInput(object):
    def __init__(self, params: dict):
        self.params = params
        self._max = None
        self._min = None
        self._mean = None
        self._std = None

    def load_data(self):
        od_data = self._load_od_data()
        od_data = self._select_model_data(od_data)
        od_data = self._normalize_od(od_data)

        dataset = {
            "OD": od_data,
            "distance": self._load_csv_matrix("distance_file", normalize=True),
            "adj": self._load_csv_matrix("adj_file"),
        }

        poi_features = self._load_csv_matrix("poi_file")
        dataset["poi"] = poi_features
        dataset["poi_normalized"] = self._normalize_columns(poi_features)
        dataset["total_od"] = self._compute_total_od(od_data)

        print("OD data shape:", dataset["OD"].shape)
        print("Distance shape:", dataset["distance"].shape)
        print("Adjacency shape:", dataset["adj"].shape)
        print("POI shape:", dataset["poi"].shape)
        print("Normalized POI shape:", dataset["poi_normalized"].shape)
        print("Normalized total_od shape:", dataset["total_od"].shape)
        return dataset

    def _load_od_data(self):
        file_path = self._resolve_input_path("od_file")
        od_key = self.params.get("od_key", "OD_matrix")

        with np.load(file_path) as loaded:
            keys = list(loaded.keys())
            if od_key not in keys:
                if len(keys) == 1:
                    od_key = keys[0]
                else:
                    raise KeyError(
                        f"'{od_key}' was not found in {file_path}. "
                        f"Available keys: {keys}"
                    )
            od_data = np.asarray(loaded[od_key], dtype=np.float32)

        if od_data.ndim == 3:
            od_data = od_data[:, :, :, np.newaxis]
        elif od_data.ndim != 4:
            raise ValueError("OD data must have shape (T, N, N) or (T, N, N, C).")

        print(f"Loaded OD data from {file_path} with key '{od_key}': {od_data.shape}")
        return od_data

    def _select_model_data(self, od_data):
        if not self.params.get("use_keyframes", False):
            print("Data source: full OD sequence")
            return od_data

        extractor = KeyFrameExtractor(output_dir=self.params["output_dir"])
        key_frame_indices = extractor.linear_discontinuity_search(
            od_data,
            quantile=self.params.get("keyframe_quantile", 0.4),
        )
        key_frame_data = od_data[key_frame_indices]
        print("Data source: key frames")
        print("Key-frame data shape:", key_frame_data.shape)
        return key_frame_data

    def _resolve_input_path(self, param_name):
        path = Path(self.params[param_name])
        if not path.is_absolute():
            path = Path(self.params["input_dir"]) / path
        if not path.exists():
            raise FileNotFoundError(
                f"Could not find {param_name} at {path}. "
                "Pass --input_dir or the specific file argument to point to your data."
            )
        return path

    def _load_csv_matrix(self, param_name, normalize=False):
        matrix = pd.read_csv(self._resolve_input_path(param_name), header=None).values
        matrix = matrix.astype(np.float32)
        if normalize:
            matrix_min = matrix.min()
            matrix_max = matrix.max()
            matrix = (matrix - matrix_min) / (matrix_max - matrix_min + 1e-8)
        return matrix

    def _normalize_od(self, x):
        norm = self.params.get("norm", "std")
        if norm == "none":
            return x
        if norm == "minmax":
            return self.minmax_normalize(x)
        if norm == "std":
            return self.std_normalize(x)
        raise ValueError(f"Invalid normalization method: {norm}")

    @staticmethod
    def _normalize_columns(x):
        x = np.asarray(x, dtype=np.float32)
        x_min = x.min(axis=0, keepdims=True)
        x_max = x.max(axis=0, keepdims=True)
        return (x - x_min) / (x_max - x_min + 1e-8)

    def _compute_total_od(self, od_data):
        train_cutoff = self._training_cutoff(od_data.shape[0])
        total_od = od_data[:train_cutoff].sum(axis=0)
        return (total_od - total_od.min()) / (total_od.max() - total_od.min() + 1e-8)

    def _training_cutoff(self, data_len):
        split_ratio = self.params["split_ratio"]
        split_sum = sum(split_ratio)
        if split_sum <= 0:
            raise ValueError("split_ratio must contain at least one positive value.")
        cutoff = int(data_len * split_ratio[0] / split_sum)
        return min(data_len, max(1, cutoff))

    def minmax_normalize(self, x: np.array):
        self._max = float(x.max())
        self._min = float(x.min())
        denominator = self._max - self._min
        if denominator < 1e-8:
            denominator = 1.0
        print("min:", self._min, "max:", self._max)
        return (x - self._min) / denominator

    def minmax_denormalize(self, x: np.array):
        if self._max is None or self._min is None:
            raise RuntimeError("Min-max statistics are not available.")
        return (self._max - self._min) * x + self._min

    def std_normalize(self, x: np.array):
        self._mean = float(x.mean())
        self._std = float(x.std())
        if self._std < 1e-8:
            self._std = 1.0
        print("mean:", round(self._mean, 4), "std:", round(self._std, 4))
        return (x - self._mean) / self._std

    def std_denormalize(self, x: np.array):
        if self._mean is None or self._std is None:
            raise RuntimeError("Standardization statistics are not available.")
        return x * self._std + self._mean

    def denormalize(self, x: np.array):
        norm = self.params.get("norm", "std")
        if norm == "none":
            return x
        if norm == "minmax":
            return self.minmax_denormalize(x)
        if norm == "std":
            return self.std_denormalize(x)
        raise ValueError(f"Invalid normalization method: {norm}")

    def normalize_value(self, value):
        norm = self.params.get("norm", "std")
        if norm == "none":
            return value
        if norm == "minmax":
            if self._max is None or self._min is None:
                raise RuntimeError("Min-max statistics are not available.")
            denominator = self._max - self._min
            if denominator < 1e-8:
                denominator = 1.0
            return (value - self._min) / denominator
        if norm == "std":
            if self._mean is None or self._std is None:
                raise RuntimeError("Standardization statistics are not available.")
            return (value - self._mean) / self._std
        raise ValueError(f"Invalid normalization method: {norm}")


class ODDataset(Dataset):
    def __init__(self, inputs: dict, output: torch.Tensor, mode: str, mode_len: dict, obs_len: int):
        self.mode = mode
        self.mode_len = mode_len
        self.inputs, self.output = self.prepare_xy(inputs, output)
        self.obs_len = obs_len

    def __len__(self):
        return self.mode_len[self.mode]

    def __getitem__(self, item: int):
        return self.inputs["x_seq"][item], self.output[item]

    def prepare_xy(self, inputs: dict, output: torch.Tensor):
        if self.mode == "train":
            start_idx = 0
        elif self.mode == "validate":
            start_idx = self.mode_len["train"]
        else:
            start_idx = self.mode_len["train"] + self.mode_len["validate"]

        x = {"x_seq": inputs["x_seq"][start_idx : start_idx + self.mode_len[self.mode]]}
        y = output[start_idx : start_idx + self.mode_len[self.mode]]
        return x, y


class DataGenerator(object):
    def __init__(self, obs_len: int, pred_len, data_split_ratio: tuple):
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.data_split_ratio = data_split_ratio

    def split2len(self, data_len: int):
        mode_len = dict()
        mode_len["validate"] = int(self.data_split_ratio[1] / sum(self.data_split_ratio) * data_len)
        mode_len["test"] = int(self.data_split_ratio[2] / sum(self.data_split_ratio) * data_len)
        mode_len["train"] = data_len - mode_len["validate"] - mode_len["test"]
        return mode_len

    def get_data_loader(self, data: dict, params: dict):
        x_seq, y_seq = self.get_feats(data["OD"])

        device = params["GPU"]
        feat_dict = {"x_seq": torch.from_numpy(np.asarray(x_seq)).float().to(device)}
        y_seq = torch.from_numpy(np.asarray(y_seq)).float().to(device)

        mode_len = self.split2len(data_len=y_seq.shape[0])
        data_loader = dict()
        for mode in ["train", "validate", "test"]:
            dataset = ODDataset(
                inputs=feat_dict,
                output=y_seq,
                mode=mode,
                mode_len=mode_len,
                obs_len=self.obs_len,
            )
            data_loader[mode] = DataLoader(
                dataset=dataset,
                batch_size=params["batch_size"],
                shuffle=False,
            )
        return data_loader

    def get_feats(self, data: np.array):
        x, y = [], []
        for i in range(self.obs_len, data.shape[0] - self.pred_len + 1):
            x.append(data[i - self.obs_len : i])
            y.append(data[i : i + self.pred_len])
        return x, y
