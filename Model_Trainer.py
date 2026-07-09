from datetime import datetime
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from tqdm import tqdm

import GINDL
import Metrics


class ModelTrainer(object):
    def __init__(self, params: dict, data: dict, data_container):
        self.params = params
        self.data_container = data_container
        self.output_dir = Path(params["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        device = params["GPU"]
        self.G_dist = torch.tensor(data["distance"], dtype=torch.float32, device=device)
        self.G = torch.tensor(data["adj"], dtype=torch.float32, device=device)
        self.poi_features = torch.tensor(data["poi_normalized"], dtype=torch.float32, device=device)
        self.total_od = torch.tensor(data["total_od"], dtype=torch.float32, device=device)

        self.model = self.get_model().to(device)
        self.criterion = self.get_loss()
        self.optimizer = self.get_optimizer()




    def get_model(self):
        if self.params["model"] == "GINDL":
            return GINDL.GINDL(
                input_dim=1,
                hidden_dim=self.params["hidden_dim"],
                lstm_hidden_dim=self.params["hidden_dim"],
                lstm_num_layers=1,
                num_nodes=self.params["N"],
                poi_dim=self.poi_features.shape[1],
            )
        raise NotImplementedError("Invalid model name.")

    def get_loss(self):
        if self.params["loss"] == "MSE":
            return nn.MSELoss(reduction="mean")
        if self.params["loss"] == "MAE":
            return nn.L1Loss(reduction="mean")
        if self.params["loss"] == "Huber":
            return nn.SmoothL1Loss(reduction="mean")
        if self.params["loss"] == "SSIM":
            return Metrics.SSIMLoss()
        raise NotImplementedError("Invalid loss function.")

    def get_optimizer(self):
        if self.params["optimizer"] == "Adam":
            return optim.Adam(
                params=self.model.parameters(),
                lr=self.params["learn_rate"],
                weight_decay=self.params["decay_rate"],
            )
        raise NotImplementedError("Invalid optimizer name.")

    def pretrain(self, data_loader: dict, modes: list, early_stop_patience=10):
        checkpoint = {"epoch": 0, "state_dict": self.model.state_dict()}
        val_loss = np.inf
        patience_count = early_stop_patience
        distance_weight_log = []
        epoch_times = []

        print("\n", datetime.now().strftime("%Y/%m/%d %H:%M:%S"))
        print(f'     {self.params["model"]} model pretraining begins:')
        for epoch in range(1, 1 + self.params["num_epochs"]):
            epoch_start_time = time.time()
            running_loss = {mode: 0.0 for mode in modes}

            for mode in modes:
                self.model.train(mode == "train")
                sample_count = 0

                for x_seq, y_true in tqdm(
                    data_loader[mode],
                    desc=f"{mode} Epoch {epoch}",
                    leave=False,
                ):
                    with torch.set_grad_enabled(mode == "train"):
                        y_pred, _ = self._forward(x_seq)
                        mask = self._make_mask(y_true)
                        loss = self.criterion(y_pred * mask, y_true * mask)

                        if mode == "train":
                            self.optimizer.zero_grad()
                            loss.backward()
                            self.optimizer.step()

                    running_loss[mode] += loss.item() * y_true.shape[0]
                    sample_count += y_true.shape[0]

                if mode == "train":
                    distance_weight_log.append(self.model.gravity_model.distance_weight.item())

                if mode == "validate" and sample_count > 0:
                    epoch_val_loss = running_loss[mode] / sample_count
                    if epoch_val_loss <= val_loss:
                        print(
                            f"Epoch {epoch}, validation loss drops from "
                            f"{val_loss:.5} to {epoch_val_loss:.5}. "
                            "Updating model checkpoint."
                        )
                        val_loss = epoch_val_loss
                        checkpoint = {"epoch": epoch, "state_dict": self.model.state_dict()}
                        torch.save(checkpoint, self._checkpoint_path("od"))
                        patience_count = early_stop_patience
                    else:
                        print(f"Epoch {epoch}, validation loss does not improve from {val_loss:.5}.")
                        patience_count -= 1
                        if patience_count == 0:
                            epoch_times.append(time.time() - epoch_start_time)
                            self._finish_pretrain(checkpoint, distance_weight_log, epoch_times, epoch)
                            return distance_weight_log

            epoch_times.append(time.time() - epoch_start_time)

        self._finish_pretrain(checkpoint, distance_weight_log, epoch_times, self.params["num_epochs"])
        return distance_weight_log

    def train(self, data_loader: dict):
        pretrain_checkpoint_path = self._checkpoint_path("od")
        if pretrain_checkpoint_path.exists():
            checkpoint = torch.load(pretrain_checkpoint_path, map_location=self.params["GPU"])
            self.model.load_state_dict(checkpoint["state_dict"])
            print("Loaded pre-trained parameters for online learning.")
        else:
            print("No pre-trained parameters found. Starting online learning from scratch.")
            checkpoint = {"step": 0, "state_dict": self.model.state_dict()}

        train_loss_min = np.inf

        print("\n", datetime.now().strftime("%Y/%m/%d %H:%M:%S"))
        print(f'     {self.params["model"]} model online learning begins:')

        for step, (x_seq, y_true) in enumerate(tqdm(data_loader["train"], desc="Online Learning", leave=False)):
            y_pred, _ = self._forward(x_seq)
            mask = self._make_mask(y_true)
            loss = self.criterion(y_pred * mask, y_true * mask)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


            train_loss = loss.item()
            print(f"Step {step}, training loss: {train_loss:.5f}")
            if train_loss < train_loss_min:
                print(
                    f"Step {step}, training loss improved from "
                    f"{train_loss_min:.5f} to {train_loss:.5f}. Updating checkpoint."
                )
                train_loss_min = train_loss
                checkpoint = {"step": step, "state_dict": self.model.state_dict()}
                torch.save(checkpoint, self._checkpoint_path("final"))

        print("\n", datetime.now().strftime("%Y/%m/%d %H:%M:%S"))
        print(f'     {self.params["model"]} model online learning ends.')
        torch.save(checkpoint, self._checkpoint_path("final"))
        return

    def test(self, data_loader: dict, modes: list):
        checkpoint_path = self._checkpoint_path("final")
        if not checkpoint_path.exists():
            checkpoint_path = self._checkpoint_path("od")
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"No checkpoint found at {self._checkpoint_path('final')} or {self._checkpoint_path('od')}."
            )

        trained_checkpoint = torch.load(checkpoint_path, map_location=self.params["GPU"])
        self.model.load_state_dict(trained_checkpoint["state_dict"])
        self.model.eval()

        for mode in modes:
            print("\n", datetime.now().strftime("%Y/%m/%d %H:%M:%S"))
            print(f'     {self.params["model"]} model testing on {mode} data begins:')

            forecast, ground_truth, probability_snapshots = [], [], []
            for x_seq, y_true in tqdm(data_loader[mode], desc=f"{mode} ", leave=False):
                with torch.no_grad():
                    y_pred, probabilities = self._forward(x_seq)
                    probability_snapshots.append(probabilities.cpu().numpy())

                y_true = y_true[:, :1]
                mask = self._make_mask(y_true)
                forecast.append((y_pred * mask).cpu().numpy())
                ground_truth.append((y_true * mask).cpu().numpy())

            if not forecast:
                print(f"No {mode} samples to evaluate.")
                continue

            np.savez(self.output_dir / f"probabilities_{mode}.npz", *probability_snapshots)

            forecast = np.concatenate(forecast, axis=0)
            ground_truth = np.concatenate(ground_truth, axis=0)
            np.savez(
                self.output_dir / f"predictions_{mode}.npz",
                forecast=forecast,
                ground_truth=ground_truth,
            )

            forecast_eval = self.data_container.denormalize(forecast)
            ground_truth_eval = self.data_container.denormalize(ground_truth)
            mse, rmse, mae, mape, ssim_value, cpc = Metrics.evaluate(forecast_eval, ground_truth_eval)

            with open(self.output_dir / f'{self.params["model"]}_prediction_scores.txt', "a") as f:
                f.write(
                    "%s, MSE, RMSE, MAE, MAPE, SSIM, CPC, "
                    "%.10f, %.10f, %.10f, %.10f, %.10f, %.10f\n"
                    % (mode, mse, rmse, mae, mape, ssim_value, cpc)
                )

    def _forward(self, x_seq):
        if self.params["model"] == "GINDL":
            return self.model(
                x_seq=x_seq,
                distance_matrix=self.G_dist,
                poi_features=self.poi_features,
                total_od=self.total_od,
            )
        raise NotImplementedError("Invalid model name.")

    def _make_mask(self, y_true):
        threshold = self.params.get("mask_threshold", 1.0)
        if threshold is None:
            return torch.ones_like(y_true)
        normalized_threshold = self.data_container.normalize_value(threshold)
        return (y_true >= normalized_threshold).float()

    def _checkpoint_path(self, suffix):
        return self.output_dir / f'{self.params["model"]}_{suffix}.pkl'

    def _finish_pretrain(self, checkpoint, distance_weight_log, epoch_times, epoch):
        print("\n", datetime.now().strftime("%Y/%m/%d %H:%M:%S"))
        print(f'     Pretraining stopped at epoch {epoch}. {self.params["model"]} model training ends.')
        torch.save(checkpoint, self._checkpoint_path("od"))
        self._save_distance_weight_log(distance_weight_log, "distance_weight_history.csv", "epoch")
        if epoch_times:
            avg_epoch_time = sum(epoch_times) / len(epoch_times)
            print(f"Average training time per epoch: {avg_epoch_time:.2f} seconds")

    def _save_distance_weight_log(self, distance_weight_log, filename, index_label):
        df = pd.DataFrame(distance_weight_log, columns=["distance_weight"])
        output_path = self.output_dir / filename
        df.to_csv(output_path, index_label=index_label)
        print(f"Distance weight history saved to {output_path}")
