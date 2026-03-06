"""A generic training wrapper."""
import logging
from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from dfadetect import cnn_features
from dfadetect.datasets import TransformDataset

LOGGER = logging.getLogger(__name__)


@dataclass
class NNDataSetting:
    use_cnn_features: bool


class Trainer:
    """Lightweight wrapper for training models with gradient descent."""

    def __init__(
        self,
        epochs: int = 20,
        batch_size: int = 32,
        device: str = "cpu",
        optimizer_fn: Callable = torch.optim.Adam,
        optimizer_kwargs: dict = None,
    ) -> None:
        if optimizer_kwargs is None:
            optimizer_kwargs = {"lr": 1e-3}
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self.optimizer_fn = optimizer_fn
        self.optimizer_kwargs = optimizer_kwargs
        self.epoch_test_losses: List[float] = []


class GMMTrainer(Trainer):
    def train(
        self,
        model: torch.nn.Module,
        dataset: TransformDataset,
        logging_prefix: str = "",
        test_len: float = 0.2,
    ) -> torch.nn.Module:
        model = model.to(self.device)
        model.train()

        test_len = int(len(dataset) * test_len)
        train_len = len(dataset) - test_len
        lengths = [train_len, test_len]
        train, test = torch.utils.data.random_split(dataset, lengths)

        train_loader = torch.utils.data.DataLoader(
            train, batch_size=1, shuffle=True
        )
        test_loader = torch.utils.data.DataLoader(
            test, batch_size=1
        )

        optimizer = self.optimizer_fn(
            model.parameters(), **self.optimizer_kwargs
        )
        LOGGER.info(
            f"Starting training of {logging_prefix} for {self.epochs} epochs!"
        )

        for epoch in range(1, self.epochs + 1):
            train_epoch_running = True
            train_iter = iter(train_loader)
            i = 0

            while train_epoch_running:
                batch = []
                for _ in range(self.batch_size):
                    try:
                        audio_file, _ = next(train_iter)
                        audio_file = audio_file.view(
                            audio_file.shape[-2:]
                        ).T
                        batch.append(audio_file)
                    except StopIteration:
                        train_epoch_running = False
                i += 1
                if len(batch) == 0:
                    break

                batch = torch.cat(batch).to(self.device)
                pred = model(batch)
                train_loss = -pred.mean()  # negative log likelihood

                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

                model._build_distributions()
                if i % 100 == 0:
                    LOGGER.info(
                        f"[{epoch:05}] Current batch loss: {train_loss.item(): 4.3f} "
                        f"[{i:05}/{int(len(train)/self.batch_size):05}]"
                    )

            LOGGER.info(
                f"Epoch [{epoch}/{self.epochs}]: "
                f"train/{logging_prefix}__loss: {train_loss.item()}"
            )

            test_loss: List[float] = []
            for audio_file, _ in test_loader:
                audio_file = audio_file.view(
                    audio_file.shape[-2:]
                ).T.to(self.device)
                test_loss.append((-model(audio_file).mean()).item())

            test_loss = float(np.mean(test_loss))
            LOGGER.info(
                f"Epoch [{epoch}/{self.epochs}]: "
                f"test/{logging_prefix}__loss: {test_loss}"
            )
            LOGGER.debug(
                f"[{epoch:05}] Epoch test loss: {test_loss: 3.3f}"
            )
            self.epoch_test_losses.append(test_loss)

        model.eval()
        return model


def forward_and_loss(model, criterion, batch_x, batch_y, **kwargs):
    batch_out = model(batch_x)
    batch_loss = criterion(batch_out, batch_y)
    return batch_out, batch_loss


class GDTrainer(Trainer):
    def train(
        self,
        dataset: torch.utils.data.Dataset,
        model: torch.nn.Module,
        nn_data_setting: NNDataSetting,
        cnn_features_setting: cnn_features.CNNFeaturesSetting,
        test_len: Optional[float] = None,
        test_dataset: Optional[torch.utils.data.Dataset] = None,
        logging_prefix: str = "",
        pos_weight: Optional[torch.FloatTensor] = None,
    ):
        # Split or use provided test set
        if test_dataset is not None:
            train = dataset
            test = test_dataset
        else:
            test_len = int(len(dataset) * test_len)
            train_len = len(dataset) - test_len
            lengths = [train_len, test_len]
            train, test = torch.utils.data.random_split(dataset, lengths)

        train_loader = DataLoader(
            train,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=6,
        )
        test_loader = DataLoader(
            test,
            batch_size=self.batch_size,
            drop_last=True,
            num_workers=6,
        )

        criterion = torch.nn.BCEWithLogitsLoss()
        optim = self.optimizer_fn(
            model.parameters(), **self.optimizer_kwargs
        )

        best_model = None
        best_acc = 0.0

        # --- lists to store history for plotting ---
        train_losses: List[float] = []
        train_accuracies: List[float] = []
        val_losses: List[float] = []
        val_accuracies: List[float] = []

        LOGGER.info(f"Starting training for {self.epochs} epochs!")

        forward_and_loss_fn = forward_and_loss
        use_cuda = self.device != "cpu"

        for epoch in range(self.epochs):
            LOGGER.info(f"Epoch num: {epoch}")

            running_loss = 0.0
            num_correct = 0.0
            num_total = 0.0
            model.train()

            # --------- TRAIN LOOP ---------
            for i, (batch_x, _, batch_y) in enumerate(train_loader):
                batch_size = batch_x.size(0)
                num_total += batch_size

                batch_x = batch_x.to(self.device)

                if nn_data_setting.use_cnn_features:
                    batch_x = cnn_features.prepare_feature_vector(
                        batch_x,
                        cnn_features_setting=cnn_features_setting,
                    )

                batch_y = (
                    batch_y.unsqueeze(1)
                    .type(torch.float32)
                    .to(self.device)
                )

                batch_out, batch_loss = forward_and_loss_fn(
                    model, criterion, batch_x, batch_y, use_cuda=use_cuda
                )
                batch_pred = (torch.sigmoid(batch_out) + 0.5).int()
                num_correct += (
                    batch_pred == batch_y.int()
                ).sum(dim=0).item()

                running_loss += batch_loss.item() * batch_size

                if i % 100 == 0:
                    LOGGER.info(
                        f"[{epoch:04d}][{i:05d}]: "
                        f"{running_loss / max(num_total,1)} "
                        f"{num_correct / max(num_total,1) * 100}"
                    )

                optim.zero_grad()
                batch_loss.backward()
                optim.step()

            running_loss /= max(num_total, 1)
            train_accuracy = num_correct / max(num_total, 1) * 100.0

            LOGGER.info(
                f"Epoch [{epoch+1}/{self.epochs}]: "
                f"train/{logging_prefix}__loss: {running_loss}, "
                f"train/{logging_prefix}__accuracy: {train_accuracy}"
            )

            # store train history
            train_losses.append(running_loss)
            train_accuracies.append(train_accuracy)

            # --------- VALIDATION LOOP ---------
            test_running_loss = 0.0
            num_correct = 0.0
            num_total = 0.0
            model.eval()

            with torch.no_grad():
                for batch_x, _, batch_y in test_loader:
                    batch_size = batch_x.size(0)
                    num_total += batch_size

                    batch_x = batch_x.to(self.device)
                    if nn_data_setting.use_cnn_features:
                        batch_x = cnn_features.prepare_feature_vector(
                            batch_x,
                            cnn_features_setting=cnn_features_setting,
                        )

                    batch_y = (
                        batch_y.unsqueeze(1)
                        .type(torch.float32)
                        .to(self.device)
                    )

                    batch_out = model(batch_x)
                    batch_loss = criterion(batch_out, batch_y)
                    test_running_loss += batch_loss.item() * batch_size

                    batch_pred = (torch.sigmoid(batch_out) + 0.5).int()
                    num_correct += (
                        batch_pred == batch_y.int()
                    ).sum(dim=0).item()

            test_running_loss /= max(num_total, 1)
            test_acc = 100.0 * num_correct / max(num_total, 1)

            LOGGER.info(
                f"Epoch [{epoch+1}/{self.epochs}]: "
                f"test/{logging_prefix}__loss: {test_running_loss}, "
                f"test/{logging_prefix}__accuracy: {test_acc}"
            )

            # store val history
            val_losses.append(test_running_loss)
            val_accuracies.append(test_acc)

            # keep best model
            if best_model is None or test_acc > best_acc:
                best_acc = test_acc
                best_model = deepcopy(model.state_dict())

            LOGGER.info(
                f"[{epoch:04d}]: {running_loss} - "
                f"train acc: {train_accuracy} - test_acc: {test_acc}"
            )

        # restore best weights
        model.load_state_dict(best_model)

        # --------- PLOTTING TRAIN / VAL CURVES ---------
        epochs_range = range(1, self.epochs + 1)

        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, train_losses, label="Train Loss")
        plt.plot(epochs_range, val_losses, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss curves")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, train_accuracies, label="Train Acc")
        plt.plot(epochs_range, val_accuracies, label="Val Acc")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.title("Accuracy curves")
        plt.legend()

        plt.tight_layout()
        plt.show()

        return model
