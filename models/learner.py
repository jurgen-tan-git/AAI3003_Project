"""Learner module for training PyTorch models with callbacks in the scikit-learn style.
"""

import abc
import enum
import os
from typing import Callable

import numpy as np
import torch
from fastprogress import master_bar, progress_bar
from fastprogress.fastprogress import ConsoleMasterBar
from sklearn.base import BaseEstimator
from sklearn.metrics import f1_score
from torch import autocast, nn, optim
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader


class CallbackRequirement(enum.Flag):
    """Callback requirements for the Learner class.

    Attributes:
        NONE: No requirements.
        TRAIN_OUTPUT: Requires the training output.
        VALID_OUTPUT: Requires the validation output.
        TRAIN_LABEL: Requires the training label.
        VALID_LABEL: Requires the validation label.
        TRAIN_LOSS: Requires the training loss.
        VALID_LOSS: Requires the validation loss.
        PERSISTENT_DATA: Requires persistent data.
    """

    NONE = 0
    TRAIN_OUTPUT = enum.auto()
    VALID_OUTPUT = enum.auto()
    TRAIN_LABEL = enum.auto()
    VALID_LABEL = enum.auto()
    TRAIN_LOSS = enum.auto()
    VALID_LOSS = enum.auto()
    PERSISTENT_DATA = enum.auto()


class Callback(metaclass=abc.ABCMeta):
    """Abstract base class for callbacks."""

    @abc.abstractmethod
    def __init__(self, *args, **kwargs):
        self.requirements: CallbackRequirement = CallbackRequirement.NONE

    @abc.abstractmethod
    def __call__(self, state_dict: dict, mbar: ConsoleMasterBar, **outputs):
        """Run the callback.

        Args:
            state_dict (dict): The state dictionary containing the current
                state of the model.
            mbar (ConsoleMasterBar): The console progress bar.
        """
        pass


class MetricCallback(Callback):
    """
    A callback that computes a metric on the model's validation output and label.

    Attributes:
        requirements (CallbackRequirement): The requirements for this callback.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.requirements = (
            CallbackRequirement.VALID_OUTPUT | CallbackRequirement.VALID_LABEL
        )

    def __call__(self, state_dict: dict, **outputs):
        """
        Update the state_dict with the metric values for the current batch.

        Args:
            state_dict (dict): A dictionary containing the current state of the training loop.
            **outputs: A dictionary containing the outputs of the model for the current batch.

        Returns:
            None
        """
        if self.__repr__() not in state_dict["metrics"]:
            state_dict["metrics"][self.__repr__()] = []
        values = self._metric(
            state_dict, outputs["valid_output"], outputs["valid_label"]
        )

        if isinstance(values, tuple):
            # check if values is a namedtuple
            if hasattr(values, "_fields") and hasattr(values, "_asdict"):
                keys = values._fields
                values = values._asdict()

                for key in keys:
                    if key not in state_dict["metrics"]:
                        state_dict["metrics"][key] = []
                    state_dict["metrics"][key].append(values[key])
        else:
            state_dict["metrics"][self.__repr__()].append(values)

    def __repr__(self) -> str:
        """
        Returns a string representation of the class name with "Callback"
        removed and converted to lowercase.

        Returns:
            str: The string representation of the class name.
        """
        return self.__class__.__name__.replace("Callback", "").lower()

    @abc.abstractmethod
    def _metric(self, state_dict: dict, valid_output, valid_label):
        """
        Calculates the metric for the validation set.

        Args:
            state_dict (dict): The state dictionary.
            valid_output: The output of the validation set.
            valid_label: The label of the validation set.

        Returns:
            The calculated metric for the validation set.
        """


class F1Callback(MetricCallback):
    """A callback that computes the F1 score of a model on a validation set.

    Args:
        multilabel (bool, optional): Whether the targets are multi-label. Defaults to True.
        threshold (float, optional): Positive prediction threshold. Defaults to 0.5.
    """

    def __init__(self, *args, multilabel=True, threshold=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.multilabel = multilabel
        self.threshold = threshold

    def _metric(self, state_dict: dict, valid_output, valid_label):
        if self.multilabel:
            valid_preds = (valid_output > self.threshold).astype(int)
            f1 = f1_score(valid_label, valid_preds, average="weighted")
        else:
            valid_preds = np.argmax(valid_output, axis=1)
            f1 = f1_score(valid_label, valid_output, average="weighted")
        return f1


class AccuracyCallback(MetricCallback):
    """
    A callback that computes the accuracy of a model on a validation set.

    Args:
        *args: Positional arguments passed to the parent class.
        multilabel (bool): Whether the problem is multilabel classification.
            Defaults to False.
        **kwargs: Keyword arguments passed to the parent class.
    """

    def __init__(
        self, *args, threshold: float = 0.85, multilabel: bool = False, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.threshold = threshold
        self.multilabel = multilabel

    def _metric(self, state_dict: dict, valid_output, valid_label):
        """
        Computes the accuracy of the model on the validation set.

        Args:
            state_dict (dict): The state dictionary of the model.
            valid_output (numpy.ndarray): The output of the model on the validation set.
            valid_label (numpy.ndarray): The labels of the validation set.

        Returns:
            float: The accuracy of the model on the validation set.
        """
        if self.multilabel:
            valid_preds = (valid_output >= self.threshold).astype(int)
            valid_acc = np.sum(valid_preds == valid_label) / (
                valid_preds.shape[0] * valid_preds.shape[1]
            )
        else:
            valid_acc = np.sum(np.argmax(valid_output, axis=1) == valid_label) / len(
                valid_label
            )
        return valid_acc


class Learner(BaseEstimator):
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        device: torch.device,
        optimizer: optim.Optimizer = None,
        scheduler: optim.lr_scheduler._LRScheduler = None,
        scaler: GradScaler = None,
        metrics: list[MetricCallback] = None,
        cbs: list[Callback] = None,
    ) -> None:
        """Learner class for training PyTorch models with callbacks.

        Args:
            model (nn.Module): The PyTorch model to train.
            criterion (nn.Module): The loss function to use.
            device (torch.device): The device to use for training.
            optimizer (optim.Optimizer, optional): Optimizer for training. Defaults to None.
            scheduler (optim.lr_scheduler._LRScheduler, optional): Learning rate scheduler. Defaults to None.
            scaler (GradScaler, optional): GradScaler for mixed precision. Defaults to None.
            metrics (list[Callable], optional): List of metrics to run as callbacks.
                Defaults to None.
            cbs (list[Callback], optional): List of callbacks. Defaults to None.
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scaler = scaler
        self.scheduler = scheduler
        if device == torch.device("cuda") and torch.cuda.is_available():
            self.device = device
            if self.scaler is None:
                self.scaler = GradScaler()
            self.model = self.model.to(self.device)
        else:
            self.device = torch.device("cpu")
        self.state_dict = {}
        self.metrics = metrics
        self.mbar = None
        self.cbs = cbs

    def fit(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
        num_epochs: int,
        lr: float = 1e-3,
        wd: float = 0.0,
        grad_clip: float = 0.0,
    ) -> None:
        """Fits the model to the training data.

        Args:
            train_loader (DataLoader): The training data loader.
            test_loader (DataLoader): The validation data loader.
            num_epochs (int): The number of epochs to train for.
            lr (float, optional): Learning rate. Defaults to 1e-3.
            wd (float, optional): Weight decay. Defaults to 0.0.
            grad_clip (float, optional): Gradient clipping. Defaults to 0.0.
        """
        if self.optimizer is None:
            self.optimizer = optim.AdamW(
                self.model.parameters(), lr=lr, weight_decay=wd
            )

        if self.scheduler is None:
            self.scheduler = optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=lr,
                steps_per_epoch=int(
                    np.ceil(len(train_loader.dataset) / train_loader.batch_size)
                ),
                epochs=num_epochs,
            )
        self.state_dict["train_loss"] = []
        self.state_dict["valid_loss"] = []
        self.state_dict["num_epochs"] = num_epochs
        self.state_dict["batch_size"] = train_loader.batch_size
        self.state_dict["dataset_len"] = len(train_loader.dataset)
        self.state_dict["epoch"] = 0
        self.state_dict["metrics"] = {}
        self.mbar = master_bar(range(num_epochs))

        for epoch in self.mbar:
            self.state_dict["epoch"] = epoch + 1
            train_loss = self._train_epoch(
                self.model,
                train_loader,
                self.criterion,
                self.optimizer,
                self.scheduler,
                self.device,
                self.scaler,
            )

            val_loss, outputs, labels = self.evaluate(test_loader)
            if self.cbs is not None:
                for cb in self.cbs:
                    if (
                        cb.requirements & CallbackRequirement.TRAIN_LOSS
                        or cb.requirements & CallbackRequirement.VALID_LOSS
                        or cb.requirements & CallbackRequirement.PERSISTENT_DATA
                    ):
                        cb(
                            self.state_dict,
                            self.mbar,
                            train_loss=train_loss,
                            valid_loss=val_loss,
                            outputs=outputs,
                            labels=labels,
                        )

    def _train_epoch(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        criterion: nn.Module,
        opt: optim.Optimizer,
        scheduler: optim.lr_scheduler._LRScheduler,
        device: torch.device,
        scaler: GradScaler = None,
    ) -> np.ndarray:
        """Train the model for one epoch.

        Args:
            model (nn.Module): The PyTorch model to train.
            train_loader (DataLoader): The training data loader.
            criterion (nn.Module): The loss function to use.
            opt (optim.Optimizer): The optimizer to use.
            scheduler (optim.lr_scheduler._LRScheduler): The learning rate scheduler.
            device (torch.device): The device to use for training.
            scaler (GradScaler, optional): GradScaler for mixed precision training.
                Defaults to None.

        Returns:
            np.ndarray: The training losses for each batch.
        """
        model.train()
        losses = []
        for data in progress_bar(train_loader, parent=self.mbar):
            # Mixed Precision if available
            if scaler is not None and device.type == "cuda":
                with autocast(device.type, torch.bfloat16):
                    inputs, labels = data
                    inputs = inputs.squeeze() if len(inputs.shape) > 2 else inputs
                    labels = labels.squeeze() if len(labels.shape) > 2 else labels
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    output = model(inputs)
                    output = output.unsqueeze(0) if len(output.shape) == 1 else output
                    loss = criterion(output, labels)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                inputs, labels = data
                inputs = inputs.squeeze() if len(inputs.shape) > 2 else inputs
                labels = labels.squeeze() if len(labels.shape) > 2 else labels
                inputs = inputs.to(device)
                labels = labels.to(device)

                output = model(inputs)
                output = output.unsqueeze(0) if len(output.shape) == 1 else output
                loss = criterion(output, labels)
                loss.backward()
                opt.step()
            losses.append(loss.detach().cpu().numpy())
            scheduler.step()

        losses = np.array(losses)
        self.state_dict["train_loss"] = losses
        if "train_loss" not in self.state_dict["metrics"].keys():
            self.state_dict["metrics"]["train_loss"] = []
        self.state_dict["metrics"]["train_loss"].append(np.mean(losses))
        return losses

    def evaluate(
        self, test_loader: DataLoader
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate the model on the validation set.

        Args:
            test_loader (DataLoader): The validation data loader.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: The validation loss, outputs, and labels.
        """
        self.model.eval()

        with torch.no_grad():
            losses, outputs, labels = [], [], []
            for data in progress_bar(test_loader, parent=self.mbar):
                inputs, label = data
                inputs = inputs.squeeze() if len(inputs.shape) > 2 else inputs
                label = label.squeeze() if len(label.shape) > 2 else label
                inputs = inputs.to(self.device)
                label = label.to(self.device)

                output = self.model(inputs)
                output = output.unsqueeze(0) if len(output.shape) == 1 else output
                loss = self.criterion(output, label)
                losses.append(loss.squeeze().detach().cpu().numpy())
                outputs.append(output.detach().cpu().numpy())
                labels.append(label.detach().cpu().numpy())

        losses = np.mean(losses)
        outputs = np.concatenate(outputs)
        labels = np.concatenate(labels)

        if "valid_loss" in self.state_dict:
            self.state_dict["valid_loss"] = np.append(
                self.state_dict["valid_loss"], losses
            )
        else:
            self.state_dict["valid_loss"] = losses.reshape(-1, 1)

        self.state_dict["metrics"]["valid_loss"] = np.mean(losses)
        self.state_dict["valid_output"] = outputs
        self.state_dict["valid_label"] = labels

        epoch_output = {}
        epoch_output["valid_loss"] = losses
        epoch_output["valid_output"] = outputs
        epoch_output["valid_label"] = labels
        epoch_output["train_loss"] = self.state_dict["metrics"]["train_loss"]

        if self.metrics is not None:
            for metric in self.metrics:
                self.state_dict["metrics"][f"valid_{metric}"] = metric(
                    self.state_dict, **epoch_output
                )

        return losses, outputs, labels

    def predict(self, test_loader: DataLoader) -> np.ndarray:
        """Predict the output of the model on the test set.

        Args:
            test_loader (DataLoader): The test data loader.

        Returns:
            np.ndarray: The model's output on the test set.
        """
        _, outputs, _ = self.evaluate(test_loader)
        return outputs


class PlotGraphCallback(Callback):
    """
    A callback that plots the training and validation loss curves during training.

    This callback requires the following attributes in the state_dict:
    - train_loss: a list of training losses for each batch
    - valid_loss: a list of validation losses for each epoch

    This callback also requires the following attributes in the state_dict:
    - batch_size: the size of the training batches
    - dataset_len: the length of the training dataset
    - num_epochs: the number of training epochs

    The callback plots the training and validation loss curves using matplotlib,
    and updates the progress bar using the ConsoleMasterBar object.

    Example usage:

    ```
    from models.learner.callbacks import PlotGraphCallback

    plot_callback = PlotGraphCallback()
    learner = Learner(callbacks=[plot_callback])
    learner.train(...)
    ```
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.requirements = (
            CallbackRequirement.TRAIN_LOSS
            | CallbackRequirement.VALID_LOSS
            | CallbackRequirement.PERSISTENT_DATA
        )
        self.values = {"train_loss": None, "valid_loss": None}

    def __call__(self, state_dict: dict, mbar: ConsoleMasterBar, **outputs):
        """
        Update the training and validation loss values and plot them on a graph.

        Args:
            state_dict (dict): A dictionary containing the current state of the training process.
                It should contain the following keys:
                - "train_loss": A list of training loss values.
                - "valid_loss": A list of validation loss values.
                - "batch_size": The size of the training batches.
                - "dataset_len": The total number of samples in the dataset.
                - "num_epochs": The total number of epochs to train for.
                - "epoch": The current epoch number.
            mbar (ConsoleMasterBar): A progress bar object used to display the graph.
            **outputs: Additional outputs that may be passed to the method (not used in this
            implementation).
        """
        if len(state_dict["train_loss"]) == 0 or len(state_dict["valid_loss"]) == 0:
            return
        # measurements are taken every batch of len batch_size until the last batch
        # which could be of size < batch size
        if self.values["train_loss"] is None:
            self.values["train_loss"] = np.array(state_dict["train_loss"])
        else:
            self.values["train_loss"] = np.append(
                self.values["train_loss"], state_dict["train_loss"]
            )
        last_minibatch_size = len(state_dict["train_loss"]) % state_dict["batch_size"]
        train_x = (
            np.arange(state_dict["dataset_len"] // state_dict["batch_size"])
            * state_dict["batch_size"]
        )
        train_x = np.append(train_x, [train_x[-1] + last_minibatch_size])
        train_x = np.concatenate(
            [
                train_x + state_dict["dataset_len"] * i
                for i in range(state_dict["epoch"])
            ]
        )

        if len(state_dict["valid_loss"]) == 1:
            valid_x = np.array([1])
        else:
            valid_x = (
                np.arange(1, len(state_dict["valid_loss"]) + 1)
                * state_dict["dataset_len"]
            )
        valid_loss = np.array(state_dict["valid_loss"])
        y = np.concatenate([self.values["train_loss"], valid_loss])
        graphs = [[train_x, self.values["train_loss"]], [valid_x, valid_loss]]
        x_margin = 20.0
        y_margin = 0.05
        x_bounds = [
            0 - x_margin,
            state_dict["dataset_len"] * state_dict["num_epochs"] + x_margin,
        ]
        y_bounds = [0.0 - y_margin, np.max(y) + y_margin]

        mbar.update_graph(graphs, x_bounds, y_bounds)


class SaveModelCallback(Callback):
    """
    Callback to save the best, every or last model based on a given metric during training.

    Args:
        model (nn.Module): The PyTorch model to be saved.
        strategy (str): The strategy to save the model. Can be "best", "every" or "last".
        root_dir (str): The root directory to save the model.
        model_pth (str): The path to save the model.
        metric (str): The metric to use for saving the model. Can be "train_loss" or "valid_loss".

    Raises:
        NotImplementedError: If the strategy is not "best", "every" or "last".
        NotImplementedError: If the metric is not "train_loss" or "valid_loss".
    """

    def __init__(
        self,
        model: nn.Module,
        strategy: str,
        root_dir: str,
        model_pth: str,
        metric: str,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.model = model
        if strategy not in ["best", "every", "last"]:
            raise NotImplementedError("Only best, every and last strategy is supported")
        self.strategy = strategy
        if not os.path.exists(root_dir):
            os.makedirs(root_dir)
        self.model_pth = os.path.join(root_dir, model_pth)
        if metric not in ["train_loss", "valid_loss"]:
            raise NotImplementedError("Only train_loss and valid_loss is supported")
        self.metric = metric
        self.best_metric = np.inf
        self.requirements = CallbackRequirement.PERSISTENT_DATA

    def __call__(self, state_dict: dict, mbar: ConsoleMasterBar, **outputs):
        """
        Method to save the model based on the given strategy and metric.

        Args:
            state_dict (dict): The state dictionary containing the current epoch and metric values.
            mbar (ConsoleMasterBar): The console progress bar.
            **outputs: Additional outputs.
        """
        if self.strategy == "best":
            if self.best_metric > state_dict[self.metric][-1]:
                self.best_metric = state_dict[self.metric][-1]
                torch.save(self.model.state_dict(), self.model_pth)
                self._print_save(mbar)
        elif self.strategy == "every":
            torch.save(self.model.state_dict(), self.model_pth)
            self._print_save(mbar)
        elif self.strategy == "last":
            if state_dict["epoch"] == state_dict["num_epochs"]:
                torch.save(self.model.state_dict(), self.model_pth)
                self._print_save(mbar)

    def _print_save(self, mbar: ConsoleMasterBar):
        """
        Method to print the saved model information.

        Args:
            mbar (ConsoleMasterBar): The console progress bar.
        """
        mbar.write(
            f"Model with {self.metric}: {self.best_metric} saved to "
            + f"{self.model_pth} by strategy {self.strategy}"
        )


class ModelProgressCallback(Callback):
    """
    A callback that logs the progress of a model during training and validation.

    Attributes:
        metrics (list): A list of metric names to log.
        losses (list): A list of loss names to log.
        requirements (CallbackRequirement): A flag that specifies the requirements of this callback.
        values (dict): A dictionary that stores the latest values of the logged metrics and losses.

    Methods:
        __call__(self, state_dict: dict, mbar: ConsoleMasterBar, **outputs):
            Logs the progress of the model.
    """

    def __init__(self, metrics: list, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.losses = ["train_loss", "valid_loss"]
        self.metrics = metrics
        self.requirements = (
            CallbackRequirement.TRAIN_LOSS
            | CallbackRequirement.VALID_LOSS
            | CallbackRequirement.PERSISTENT_DATA
        )
        self.values = {}

    def __call__(self, state_dict: dict, mbar: ConsoleMasterBar, **outputs):
        """
        Update the values of the monitored metrics and print them to the console.

        Args:
            state_dict (dict): A dictionary containing the current state of the training loop.
            mbar (ConsoleMasterBar): A console progress bar to display the metrics.
            **outputs: Additional outputs that may be passed to the callback.

        Returns:
            None
        """
        if len(state_dict["train_loss"]) == 0 or len(state_dict["valid_loss"]) == 0:
            return
        for metric in self.metrics:
            self.values[metric] = state_dict["metrics"][metric][-1]

        if state_dict["epoch"] == 1:
            mbar.write([metric for metric in self.losses + self.metrics], table=True)

        if "train_loss" in self.losses:
            num_batches = int(
                np.ceil(state_dict["dataset_len"] / state_dict["batch_size"])
            )
            self.values["train_loss"] = np.mean(state_dict["train_loss"][-num_batches:])
        if "valid_loss" in self.losses:
            self.values["valid_loss"] = state_dict["valid_loss"][-1]

        mbar.write(
            [
                f"{l:.6f}" if isinstance(l, float) else str(l)
                for l in [self.values[metric] for metric in self.losses + self.metrics]
            ],
            table=True,
        )
