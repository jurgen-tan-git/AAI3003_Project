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

    :member NONE: No requirements.
    :member TRAIN_OUTPUT: Requires the training output.
    :member VALID_OUTPUT: Requires the validation output.
    :member TRAIN_LABEL: Requires the training label.
    :member VALID_LABEL: Requires the validation label.
    :member TRAIN_LOSS: Requires the training loss.
    :member VALID_LOSS: Requires the validation loss.
    :member PERSISTENT_DATA: Requires persistent data.
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

        :param state_dict: The state dictionary containing the current state of the model.
        :type state_dict: dict
        :param mbar: The console progress bar.
        :type mbar: ConsoleMasterBar
        """


class MetricCallback(Callback):
    """
    A callback that computes a metric on the model's validation output and label.

    :param requirements: The requirements for this callback.
    :type requirements: CallbackRequirement
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.requirements = (
            CallbackRequirement.VALID_OUTPUT | CallbackRequirement.VALID_LABEL
        )

    def __call__(self, state_dict: dict, **outputs):
        """
        Update the state_dict with the metric values for the current batch.

        param: state_dict: The state dictionary containing the current state of the training loop.
        type: state_dict: dict
        param: outputs: A dictionary containing the outputs of the model for the current batch.
        type: outputs: dict
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

        :return: The string representation of the class name.
        :rtype: str
        """
        return self.__class__.__name__.replace("Callback", "").lower()

    @abc.abstractmethod
    def _metric(self, state_dict: dict, valid_output, valid_label):
        """
        Calculates the metric for the validation set.

        :param state_dict: The state dictionary.
        :type state_dict: dict
        :param valid_output: The output of the validation set.
        :type valid_output: np.ndarray
        :param valid_label: The label of the validation set.
        :type valid_label: np.ndarray
        :return: The calculated metric for the validation set.
        :rtype: float
        """


class F1Callback(MetricCallback):
    """A callback that computes the F1 score of a model on a validation set.

    :param multilabel: Whether the targets are multi-label, defaults to True.
    :type multilabel: bool, optional
    :param threshold: Positive prediction threshold, defaults to 0.5.
    :type threshold: float, optional
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

    :param multilabel: Whether the problem is multilabel classification.
    :type multilabel: bool
    :param threshold: Positive prediction threshold, defaults to 0.85.
    :type threshold: float
    """

    def __init__(self, threshold: float = 0.85, multilabel: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.multilabel = multilabel

    def _metric(self, state_dict: dict, valid_output, valid_label) -> float:
        """
        Computes the accuracy of the model on the validation set.

        :param state_dict: The state dictionary of the model.
        :type state_dict: dict
        :param valid_output: The output of the model on the validation set.
        :type valid_output: np.ndarray
        :param valid_label: The labels of the validation set.
        :type valid_label: np.ndarray
        :return: The accuracy of the model on the validation set.
        :rtype: float
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

        :param model: The PyTorch model to train.
        :type model: nn.Module
        :param criterion: The loss function to use.
        :type criterion: nn.Module
        :param device: The device to use for training.
        :type device: torch.device
        :param optimizer: Optimizer for training, defaults to None.
        :type optimizer: optim.Optimizer, optional
        :param scheduler: Learning rate scheduler, defaults to None.
        :type scheduler: optim.lr_scheduler._LRScheduler, optional
        :param scaler: GradScaler for mixed precision, defaults to None.
        :type scaler: GradScaler, optional
        :param metrics: List of metrics to run as callbacks, defaults to None.
        :type metrics: list[MetricCallback], optional
        :param cbs: List of callbacks, defaults to None.
        :type cbs: list[Callback], optional
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

        :param train_loader: The training data loader.
        :type train_loader: DataLoader
        :param test_loader: The validation data loader.
        :type test_loader: DataLoader
        :param num_epochs: The number of epochs to train for.
        :type num_epochs: int
        :param lr: Learning rate, defaults to 1e-3.
        :type lr: float, optional
        :param wd: Weight decay, defaults to 0.0.
        :type wd: float, optional
        :param grad_clip: Gradient clipping, defaults to 0.0.
        :type grad_clip: float, optional
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

        :param model: The PyTorch model to train.
        :type model: nn.Module
        :param train_loader: The training data loader.
        :type train_loader: DataLoader
        :param criterion: The loss function to use.
        :type criterion: nn.Module
        :param opt: The optimizer to use.
        :type opt: optim.Optimizer
        :param scheduler: The learning rate scheduler.
        :type scheduler: optim.lr_scheduler._LRScheduler
        :param device: The device to use for training.
        :type device: torch.device
        :param scaler: GradScaler for mixed precision training, defaults to None.
        :type scaler: GradScaler, optional
        :return: The training losses for each batch.
        :rtype np.ndarray
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

        :param test_loader: The validation data loader.
        :type test_loader: DataLoader
        :return: The validation loss, outputs, and labels.
        :rtype: tuple[np.ndarray, np.ndarray, np.ndarray]
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

        :param test_loader: The test data loader.
        :type test_loader: DataLoader
        :return: The model's output on the test set.
        :rtype: np.ndarray
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

        :param state_dict: A dictionary containing the current state of the training process.
        :type state_dict: dict
        :param mbar: A progress bar object used to display the graph.
        :type mbar: ConsoleMasterBar
        :param outputs: Additional outputs that may be passed to the method (not used in this implementation).
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

    :param model: The PyTorch model to be saved.
    :type model: nn.Module
    :param strategy: The strategy to save the model. Can be "best", "every" or "last".
    :type strategy: str
    :param root_dir: The root directory to save the model.
    :type root_dir: str
    :param model_pth: The path to save the model.
    :type model_pth: str
    :param metric: The metric to use for saving the model. Can be "train_loss" or "valid_loss".
    :type metric: str
    :raise NotImplementedError: If the strategy is not "best", "every" or "last".
    :raise NotImplementedError: If the metric is not "train_loss" or "valid_loss".
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

        :param state_dict: The state dictionary containing the current epoch and metric values.
        :type state_dict: dict
        :param mbar: The console progress bar.
        :type mbar: ConsoleMasterBar
        :param outputs: Additional outputs.
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

        :param mbar: The console progress bar.
        :type mbar: ConsoleMasterBar
        """
        mbar.write(
            f"Model with {self.metric}: {self.best_metric} saved to "
            + f"{self.model_pth} by strategy {self.strategy}"
        )


class ModelProgressCallback(Callback):
    """
    A callback that logs the progress of a model during training and validation.

    :param metrics: A list of metric names to log.
    :type metrics: list
    :param losses: A list of loss names to log.
    :type losses: list
    :param requirements: A flag that specifies the requirements of this callback.
    :type requirements: CallbackRequirement
    :param values: A dictionary that stores the latest values of the logged metrics and losses.
    :type values: dict
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

        :param state_dict: A dictionary containing the current state of the training loop.
        :type state_dict: dict
        :param mbar: A console progress bar to display the metrics.
        :type mbar: ConsoleMasterBar
        :param outputs: Additional outputs that may be passed to the callback.
        :type outputs: dict
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
