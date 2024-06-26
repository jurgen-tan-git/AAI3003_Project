"""TfIdfDataset class for loading data from a csv file and a directory of articles
"""

import os

import numpy as np
import scipy
import torch
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import Dataset

from dataset.transformers_dataset import load_data


class TfIdfDataset(Dataset):
    """TfIdfDataset class for loading data from a csv file and a directory of articles

    :param labelled_csv: Path to the labelled CSV file.
    :type labelled_csv: str | os.PathLike
    :param articles_dir: Directory containing the articles.
    :type articles_dir: str | os.PathLike
    :param use_original_text: Whether to use the original text or not.
    :type use_original_text: bool, optional
    :param padded_shape: Shape to pad the input tensor to.
    :type padded_shape: tuple[int, int], optional
    """

    def __init__(
        self,
        labelled_csv: str | os.PathLike,
        articles_dir: str | os.PathLike,
        use_original_text: bool = False,
        padded_shape: tuple[int, int] = (0, 0),
    ) -> None:
        self.df = load_data(labelled_csv, articles_dir, use_original_text)
        self.vectorizer = TfidfVectorizer(
            tokenizer=word_tokenize, stop_words="english", max_features=10000
        )
        self.X = self.vectorizer.fit_transform(self.df["Text"])
        self.features = self.vectorizer.get_feature_names_out()
        self.y = self.df.iloc[:, 2:-1].to_numpy()
        self.padded_shape = padded_shape

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.X[idx]
        pad_x = self.padded_shape[0]
        pad_y = self.padded_shape[1] - x.shape[1]
        x = self._csr_matrix_to_tensor(x, (pad_x, pad_y))
        x = x.to_dense()
        y = self.y[idx]
        return x.float(), torch.tensor(y, dtype=torch.float).squeeze()

    def get_feature_names(self) -> list[str]:
        """Get the feature names.

        return: List of feature names.
        rtype: list[str]
        """
        return self.features

    def get_len_features(self) -> int:
        """Get the length of the features.

        return: Length of the features.
        rtype: int
        """
        return len(self.features)

    def _csr_matrix_to_tensor(
        self, csr: scipy.sparse.csr_matrix, padding: tuple[int, int] = (0, 0)
    ) -> torch.Tensor:
        """Convert a CSR matrix to a PyTorch tensor.

        :param csr: CSR matrix to convert.
        :type csr: scipy.sparse.csr_matrix
        :param padding: Padding to add to the tensor, defaults to (0, 0)
        :type padding: tuple[int, int], optional
        :return: PyTorch tensor.
        :rtype: torch.Tensor
        """
        coo = csr.tocoo()
        values = coo.data
        indices = np.vstack((coo.row + padding[0], coo.col + padding[1]))
        shape = [coo.shape[0] + padding[0], coo.shape[1] + padding[1]]

        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)

        return torch.sparse_coo_tensor(i, v, torch.Size(shape))

    def set_padded_shape(self, padded_shape: tuple[int, int] = (0, 0)) -> None:
        """Sets the padded shape after initialization.

        :param padded_shape: Shape to pad the input tensor to, defaults to (0, 0)
        :type padded_shape: tuple[int, int], optional
        """
        self.padded_shape = padded_shape
