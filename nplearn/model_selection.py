import numpy as np
from typing import Generator, Any, Tuple, Optional, Union

class train_test_split():
    def __init__(self, train_size: Optional[Union[float, int]] = None, test_size: Optional[Union[float, int]] = None, shuffle: bool = False, random_state: Optional[int] = None, stratify: Optional[np.ndarray] = None, *args, **kwargs):
        if train_size is None and test_size is None:
            self.test_size = 0.25
            self.train_size = None
        else:
            self.train_size = train_size
            self.test_size = test_size
        self.shuffle = shuffle
        self.random_state = random_state
        self.stratify = stratify
    
    def _get_split_indices(self, n_samples : int)-> Tuple[int, int]:
        train_size = self.train_size
        test_size = self.test_size

        if (test_size is not None) and (train_size is None):
            if isinstance(test_size, float):
                if not (0.0 < test_size < 1.0):
                    raise ValueError("The test_size proportion must be between 0.0 and 1.0.")
                n_test = int(n_samples * test_size)
                n_train = n_samples - n_test
            elif isinstance(test_size, int):
                if not (0 < test_size < n_samples):
                    raise ValueError("The test_size sample count must be less than the total number of samples.")
                n_test = test_size
                n_train = n_samples - n_test
            else:
                raise ValueError("test_size must be int or float.")
        elif (train_size is not None) and (test_size is None):
            if isinstance(train_size, float):
                if not (0.0 < train_size < 1.0):
                    raise ValueError("The train_size proportion must be between 0.0 and 1.0.")
                n_train = int(n_samples * train_size)
                n_test = n_samples - n_train
            elif isinstance(train_size, int):
                if not (0 < train_size < n_samples):
                    raise ValueError("The train_size sample count must be less than the total number of samples.")
                n_train = train_size
                n_test = n_samples - n_train
            else:
                raise ValueError("train_size must be int or float.")
        else:
            if isinstance(test_size, float) and isinstance(train_size, float):
                if test_size + train_size > 1.0: # 使用 > 1.0 进行检查，更安全
                     raise ValueError("The sum of train_size and test_size proportions cannot be greater than 1.0.")
                n_test = int(n_samples * test_size)
                n_train = n_samples - n_test
            elif isinstance(test_size, int) and isinstance(train_size, int):
                if test_size + train_size > n_samples:
                    raise ValueError("The sum of train_size and test_size counts cannot be greater than n_samples.")
                n_test = test_size
                n_train = train_size
            else:
                raise ValueError("train_size and test_size must be of the same type (both int or both float).")
            
        if (n_train <= 0 or  n_test <= 0):
            raise ValueError("The size of train set and test set could'nt be zero.")
        
        return n_train, n_test
    
    def split(self, X : np.ndarray, y : np.ndarray)-> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples.")
        n_samples = X.shape[0]
        n_train, n_test = self._get_split_indices(n_samples)

        if self.random_state is not None:
            rng = np.random.default_rng(self.random_state)
        else:
            rng = np.random.default_rng()

        if self.stratify is None:
            indices = np.arange(n_samples)
            if self.shuffle:
                shuffled_indices = rng.permutation(n_samples)
            else:
                shuffled_indices = indices
            test_indices = shuffled_indices[:n_test]
            train_indices = shuffled_indices[n_test:]
        else:
            stratify_y = self.stratify if self.stratify.ndim == 1 else self.stratify.flatten()
            unique_classes, class_counts = np.unique(stratify_y, return_counts=True)
            test_indices_list = []
            for class_val, class_count in zip(unique_classes, class_counts):
                class_indices = np.where(stratify_y == class_val)[0]
                class_test_size = int(np.round(n_test * (class_count / n_samples)))
                if class_test_size <= 0 and class_count > 0:
                    class_test_size = 1
                if self.shuffle:
                    class_indices = rng.permutation(class_indices)
                test_indices_list.append(class_indices[:class_test_size])
            test_indices = np.concatenate(test_indices_list)
            if len(test_indices) > n_test:
                test_indices = rng.permutation(test_indices)[:n_test]
            full_indices = np.arange(n_samples)
            train_indices = np.setdiff1d(full_indices, test_indices)

        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]
        return X_train, X_test, y_train, y_test


class KFold():
    def __init__(self, n_splits : int = 5, shuffle : bool = False, random_state : int = 42, *args, **kwargs):
        if not isinstance(n_splits, int) or n_splits <= 1:
            raise ValueError(f"n_splits must be int larger than 1, now it is {n_splits}")
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state : int = random_state
    
    def get_n_splits(self, X : np.ndarray = None, y : np.ndarray = None, groups = None) -> int:
        return self.n_splits
    
    def split(self, X : np.ndarray, y : np.ndarray = None, groups = None) -> Generator[Tuple[np.ndarray, np.ndarray], Any, Any]:
        if X.ndim == 0:
            n_samples = 1
        elif X.ndim == 1:
            n_samples = X.shape[0]
        else:
            n_samples = X.shape[0]
        if self.n_splits > n_samples:
            raise ValueError("n_splits couldn't be larger than n_samples")
        indices = np.arange(n_samples)
        if self.shuffle:
            if self.random_state is not None:
                rng = np.random.default_rng(self.random_state)
            else:
                rng = np.random.default_rng()
            indices = rng.permutation(n_samples)
        
        base_size = n_samples // self.n_splits
        remainder = n_samples % self.n_splits

        fold_sizes = np.full(self.n_splits, base_size, dtype=int)
        fold_sizes[:remainder] += 1

        fold_starts = np.concatenate([[0], np.cumsum(fold_sizes)])
        for i in range(self.n_splits):
            start, stop = fold_starts[i], fold_starts[i + 1]
            test_index = indices[start:stop]
            train_index =np.concatenate([indices[:start], indices[stop:]])
            yield train_index, test_index