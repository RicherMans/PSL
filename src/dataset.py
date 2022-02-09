import torch
import pandas as pd
import numpy as np
from loguru import logger
from h5py import File
from typing import Dict, List, Iterable, Tuple, Union
from torch.multiprocessing import Queue


class UnlabeledHDF5Dataset(torch.utils.data.Dataset):

    def __init__(self, data_frame: pd.DataFrame, transforms=None):
        super(UnlabeledHDF5Dataset, self).__init__()
        self.transforms = torch.nn.Sequential(
        ) if transforms == None else transforms
        self._datasetcache = {}
        self._dataframe = data_frame
        filename, hdf5path = self._dataframe.iloc[0][['filename', 'hdf5path']]
        with File(hdf5path, 'r') as store:
            self.datadim = store[filename].shape[-1]

    def __del__(self):
        if self._datasetcache is not None:
            for k, cache in self._datasetcache.items():
                try:
                    cache.close()
                except:
                    pass

    def _readdata(self, hdf5path: str, fname: str) -> torch.Tensor:
        if not hdf5path in self._datasetcache:
            self._datasetcache[hdf5path] = File(hdf5path, 'r')
        data = self._datasetcache[hdf5path][f"{fname}"][:]
        if np.issubdtype(data.dtype, np.integer):
            data = (data / 32768.).astype('float32')
        return self.transforms(torch.as_tensor(data, dtype=torch.float32))

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, str]:
        fname, hdf5path = self._dataframe.iloc[index][['filename', 'hdf5path']]
        return self._readdata(hdf5path, fname), fname

    def __len__(self):
        return len(self._dataframe)


class WeakHDF5Dataset(UnlabeledHDF5Dataset):
    """
    HDF5 dataset indexed by a labels dataframe. 
    Indexing is done via the dataframe since we want to preserve some storage
    in cases where oversampling is needed ( pretty likely )
    """

    def __init__(
        self,
        data_frame: pd.DataFrame,
        transforms: torch.nn.Sequential = None,
        num_classes: int = None,
    ):
        super(WeakHDF5Dataset, self).__init__(data_frame=data_frame,
                                              transforms=transforms)
        if num_classes is None:
            unique_nums = np.unique(np.concatenate(data_frame['labels']))
            self._num_classes = len(unique_nums)
        else:
            self._num_classes = num_classes
        logger.debug(f"Num classes in dataloader: {self._num_classes}")

    def __getitem__(self,
                    index: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        fname, label_idxs, hdf5path = self._dataframe.iloc[index][[
            'filename', 'labels', 'hdf5path'
        ]]
        #Generate target from int list [1,5,7] --> [0,1,0,0,0,1,0,1]
        target = torch.zeros(self._num_classes, dtype=torch.float32).scatter_(
            0, torch.as_tensor(label_idxs), 1)
        data = self._readdata(hdf5path, fname)
        return data, target, fname


class WeakChunkedHDF5Dataset(WeakHDF5Dataset):

    def __init__(self,
                 data_frame,
                 transforms=None,
                 num_classes: int = None,
                 sample_rate: int = 16000):
        super().__init__(data_frame, transforms, num_classes)
        self._sr = sample_rate

    def _readdata(self, hdf5path: str, fname: str, from_time: int,
                  to_time: int) -> torch.Tensor:
        if not hdf5path in self._datasetcache:
            self._datasetcache[hdf5path] = File(hdf5path, 'r')
        data = self._datasetcache[hdf5path][f"{fname}"][from_time:to_time]
        if np.issubdtype(data.dtype, np.integer):
            data = (data / 32768.).astype('float32')
        return self.transforms(torch.as_tensor(data, dtype=torch.float32))

    def __getitem__(self,
                    index: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        fname, label_idxs, from_time, to_time, hdf5path = self._dataframe.iloc[
            index][['filename', 'labels', 'from', 'to', 'hdf5path']]
        #Generate target from int list [1,5,7] --> [0,1,0,0,0,1,0,1]
        target = torch.zeros(self._num_classes, dtype=torch.float32).scatter_(
            0, torch.as_tensor(label_idxs), 1)
        from_time = int(from_time * self._sr)
        to_time = int(to_time * self._sr)
        data = self._readdata(hdf5path, fname, from_time, to_time)
        return data, target, fname


class BalancedSampler(torch.utils.data.Sampler):

    def __init__(self,
                 labels: List[List[int]],
                 num_classes=None,
                 random_state=None):
        self._random_state = np.random.RandomState(seed=random_state)
        if num_classes is None:
            unique_labels = np.unique(np.concatenate(labels))
            self._num_classes = len(unique_labels)
        else:
            self._num_classes = num_classes
        label_to_idx_list = [[] for _ in range(self._num_classes)]
        label_to_length = []
        for idx, lbs in enumerate(labels):
            for lb in lbs:
                label_to_idx_list[lb].append(idx)
        for i in range(len(label_to_idx_list)):
            label_to_idx_list[i] = np.array(label_to_idx_list[i])
            self._random_state.shuffle(label_to_idx_list[i])
            label_to_length.append(len(label_to_idx_list[i]))
        self.label_to_idx_list = label_to_idx_list
        self.label_to_length = label_to_length
        self.pointers_of_classes = [0] * self._num_classes
        self._len = len(labels)
        self.queue = Queue()

    def getitemindex(self, lab_idx: int):
        '''
        returns next index, given a label index
        '''
        cur_item = self.pointers_of_classes[lab_idx]
        self.pointers_of_classes[lab_idx] += 1
        index = self.label_to_idx_list[lab_idx][cur_item]
        #Reshuffle and reset points if overlength
        if self.pointers_of_classes[lab_idx] >= self.label_to_length[
                lab_idx]:  #Reset
            self.pointers_of_classes[lab_idx] = 0
            self._random_state.shuffle(self.label_to_idx_list[lab_idx])
        return index

    def populate_queue(self):
        # Can be overwritten by subclasses
        classes_set = np.arange(self._num_classes).tolist()
        self._random_state.shuffle(classes_set)
        for c in classes_set:
            self.queue.put(c)  # Push to queue class indices

    def __iter__(self):
        while True:
            if self.queue.empty():
                self.populate_queue()
            lab_idx = self.queue.get()  # Get next item, single class index
            index = self.getitemindex(lab_idx)
            yield index

    def __len__(self):
        return self._len


def pad(tensorlist: List[torch.Tensor],
        padding_value: float = 0.) -> torch.Tensor:
    # Tensors are expected to be B, ..., T
    lengths = [f.shape[-1] for f in tensorlist]
    dims = tensorlist[0].shape
    trailing_dims = dims[:-1]
    batch_dim = len(lengths)
    num_raw_samples = max(lengths)
    out_dims = (batch_dim, ) + trailing_dims + (num_raw_samples, )
    out_tensor = torch.full(out_dims,
                            fill_value=padding_value,
                            dtype=torch.float32)
    for i, tensor in enumerate(tensorlist):
        length = tensor.shape[-1]
        out_tensor[i, ..., :length] = tensor[..., :length]
    return out_tensor


def sequential_pad(batches):
    datas, *targets, fnames = zip(*batches)
    targets = tuple(map(lambda x: torch.stack(x), targets))
    return pad(datas), *targets, fnames


class MultiDataLoader(torch.utils.data.IterableDataset):

    def __init__(self, **datasets: Dict[str, torch.utils.data.DataLoader]):
        self.dataloaders = datasets
        self.dataloader_iters = {
            k: iter(v)
            for k, v in self.dataloaders.items()
        }

    def __iter__(self):
        while True:
            datas = {}
            for key in self.dataloader_iters:
                try:
                    batch = next(self.dataloader_iters[key])
                except StopIteration:
                    # Reset iterator
                    self.dataloader_iters[key] = iter(self.dataloaders[key])
                    batch = next(self.dataloader_iters[key])
                datas[key] = batch
            yield datas
            datas = []

    def __len__(self):
        return min(len(dl) for dl in self.dataloaders.values())
