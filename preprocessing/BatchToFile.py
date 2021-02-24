# Author Markus Laubenthal

import numpy as np
import h5py

class BatchProcessToFile:
    def __init__(self, filename):
        self.filename = filename
        self.f = None
        pass

    def open_file(self):
        self.f = h5py.File(self.filename,'w')

    def close_file(self):
        self.f.close()

    def initialize_dataset(self, dataset_name, output_shape):
        dataset = None
        save_shape = (0,) + output_shape
        max_shape = (None,) + output_shape

        self.open_file()

        if dataset_name in self.f:
            dataset = self.f[dataset_name]
        else:
            dataset = self.f.create_dataset(dataset_name, save_shape, dtype='f', maxshape=max_shape)
        return dataset


    def batch(self, processor, items, dataset_name, batch_size=32):
        nItems = 0
        if isinstance(items, list):
            nItems = len(items)
        elif isinstance(items, np.ndarray):
            nItems = items.shape[0]

        output_shape = processor.get_output_shape()

        dataset = self.initialize_dataset(dataset_name, output_shape)

        batch_start = 0
        while batch_start < nItems:
            batch_end = batch_start + batch_size
            result = processor.execute(items[batch_start:batch_end])
            last_index = dataset.shape[0]
            dataset.resize(dataset.shape[0] + result.shape[0], axis = 0)
            dataset[last_index:] = result
            batch_start += batch_size
        self.close_file()
