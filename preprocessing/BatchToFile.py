# Author Markus Laubenthal

import numpy as np
import h5py

class BatchProcessToFile:
    def __init__(self, filename):
        self.filename = filename
        self.f = None
        self.mutation_strategy = None
        pass

    def open_file(self):
        self.f = h5py.File(self.filename,'w')

    def close_file(self):
        self.f.close()

    def initialize_dataset(self, dataset_name, output_shapes, postfixes):
        datasets = []
        self.open_file()
        for i, output_shape in enumerate(output_shapes):
            save_shape = (0,) + output_shape
            max_shape = (None,) + output_shape
            dset_name = dataset_name + "_" + postfixes[i]
            if dataset_name in self.f:
                dataset = self.f[dataset_name]
            else:
                dataset = self.f.create_dataset(dset_name, save_shape, dtype='f', maxshape=max_shape)
            datasets.append(dataset)
        return datasets


    def batch(self, processor, items, dataset_name, batch_size=32):
        nItems = 0
        if isinstance(items, list):
            nItems = len(items)
        elif isinstance(items, np.ndarray):
            nItems = items.shape[0]

        output_shapes = processor.get_output_shapes()
        postfixes = processor.get_dataset_name_postfixes()
        datasets = self.initialize_dataset(dataset_name, output_shapes, postfixes)

        batch_start = 0
        while batch_start < nItems:
            batch_end = batch_start + batch_size
            result = processor.execute(items[batch_start:batch_end])
            for i,dataset in enumerate(datasets):
                last_index = dataset.shape[0]
                dataset.resize(dataset.shape[0] + result[i].shape[0], axis = 0)
                dataset[last_index:] = result[i]
            batch_start += batch_size
        self.close_file()
