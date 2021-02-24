from preprocessing.BatchToFile import BatchProcessToFile
from preprocessing.Dummy.DummyProcessor import DummyProcessor
import numpy as np

dummy = DummyProcessor()
batch_processor = BatchProcessToFile("test.h5")
items = np.random.randint(0,100, (44,))
batch_processor.batch(dummy, items, 'testset')
