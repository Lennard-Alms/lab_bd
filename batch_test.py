from preprocessing.BatchToFile import BatchProcessToFile
from preprocessing.Dummy.DummyProcessor import DummyProcessor
from preprocessing.FeatureExtractor import VGGFeatureExtractorMax
import numpy as np

processor = VGGFeatureExtractorMax()
batch_processor = BatchProcessToFile("test.h5")
items = np.random.randint(0,255, (500,200,200,3))
batch_processor.batch(processor, items, 'testset')
