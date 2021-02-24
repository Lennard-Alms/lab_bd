from preprocessing.BatchToFile import BatchProcessToFile
from preprocessing.Dummy.DummyProcessor import DummyProcessor
from preprocessing.FeatureExtractor import VGGFeatureExtractorMax
from preprocessing.ImageMutation import PatchMutation
import numpy as np

mutation_strategy = PatchMutation(np.empty((7,100,100,3)).astype(np.uint8), mutation_probability=0.2)
processor = VGGFeatureExtractorMax(mutation_strategy=mutation_strategy)
print(processor.get_no_of_outputs())
print(processor.get_output_shapes())
print(processor.get_dataset_name_postfixes())
batch_processor = BatchProcessToFile("test.h5")
items = np.random.randint(0,255, (500,200,200,3))
batch_processor.batch(processor, items, 'testset')
