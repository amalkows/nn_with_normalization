import numpy as np
import random

from struct import *
from tqdm import tnrange, tqdm_notebook

class DataSet:
    input_size = 0
    output_size = 0
    samples_count = 0
    inputs = []
    outputs = []
    double_size = 8
    min_value = 0
    max_value = 0

    def __init__(self, file_path, zero_one):
        self.read_data_bin(file_path, zero_one)
        
    def scale_to_zero_one(self, data):
        data = (data - self.min_value)/(self.max_value-self.min_value) 
        return data
    
    def scale_to_original(self, data):
        return data*(self.max_value-self.min_value) + self.min_value
    
    def read_data_bin(self, file_path, zero_one):
        self.inputs = []
        self.outputs = []
        file = open(file_path, "rb")
        self.samples_count, self.input_size, self.output_size = unpack("iii", file.read(12))
                
        for i in tnrange(self.samples_count):
            self.inputs.append(unpack(str(self.input_size)+"d", file.read(self.double_size * self.input_size)))
            self.outputs.append(unpack(str(self.output_size)+"d", file.read(self.double_size * self.output_size)))

        self.inputs = np.array(self.inputs)
        self.outputs = np.array(self.outputs)

        self.min_value = self.inputs.min()        
        self.max_value = self.inputs.max()
        if(zero_one):
            self.inputs = self.scale_to_zero_one(self.inputs)
        
        file.close()
    
    def get_next_bach(self,count):
        indexes = random.sample(range(self.samples_count), count)
        return [self.inputs[indexes], self.outputs[indexes]]