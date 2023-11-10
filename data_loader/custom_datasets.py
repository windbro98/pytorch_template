from torch.utils.data import Dataset
import os
import pandas as pd
from utils.util import *

# example dataset
class OpticDepthDataset(Dataset):
    def __init__(self, data_dir, transforms, train):
        self.train_csv = pd.read_csv(os.path.join(data_dir, 'train.csv')) 
        self.test_csv = pd.read_csv(os.path.join(data_dir, 'test.csv')) 
        self.tfs = transforms
        self.train = train

    def __getitem__(self, index):
        if(self.train):
            return self.tfs(extract_image_L(self.train_csv.loc[index][0])), float(self.train_csv.loc[index][1])
        else:
            return self.tfs(extract_image_L(self.test_csv.loc[index][0])), float(self.test_csv.loc[index][1])
