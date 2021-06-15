from torch.utils.data import Dataset
import pickle
import torch

from Models import MiniModel


def unpickle(file):
    with open(file, 'rb') as fo:
        d = pickle.load(fo)
    return d


class BreakfastTexts(Dataset):

    def __init__(self, features_path):
        self.data = unpickle(features_path)

        self.mini_model = MiniModel()
        self.reshape_data()

    def reshape_data(self):
        for (i, feature) in self.data.items():
            self.data[i] = self.mini_model(feature)

    def __len__(self):
        return len(self.data.items())

    def __getitem__(self, index):
        return self.data[index+1]


class BreakfastClips(Dataset):

    def __init__(self):
        self.data = torch.zeros(100, 224, 224, 1024)            # !!! DUMMY DATA !!!
        self.data = torch.mean(self.data, dim=(1, 2))

    def get_sample(self, index, size):
        sample = []
        for i in range(index, index+size):
            sample.append(self[i])
        return sample

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
