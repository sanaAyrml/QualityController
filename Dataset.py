import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import Dataset, DataLoader,Subset

def get_class_i(x, y, i):
    """
    x: trainset.train_data or testset.test_data
    y: trainset.train_labels or testset.test_labels
    i: class label, a number between 0 to 9
    return: x_i
    """
    # Convert to a numpy array
    y = np.array(y)
    # Locate position of labels that equal to i
    pos_i = np.argwhere(y == i)
    # Convert the result into a 1-D list
    pos_i = list(pos_i[:, 0])
    # Collect all data that match the desired label
    x_i = [x[j] for j in pos_i]

    return x_i


class DatasetMaker(Dataset):
    def __init__(self, datasets, transformFunc):
        """
        datasets: a list of get_class_i outputs, i.e. a list of list of images for selected classes
        """
        self.labels = np.array(self.assign_labels(datasets))
        self.datasets = []
        for d in datasets:
            self.datasets += d
        self.lengths = [len(d) for d in datasets]
        print(self.lengths)
        self.transformFunc = transformFunc

    def __getitem__(self, i):
        img = self.transformFunc(self.datasets[i])
        return img, self.labels[i], 0

    def __len__(self):
        return sum(self.lengths)

    def assign_labels(self,datasets, verbose=False):
        """
        Given the absolute index, returns which bin it falls in and which element of that bin it corresponds to.
        """
        labels = []
        label = 0
        for d in datasets:
            for i in range(len(d)):
                labels.append(label)
            label += 1
            print(label)
        return labels