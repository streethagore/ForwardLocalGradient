import os
import sys
import pickle
import numpy as np
from PIL import Image
from torchvision.datasets.vision import VisionDataset
from utils.tools import print_debug


class Imagenet32(VisionDataset):
    """
    """

    def __init__(self, root, train=True, transform=None, target_transform=None, cuda=False, sz=32, n_arrays=10):

        super(Imagenet32, self).__init__(root, transform=transform,
                                         target_transform=target_transform)
        self.base_folder = ''
        self.train = train  # training set or test set
        self.cuda = cuda

        self.data = []
        self.targets = []

        if not self.train:
            n_arrays = 1

        # now load the picked numpy arrays
        for i in range(1, 1 + n_arrays):
            if self.train:
                file_name = 'train_data_batch_' + str(i)
            else:
                file_name = 'val_data'

#             print_debug(self.root, self.base_folder, file_name)
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])
        self.targets = [t - 1 for t in self.targets]
        self.data = np.vstack(self.data).reshape((-1, 3, sz, sz))
        if self.cuda:
            import torch
            self.data = torch.FloatTensor(self.data).cuda()  # .half().cuda()#type(torch.cuda.HalfTensor)
        else:
            self.data = self.data.transpose((0, 2, 3, 1))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        if self.cuda:
            img = self.transform(img)
            return img, target

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)
