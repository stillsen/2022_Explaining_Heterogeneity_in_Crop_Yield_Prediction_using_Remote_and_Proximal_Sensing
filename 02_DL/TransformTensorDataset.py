from torch.utils.data.dataset import Dataset, ConcatDataset, TensorDataset
from torchvision.datasets.vision import VisionDataset
from lightly.data import LightlyDataset
from typing import Any, Callable, List, Optional, Tuple
from PIL import Image
import numpy as np

class TransformTensorDataset(Dataset):
    """TensorDataset with support of transforms/augmentations.
    """
    def __init__(self, tensors:tuple, transform=None):
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[0][index]

        if self.transform:
            x = self.transform(image=x)

        y = self.tensors[1][index]

        # return torch.from_numpy(x), torch.from_numpy(y)
        return x['image'],y

    def __len__(self):
        # return self.tensors[0].size(0)
        return len(self.tensors[0])
#
class CustomVisionDataset(VisionDataset):

    def __init__(self, pil_imgs:tuple, transform:Optional[Callable] = None):
        super().__init__(root=None,transform=transform)
        self.dataset = pil_imgs
        # self.transform = transform
        # populate it with the torch dataset
        # if transform is not None:
        #     self.dataset.transform = transform

    def __getitem__(self, index):
        x = Image.fromarray(np.uint8(self.dataset[0][index]))
        # x = self.dataset[0][index]

        if self.transform:
            x = self.transform(x)

        y = self.dataset[1][index]

        # return torch.from_numpy(x), torch.from_numpy(y)
        return x,y

    def __len__(self):
        # return self.tensors[0].size(0)
        return len(self.dataset[0])
