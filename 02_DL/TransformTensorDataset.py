from torch.utils.data.dataset import Dataset, ConcatDataset, TensorDataset

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
