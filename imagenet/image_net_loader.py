from torch.utils.data import Dataset


class ImageNetDataset(Dataset):
    def __init__(self, ds, transform=None):
        self.ds = ds
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):

        image = self.ds[idx]['image']
        label = self.ds[idx]['label']

        if self.transform:
            image = self.transform(image)

        return image, label
