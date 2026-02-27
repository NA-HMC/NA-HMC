import torch
from torch.utils.data import Dataset
import numpy as np
import os

class NpyDataset(Dataset):
    def __init__(self, nums=400, root_dir='exp/image_samples/trainset_celeba', steps=['999']):
        """
        Args:
            root_dir (str)
        """
        super(NpyDataset, self).__init__()
        self.steps = steps
        self.root_dir = root_dir
        self.num = nums
        self.file_list = [f"{i}.npy" for i in range(nums)] 

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self):
            raise IndexError("Index out of range")
        data = []
        for step in self.steps:
            if step == self.steps[-1]:
                file_path = os.path.join(self.root_dir, step, 'x0_' + self.file_list[idx])
            else:
                file_path = os.path.join(self.root_dir, step, 'x0_pred_' + self.file_list[idx])
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File {file_path} does not exist.")

            data.append(np.expand_dims(np.load(file_path), axis=0))
        data = np.vstack(data)
        gt = np.load(os.path.join(self.root_dir, 'orig', self.file_list[idx]))
        y_0 = np.load(os.path.join(self.root_dir, 'y_0', 'y0_'+self.file_list[idx]))

        return torch.tensor(data, dtype=torch.float32), torch.tensor(gt, dtype=torch.float32), torch.tensor(y_0, dtype=torch.float32)


class NpyDataset_cache(Dataset):
    def __init__(self, nums=400, root_dir='exp/image_samples/trainset_celeba', steps=['999']):
        """
        Args:
            root_dir (str)
        """
        super(NpyDataset_cache, self).__init__()
        self.steps = steps
        self.root_dir = root_dir
        self.num = nums
        self.file_list = [f"{i}.npy" for i in range(nums)]
        self._read_all_data()

    def _read_all_data(self):
        self.data_list = []
        self.gt_list = []
        self.y_0_list = []
        for idx in range(self.__len__()):
            data = []
            for step in self.steps:
                if step == self.steps[-1]:
                    file_path = os.path.join(self.root_dir, step, 'x0_' + self.file_list[idx])
                else:
                    file_path = os.path.join(self.root_dir, step, 'x0_pred_' + self.file_list[idx])
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"File {file_path} does not exist.")

                data.append(np.expand_dims(np.load(file_path), axis=0))
            data = np.vstack(data)
            gt = np.load(os.path.join(self.root_dir, 'orig', self.file_list[idx]))
            y_0 = np.load(os.path.join(self.root_dir, 'y_0', 'y0_'+self.file_list[idx]))
            self.data_list.append(data)
            self.gt_list.append(gt)
            self.y_0_list.append(y_0)
        # return
        self.data_list = np.stack(self.data_list, axis=0)
        self.gt_list = np.stack(self.gt_list, axis=0)
        self.y_0_list = np.stack(self.y_0_list, axis=0)

    def get_all_data(self):
        return torch.tensor(self.data_list, dtype=torch.float32), torch.tensor(self.gt_list, dtype=torch.float32), torch.tensor(self.y_0_list, dtype=torch.float32)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self):
            raise IndexError("Index out of range")
        data = self.data_list[idx]
        gt = self.gt_list[idx]
        y_0 = self.y_0_list[idx]

        return torch.tensor(data, dtype=torch.float32), torch.tensor(gt, dtype=torch.float32), torch.tensor(y_0, dtype=torch.float32)

if __name__ == "__main__":
    root_dir = "path/to/npy/files"
    dataset = NpyDataset(root_dir=root_dir)

    print(f"Dataset size: {len(dataset)}")

    sample = dataset[0]
    print(f"Sample shape: {sample.shape}")

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)
    for batch_idx, batch in enumerate(dataloader):
        print(f"Batch {batch_idx} shape: {batch.shape}")
        break
