import torch
from dataset import assign_loaders

torch.manual_seed(0)


class BaseNodes:
    def __init__(
            self,
            labels_list,
            trial_number,
            label_ratio,
            number_client,
            server_ID,
            batch_size,
            window_size,
            width,
            transform,
            num_user
    ):
        self.labels_list = labels_list
        self.trial_number = trial_number
        self.label_ratio = label_ratio
        self.number_client = number_client
        self.server_ID = server_ID
        self.batch_size = batch_size
        self.window_size = window_size
        self.width = width
        self.transform = transform
        self.num_user = num_user

        self.client_unlablled_loaders, self.client_lablled_loaders, self.server_loaders = None, None, None
        self._init_dataloaders()

    def _init_dataloaders(self):
        self.client_loaders, self.server_loaders = assign_loaders(
            self.labels_list,
            self.trial_number,
            self.label_ratio,
            self.number_client,
            self.server_ID,
            self.batch_size,
            self.window_size,
            self.width,
            self.transform,
            self.num_user
        )

    def __len__(self):
        return self.number_client
