import torch
from dataset import assign_loaders
torch.manual_seed(0)


class Clients:
    def __init__(
            self,
            address,
            trial_number,
            label_ratio,
            server_ID,
            #batch_size,
            window_size,
            width,
            transform,
            num_user  # all users we have
    ):
        self.trial_number = trial_number
        self.label_ratio = label_ratio
        #self.number_client = number_client
        self.server_ID = server_ID
        #self.batch_size = batch_size
        self.window_size = window_size
        self.width = width
        self.transform = transform
        self.num_user = num_user
        self.address = address

        self.client_loaders, self.client_labeled_loaders, self.server_loaders = None, None, None
        self._init_dataloaders()

    def _init_dataloaders(self):
        self.client_labeled_loaders, self.client_loaders, self.server_loaders, self.labels_list = assign_loaders(
            self.address,
            self.trial_number,
            self.label_ratio,
            #self.number_client,
            self.server_ID,
            #self.batch_size,
            self.window_size,
            self.width,
            self.transform,
            self.num_user
        )

    def __len__(self):
        return self.num_user
