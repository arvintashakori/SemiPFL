import torch
from dataset import assign_loaders
from torchvision import transforms

torch.manual_seed(0)


class BaseNodes:
    def __init__(
            self,
            address,
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
        self.trial_number = trial_number
        self.label_ratio = label_ratio
        self.number_client = number_client
        self.server_ID = server_ID
        self.batch_size = batch_size
        self.window_size = window_size
        self.width = width
        self.transform = transform
        self.num_user = num_user
        self.address=address

        self.client_unlablled_loaders, self.client_lablled_loaders, self.server_loaders = None, None, None
        self._init_dataloaders()

    def _init_dataloaders(self):
        self.client_loaders, self.server_loaders, self.labels_list = assign_loaders(
            self.address,
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


transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
# address='Scripts/Mobiact/data'
address='./data'
data = BaseNodes(
    address,   # directory path
    trial_number=0,
    label_ratio=0.8,
    number_client=10,  # how many users is distributed to client training
    server_ID=1,
    batch_size=128,
    window_size=30,
    width=9,
    transform=transform,
    num_user=59) # all users we have
