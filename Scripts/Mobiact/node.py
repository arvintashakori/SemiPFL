
class Node:
    def __init__(
        self,
        number_of_users,
        labeled_ratio
    ):
        self.number_of_users = number_of_users
        self.labeled_ratio = labeled_ratio


    def _init_dataloaders(self):
        return

    def __len__(self):
        return self.number_of_users
