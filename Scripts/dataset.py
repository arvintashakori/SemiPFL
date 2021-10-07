from torch.utils.data.dataset import Dataset
import random
import itertools
import torch
import numpy as np

torch.manual_seed(0)
np.random.seed(0)


class DatasetFromNPY(Dataset):
    def __init__(self, data, height, width, transforms):
        # height is the window size: 30
        self.height = height
        self.width = width
        self.transforms = transforms
        data = data[np.lexsort(data.T)]
        num_labels = np.unique(data[:, height * width], return_index=True)
        count_label = []
        for item in num_labels[0]:
            count_label.append(list(data[:, height * width]).count(item))
        min_num = sorted(count_label)
        count_label = np.append(num_labels[1], (len(data)))
        self.data = np.zeros((len(min_num) * min_num[0], height * width + 1))
        for k in range(len(num_labels[0])):
            self.data[k * min_num[0]:(k + 1) * min_num[0], :] = np.array(
                random.sample(list(data[count_label[k]:count_label[k + 1], :]), min_num[0]))
            # print('self.data', k * min_num[0], (k + 1) * min_num[0])
            # print('data', count_label[k], count_label[k + 1])
        self.labels = self.data[:, height * width]
        self.labels = np.array(list(map(lambda x: list(num_labels[0]).index(x), self.data[:, height * width])))
        self.data = np.delete(self.data, -1, axis=1)

    def __getitem__(self, index):
        single_adl_label = int(self.labels[index])
        adl_as_np = np.asarray(self.data[index, :]).reshape(
            self.height, self.width).astype(float)
        if self.transforms is not None:
            adl_as_tensor = self.transforms(adl_as_np)
        else:
            adl_as_tensor = adl_as_np
        return adl_as_tensor, single_adl_label

    def __len__(self):
        return self.data.shape[0]


def assign_loaders(address, trial_number, label_ratio, eval_ratio, server_ID, windowsize, width,
                   transform, num_user):
    server_data = []
    num_user_list = range(num_user)
    for server in server_ID:
        server_data.append(np.load(address + 'user' + str(server) + 'trail_'
                                   + str(trial_number) + '.npy', mmap_mode='r'))

    server_loaders = DatasetFromNPY(np.array(list(itertools.chain.from_iterable(server_data))),
                                        width, windowsize, transform)  # load test dataset
    num_user_list = np.delete(num_user_list, server_ID)
    # labels_list = np.unique(server_data[:, windowsize * width])

    # num_user_list = list(map(lambda x: np.delete(num_user_list, x), server_ID))
    # num_user_list = random.sample(num_user_list, number_client)
    client_dataset = []
    client_lablled_dataset = []
    eval_dataset = []
    for user in num_user_list:  # return a list of user data and shuffle each user's data before return
        file_name = address + 'user' + \
                    str(user) + 'trail_' + str(trial_number) + '.npy'
        client_data = np.load(file_name, mmap_mode='r')
        client_data = np.array(client_data)
        np.random.shuffle(client_data)
        eval_data = client_data[0:int(client_data.shape[0] * eval_ratio)]
        client_data = client_data[int(client_data.shape[0] * eval_ratio):-1]
        client_dataset.append(
            DatasetFromNPY(client_data, width, windowsize, transform))  # load 59 users' data into client_dataset
        # client_data = np.array(list(itertools.chain.from_iterable(client_data)))
        client_lablled = client_data[0:int(client_data.shape[0] * label_ratio)]
        # lablled_user_list = random.sample(num_user_list, int(client_data.shape[0] * (label_ratio))) #
        # client_lablled = random.sample(client_data,
        #                                int(client_data.shape[0] * (label_ratio)))  # return randomly labelled data
        client_lablled_dataset.append(DatasetFromNPY(
            client_lablled, width, windowsize, transform))
        eval_dataset.append(DatasetFromNPY(
            eval_data, width, windowsize, transform))
    # server_loaders = torch.utils.data.DataLoader(
    #     server_loaders, batch_size=batch_size, shuffle=True)
    return client_dataset, client_lablled_dataset, server_loaders, eval_dataset
