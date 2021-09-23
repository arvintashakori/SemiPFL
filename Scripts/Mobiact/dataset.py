from torch.utils.data.dataset import Dataset
import random, itertools, torch
import numpy as np

class DatasetFromNPY(Dataset):
    def __init__(self, data, height, width, transforms):
        # height is the window size: from 100 - 1000
        self.height = height
        self.width = width
        self.transforms = transforms
        data = data[np.lexsort(data.T)]
        num_labels = np.unique(data[:, height * width], return_index=True)
        count_label = []
        for item in num_labels[0]:
            count_label.append(list(data[:, height * width]).count(item))
        min_num = sorted(count_label)
        self.data = np.zeros((len(min_num) * min_num[0], height * width + 1))
        for k in range(len(num_labels[0]) - 1):
            self.data[k * min_num[0]:(k + 1) * min_num[0], :] = np.array(
                random.sample(list(data[num_labels[1][k]:num_labels[1][k + 1], :]), min_num[0]))
        self.labels = self.data[:, height * width]
        self.data = np.delete(self.data, -1, axis=1)

    def __getitem__(self, index):
        single_adl_label = int(self.labels[index])
        # single_adl_label = char_label.index(single_adl_label)
        # 读取所有像素值，并将 1D array ([784]) reshape 成为 2D array ([28,28])
        # x=self.data.iloc[index][0:9]
        adl_as_np = np.asarray(self.data[index, :]).reshape(self.height, self.width).astype(float)
        # 把 numpy array 格式的图像转换成灰度 PIL image
        # img_as_img = Image.fromarray(ADL_as_np)
        # img_as_img = ADL_as_np.convert('L')
        # 将图像转换成 tensor should be in tensor form,
        if self.transforms is not None:
            adl_as_tensor = self.transforms(adl_as_np)
        else:
            adl_as_tensor = adl_as_np
            # 返回图像及其 label

        return adl_as_tensor, single_adl_label

    def __len__(self):
        return self.data.shape[0]

def assign_loaders(labels_list,trial_number,label_ratio,number_client,server_ID,batch_size,windowsize,width,transform,num_user):
    server_loaders=DatasetFromNPY(np.load( str(server_ID) + 'trail_' + str(trial_number) + '.npy', mmap_mode='r'), width, windowsize, transform)  # load test dataset
    num_user_list = np.delete(range(num_user), server_ID)
    num_user_list = random.sample(num_user_list,number_client)
    client_data=[]
    for user in num_user_list:
        file_name = str(user) + 'trail_' + str(trial_number) + '.npy'
        client_data.append(np.load(file_name, mmap_mode='r'))
    client_data=np.array(list(itertools.chain.from_iterable(client_data)))
    client_unlablled=client_data[0:int(client_data.shape[0]*label_ratio)]
    client_lablled=client_data[-int(client_data.shape[0]*(1-label_ratio)):-1]
    client_lablled_loaders = DatasetFromNPY(client_lablled, width,  windowsize, transform)
    client_unlablled_loaders = DatasetFromNPY(client_unlablled, width,  windowsize, transform)
    server_loaders = torch.utils.data.DataLoader(server_loaders, batch_size=batch_size, shuffle=True)
    client_lablled_loaders = torch.utils.data.DataLoader(client_lablled_loaders, batch_size=batch_size, shuffle=True)
    client_unlablled_loaders = torch.utils.data.DataLoader(client_unlablled_loaders, batch_size=batch_size, shuffle=True)

    return client_lablled_loaders,client_unlablled_loaders,  server_loaders
