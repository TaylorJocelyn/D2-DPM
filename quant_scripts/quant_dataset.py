import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

class DiffusionInputDataset(Dataset):
    def __init__(self, data_path):
        data_list = torch.load(data_path, map_location='cpu') ## its a list of tuples of tensors
        self.xt_list = []
        self.t_list = []
        self.y_list = []
        ## datalist[i][0].shape (B,4,32,32), flat B dimension
        for i in range(len(data_list)):
            for b in range(len(data_list[i][0])):
                self.xt_list.append(data_list[i][0][b])
                self.t_list.append(data_list[i][1][b])
                self.y_list.append(data_list[i][2][b])

    def __len__(self):
        return len(self.xt_list)
    
    def __getitem__(self, idx):
        return self.xt_list[idx], self.t_list[idx], self.y_list[idx]

class lsunInputDataset(Dataset):
    def __init__(self, data_path):
        data_list = torch.load(data_path) ## its a list of tuples of tensors
        self.xt_list = []
        self.t_list = []
        # self.y_list = []
        ## datalist[i][0].shape (B,4,32,32), flat B dimension
        for i in range(len(data_list)):
            for b in range(len(data_list[i][0])):
                self.xt_list.append(data_list[i][0][b])
                self.t_list.append(data_list[i][1][b])
                # self.y_list.append(data_list[i][2][b]) ## its None

    def __len__(self):
        return len(self.xt_list)
    
    def __getitem__(self, idx):
        return self.xt_list[idx], self.t_list[idx]
    
def get_train_samples(train_loader, num_samples, dataset_type='imagenet'):
    image_data, t_data, y_data = [], [], []
    if dataset_type == 'imagenet':
        for (image, t, y) in train_loader:
            image_data.append(image)
            t_data.append(t)
            y_data.append(y)
            if len(image_data) >= num_samples:
                break
                
        return torch.cat(image_data, dim=0)[:num_samples], torch.cat(t_data, dim=0)[:num_samples], torch.cat(y_data, dim=0)[:num_samples]
    
    elif dataset_type == 'lsun':
        for (image, t) in train_loader:
            image_data.append(image)
            t_data.append(t)
            if len(image_data) >= num_samples:
                break
                
        return torch.cat(image_data, dim=0)[:num_samples], torch.cat(t_data, dim=0)[:num_samples]

def get_calibration_set(data_path, dataset_type='imagenet'):
    dataset = torch.load(data_path, map_location='cpu')
    # dataset = lsunInputDataset(data_path)
    data_loader = DataLoader(dataset=dataset, batch_size=8, shuffle=True)
    if dataset_type == 'imagenet':
        cali_images, cali_t, cali_y = get_train_samples(data_loader, num_samples=2048, dataset_type=dataset_type)
        return cali_images, cali_t, cali_y
    elif dataset_type == 'lsun':
        cali_images, cali_t = get_train_samples(data_loader, num_samples=2048, dataset_type=dataset_type)
        return cali_images, cali_t

def concat_calibration_subset(data_paths):
    dataset_list = []
    for path in data_paths:
        dataset = DiffusionInputDataset(path)
        dataset_list.append(dataset)
    
    d_0 = dataset_list[0]
    for i in range(1, len(dataset_list)):
        d_i = dataset_list[i]
        d_0.xt_list.extend(d_i.xt_list)
        d_0.t_list.extend(d_i.t_list)
        d_0.y_list.extend(d_i.y_list)

    return d_0

def concat_calibration_subset_lsun(data_paths):
    dataset_list = []
    for path in data_paths:
        dataset = lsunInputDataset(path)
        dataset_list.append(dataset)
    
    d_0 = dataset_list[0]
    for i in range(1, len(dataset_list)):
        d_i = dataset_list[i]
        d_0.xt_list.extend(d_i.xt_list)
        d_0.t_list.extend(d_i.t_list)

    return d_0

def concat_error(data_path1, data_path2):
    error1 = torch.load(data_path1, map_location='cpu')
    error2 = torch.load(data_path2, map_location='cpu')
    error1.extend(error2)

    return error1


if __name__ == '__main__':
    # dataset = concat_calibration_subset_lsun(['reproduce/lsun_bedroom_eta1.0_step200/data/image_input_bedroom_1.pth', 'reproduce/lsun_bedroom_eta1.0_step200/data/image_input_bedroom_2.pth', 'reproduce/lsun_bedroom_eta1.0_step200/data/image_input_bedroom_3.pth', 'reproduce/lsun_bedroom_eta1.0_step200/data/image_input_bedroom_4.pth'])
    # get_calibration_set('reproduce/scale1.5_eta1.0_step250/imagenet/w4a8/data/imagenet_input_1000classes.pth')
    dataset = concat_error('reproduce/lsun_bedroom_eta1.0_step200/w4a8/data_error_t_w4a8_eta1.0_step200_1.pth', 'reproduce/lsun_bedroom_eta1.0_step200/w4a8/data_error_t_w4a8_eta1.0_step200_2.pth')
    # dataset = concat_calibration_subset_lsun(['reproduce/lsun_bedroom_eta1.0_step200/data/image_input_bedroom_1.pth',
    #                                      'reproduce/lsun_bedroom_eta1.0_step200/data/image_input_bedroom_2.pth',
    #                                      'reproduce/lsun_bedroom_eta1.0_step200/data/image_input_bedroom_3.pth',
    #                                      'reproduce/lsun_bedroom_eta1.0_step200/data/image_input_bedroom_4.pth'])
    # dataset = DiffusionInputDataset('reproduce/scale1.5_eta0.0_step250/imagenet/w4a8/imagenet_input_1000classes.pth')
    # torch.save(dataset, 'reproduce/scale1.5_eta0.0_step250/imagenet/w4a8/imagenet_input_1000classes_dataset.pth')
    
    # dataset = lsunInputDataset('reproduce/lsun_church_eta0.0_step200/data/image_input_church.pth')
    torch.save(dataset, 'reproduce/lsun_bedroom_eta1.0_step200/w4a8/data_error_t_w4a8_eta1.0_step200.pth')
    
