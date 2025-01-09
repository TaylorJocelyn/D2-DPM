import numpy as np
import torch
import sys
sys.path.append('.')
from quant_scripts.quant_dataset import DiffusionInputDataset

def resample_calibration(data_path):
    
    ddim_step = 250
    t_mean = 0.4
    t_std = 0.4
    num_samples = 1024
    t_i = np.random.normal(t_mean, t_std, num_samples) * (ddim_step-1)
    t_i = np.clip(np.round(t_i), 0, ddim_step-1)
    
    dataset = torch.load(data_path, map_location='cpu')
    # dataset = DiffusionInputDataset(data_path)
    x = dataset.xt_list
    t = dataset.t_list 
    y = dataset.y_list

    st = np.zeros((1000, 8, ddim_step))
    
    calib_xt, calib_y, calib_t = [], [], []
    for i in range(t_i.shape[0]):
        ct = int(t_i[i])
        
        while True:
            c = np.random.randint(0, 1000)
            idx = np.random.randint(0, 8)

            if st[c][idx][ct] == 0:
                st[c][idx][ct] = 1
                break
        
        j = ddim_step * 8 * c + (ddim_step-1-ct) * 8 + idx
        calib_xt.append(x[j].unsqueeze(0))
        calib_y.append(y[j].unsqueeze(0))
        calib_t.append(t[j].unsqueeze(0))

    cali_xt, cali_t, cali_y = torch.cat(calib_xt, dim=0), torch.cat(calib_t, dim=0), torch.cat(calib_y, dim=0)
    return cali_xt, cali_t, cali_y
    





if __name__ == '__main__':
    resample_calibration('scale3.0_eta0.0_step20/imagenet/w4a8/imagenet_input_1000classes.pth')