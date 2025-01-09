import numpy as np
import torch

n_bits_w = 4
n_bits_a = 8

scale = 3.0
ddim_eta = 0.0
ddim_steps = 20

def tensor_wise_gaussian_model():
    data_error_t_list = torch.load('scale{}_eta{}_step{}/imagenet/w{}a{}/data_error_t_w{}a{}_scale{}_eta{}_step{}.pth'.format(scale, ddim_eta, ddim_steps, n_bits_w, n_bits_a, n_bits_w, n_bits_a, scale, ddim_eta, ddim_steps), map_location='cpu')  ## replace error file here
    data_list = []
    error_list = [] 
    t_list = []
    quant_list = []
    for i in range(len(data_error_t_list)):
        for j in range(len(data_error_t_list[i][0])):
            data_list.append(data_error_t_list[i][0][j])
            error_list.append(torch.pow(data_error_t_list[i][1][j],1))
            t_list.append(data_error_t_list[i][2][j])
            quant_list.append(data_error_t_list[i][3][j])

    data_tensor = torch.stack(data_list) 
    error_tensor = torch.stack(error_list)
    t_tensor = torch.stack(t_list)
    quant_tensor = torch.stack(quant_list)

    t_data_dict, t_error_dict, t_quant_dict = {}, {}, {}

    for i in range(len(t_tensor)):
        int_t = t_tensor[i].item()
        if int_t not in t_data_dict.keys():
            t_data_dict[int_t] = data_tensor[i]
            t_error_dict[int_t] = error_tensor[i]
            t_quant_dict[int_t] = quant_tensor[i]
        else:
            t_data_dict[int_t] = torch.cat([t_data_dict[int_t], data_tensor[i]], dim=0)
            t_error_dict[int_t] = torch.cat([t_error_dict[int_t], error_tensor[i]], dim=0)
            t_quant_dict[int_t] = torch.cat([t_quant_dict[int_t], quant_tensor[i]], dim=0)
    
    # quant_mu_dict = {}
    # quant_sigma_dict = {}
    mu_dict = {}
    cov_dict = {}
    for k in t_data_dict.keys():
        flatten_data = t_data_dict[k].flatten()
        flatten_error = t_error_dict[k].flatten()
        flatten_quant = t_quant_dict[k].flatten()

        fp_out, error, quant = flatten_data.numpy(), flatten_error.numpy(), flatten_quant.numpy()

        # Estimate quant_error's gaussian-1d parameters 
        # First remove outliers
        mean = np.mean(error)
        std = np.std(error)

        threshold = 4

        outliers = (np.abs(error - mean) > threshold * std)
        error = error[~outliers]
        quant = quant[~outliers]

        quant_err = np.vstack((quant, error))
        mean = np.mean(quant_err, axis=1)
        covariance = np.cov(quant_err)

        # n = len(quant)
        # covariance[0][0] = covariance[0][0] * n / (n-1) 
        # covariance[1][1] = covariance[1][1] * n / (n-1)

        # quant-error
        mu_dict[k] = mean
        cov_dict[k] = covariance

        # quant_mu_dict[k] = torch.mean(t_quant_dict[k], dim=0).numpy()
        # quant_sigma_dict[k] = torch.pow(torch.std(t_quant_dict[k], dim=0), 2).numpy()

    np.save("scale{}_eta{}_step{}/imagenet/w{}a{}/gaussian_correct/mu_dict.npy".format(scale, ddim_eta, ddim_steps, n_bits_w, n_bits_a), mu_dict, allow_pickle=True)
    np.save("scale{}_eta{}_step{}/imagenet/w{}a{}/gaussian_correct/cov_dict.npy".format(scale, ddim_eta, ddim_steps, n_bits_w, n_bits_a), cov_dict, allow_pickle=True)

def tensor_wise_integeral_gaussian_model():
    data_error_t_list = torch.load('scale{}_eta{}_step{}/imagenet/w{}a{}/data_error_t_w{}a{}_scale{}_eta{}_step{}.pth'.format(scale, ddim_eta, ddim_steps, n_bits_w, n_bits_a, n_bits_w, n_bits_a, scale, ddim_eta, ddim_steps), map_location='cpu')  ## replace error file here
    data_list = []
    error_list = [] 
    t_list = []
    quant_list = []
    for i in range(len(data_error_t_list)):
        for j in range(len(data_error_t_list[i][0])):
            data_list.append(data_error_t_list[i][0][j])
            error_list.append(torch.pow(data_error_t_list[i][1][j],1))
            t_list.append(data_error_t_list[i][2][j])
            quant_list.append(data_error_t_list[i][3][j])

    data_tensor = torch.stack(data_list) 
    error_tensor = torch.stack(error_list)
    t_tensor = torch.stack(t_list)
    quant_tensor = torch.stack(quant_list)

    t_data_dict, t_error_dict, t_quant_dict = {}, {}, {}

    for i in range(len(t_tensor)):
        int_t = t_tensor[i].item()
        if int_t not in t_data_dict.keys():
            t_data_dict[int_t] = data_tensor[i]
            t_error_dict[int_t] = error_tensor[i]
            t_quant_dict[int_t] = quant_tensor[i]
        else:
            t_data_dict[int_t] = torch.cat([t_data_dict[int_t], data_tensor[i]], dim=0)
            t_error_dict[int_t] = torch.cat([t_error_dict[int_t], error_tensor[i]], dim=0)
            t_quant_dict[int_t] = torch.cat([t_quant_dict[int_t], quant_tensor[i]], dim=0)
    
    # quant_mu_dict = {}
    # quant_sigma_dict = {}
    mu_dict = {}
    cov_dict = {}
    for k in t_data_dict.keys():
        flatten_data = t_data_dict[k].flatten()
        flatten_error = t_error_dict[k].flatten()
        flatten_quant = t_quant_dict[k].flatten()

        fp_out, error, quant = flatten_data.numpy(), flatten_error.numpy(), flatten_quant.numpy()

        # Estimate quant_error's gaussian-1d parameters 
        # First remove outliers
        mean = np.mean(error)
        std = np.std(error)

        threshold = 4

        outliers = (np.abs(error - mean) > threshold * std)
        error = error[~outliers]
        quant = quant[~outliers]

        quant_err = np.vstack((quant, error))
        mean = np.mean(quant_err, axis=1)
        covariance = np.cov(quant_err)

        # n = len(quant)
        # covariance[0][0] = covariance[0][0] * n / (n-1) 
        # covariance[1][1] = covariance[1][1] * n / (n-1)

        # quant-error
        mu_dict[k] = mean
        cov_dict[k] = [covariance]

        # fp32-error
        fp_out = fp_out[~outliers]
        fp_err = np.vstack((fp_out, error))
        covariance_fp_err = np.cov(fp_err)
        cov_dict[k].append(covariance_fp_err[0][1])

        # quant_mu_dict[k] = torch.mean(t_quant_dict[k], dim=0).numpy()
        # quant_sigma_dict[k] = torch.pow(torch.std(t_quant_dict[k], dim=0), 2).numpy()

    np.save("scale{}_eta{}_step{}/imagenet/w{}a{}/gaussian_correct/mu_dict_new.npy".format(scale, ddim_eta, ddim_steps, n_bits_w, n_bits_a), mu_dict, allow_pickle=True)
    np.save("scale{}_eta{}_step{}/imagenet/w{}a{}/gaussian_correct/cov_dict_new.npy".format(scale, ddim_eta, ddim_steps, n_bits_w, n_bits_a), cov_dict, allow_pickle=True)
   
def channel_wise_gaussian_model():
    data_error_t_list = torch.load('scale{}_eta{}_step{}/imagenet/w{}a{}/data_error_t_w{}a{}_scale{}_eta{}_step{}.pth'.format(scale, ddim_eta, ddim_steps, n_bits_w, n_bits_a, n_bits_w, n_bits_a, scale, ddim_eta, ddim_steps), map_location='cpu')  ## replace error file here
    data_list = []
    error_list = [] 
    t_list = []
    quant_list = []
    for i in range(len(data_error_t_list)):
        for j in range(len(data_error_t_list[i][0])):
            data_list.append(data_error_t_list[i][0][j])
            error_list.append(torch.pow(data_error_t_list[i][1][j],1))
            t_list.append(data_error_t_list[i][2][j])
            quant_list.append(data_error_t_list[i][3][j])

    data_tensor = torch.stack(data_list) 
    error_tensor = torch.stack(error_list)
    t_tensor = torch.stack(t_list)
    quant_tensor = torch.stack(quant_list)

    t_data_dict, t_error_dict, t_quant_dict = [{}, {}, {}], [{}, {}, {}], [{}, {}, {}]

    for i in range(len(t_tensor)): # shape(20480) 20*1024
        int_t = t_tensor[i].item()
        for j in range(3):
            if int_t not in t_data_dict[j].keys():
                t_data_dict[j][int_t] = torch.unsqueeze(data_tensor[i][j], dim=0)
                t_error_dict[j][int_t] = torch.unsqueeze(error_tensor[i][j], dim=0)
                t_quant_dict[j][int_t] = torch.unsqueeze(quant_tensor[i][j], dim=0)
            else:
                t_data_dict[j][int_t] = torch.cat([t_data_dict[j][int_t], torch.unsqueeze(data_tensor[i][j], dim=0)], dim=0)
                t_error_dict[j][int_t] = torch.cat([t_error_dict[j][int_t], torch.unsqueeze(error_tensor[i][j], dim=0)], dim=0)
                t_quant_dict[j][int_t] = torch.cat([t_quant_dict[j][int_t], torch.unsqueeze(quant_tensor[i][j], dim=0)], dim=0)
    
    mu_dict = {}
    cov_dict = {}
    
    for channel in range(3):
        for k in t_data_dict[channel].keys():
            flatten_data = t_data_dict[channel][k].flatten()
            flatten_error = t_error_dict[channel][k].flatten()
            flatten_quant = t_quant_dict[channel][k].flatten()

            data, error, quant = flatten_data.numpy(), flatten_error.numpy(), flatten_quant.numpy()

            # Estimate quant_error's gaussian-1d parameters 
            # First remove outliers
            mean = np.mean(error)
            std = np.std(error)

            # if k <= 50:
            #     threshold = 12
            # elif k <= 480:
            #     threshold = 8
            # elif k <=1000:
            #     threshold = 4

            threshold = 4

            outliers = (np.abs(error - mean) > threshold * std)
            error = error[~outliers]
            quant = quant[~outliers]

            data_point = np.vstack((quant, error))
            data_mean = np.mean(data_point, axis=1) # shape(2, )
            covariance = np.cov(data_point) # shape(2, 2)

            # n = len(quant)
            # covariance[0][0] = covariance[0][0] * n / (n-1) 
            # covariance[1][1] = covariance[1][1] * n / (n-1)

            # mu_dict[channel][k] = data_mean
            # cov_dict[channel][k] = covariance

            if k not in mu_dict:
                mu_dict[k] = torch.unsqueeze(torch.tensor(data_mean), dim=1)
            else:
                mu_dict[k] = torch.cat([mu_dict[k], torch.unsqueeze(torch.tensor(data_mean), dim=1)], dim=1)

            if k not in cov_dict:
                cov_dict[k] = torch.unsqueeze(torch.tensor(covariance), dim=2)
            else:
                cov_dict[k] = torch.cat([cov_dict[k], torch.unsqueeze(torch.tensor(covariance), dim=2)], dim=2)


    np.save("scale{}_eta{}_step{}/imagenet/w{}a{}/channel_wise_gaussian_correct/mu_dict.npy".format(scale, ddim_eta, ddim_steps, n_bits_w, n_bits_a), mu_dict, allow_pickle=True)
    np.save("scale{}_eta{}_step{}/imagenet/w{}a{}/channel_wise_gaussian_correct/cov_dict.npy".format(scale, ddim_eta, ddim_steps, n_bits_w, n_bits_a), cov_dict, allow_pickle=True)

def load_gaussian_params():

    mu = np.load('scale3.0_eta0.0_step20/imagenet/w{}a{}/channel_wise_gaussian_correct/mu_dict_4.npy'.format(n_bits_w, n_bits_a), allow_pickle=True).item()
    cov = np.load('scale3.0_eta0.0_step20/imagenet/w{}a{}/channel_wise_gaussian_correct/cov_dict_4.npy'.format(n_bits_w, n_bits_a), allow_pickle=True).item()
    x = mu

if __name__ == '__main__':
    # tensor_wise_gaussian_model()
    channel_wise_gaussian_model()
