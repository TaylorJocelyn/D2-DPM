import numpy as np
from sklearn.covariance import LedoitWolf, OAS
import torch
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from scipy.stats import norm
from scipy.stats import multivariate_normal
from scipy.optimize import curve_fit
from scipy.interpolate import BSpline, make_interp_spline, splrep, splev, UnivariateSpline, InterpolatedUnivariateSpline
from scipy.stats import multivariate_normal
import sys
sys.path.append('.')
from quant_scripts.gaussian_2d import plot_guassian_2d 
# from cov_shrinkage import cov1Para, cov2Para, covCor, covDiag, covMarket, GIS, LIS, QIS

n_bits_w = 4
n_bits_a = 8

def plot_estimate_1d_gaussian_channel_wise(type):
    data_error_t_list = torch.load('reproduce/scale3.0_eta0.0_step20/imagenet/w4a8/data_error_t_w4a8_scale3.0_eta0.0_step20.pth', map_location='cpu')  ## replace error file here
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

    data_tensor = torch.stack(data_list) # shape(20480, 3, 64, 64)
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
    
    for channel in range(3):
        step = 0
        img_num = 0
        
        empirical_sigma = {}
        kde_sigma = {}
        
        fig, axs = plt.subplots(4, 5, figsize=(40, 50))

        step = 0
        img_num = 0
        for k in t_data_dict[channel].keys():
            flatten_data = t_data_dict[channel][k].flatten()
            flatten_error = t_error_dict[channel][k].flatten()
            flatten_quant = t_quant_dict[channel][k].flatten()

            i = step // 5
            j = step % 5 
            data, error, quant = flatten_data.numpy(), flatten_error.numpy(), flatten_quant.numpy()

            # variance shrinkage estimation 

            # mean = np.mean(error)
            # std = np.std(error)
            # threshold = 4
            # outliers = (np.abs(error - mean) > threshold * std)
            # error = error[~outliers]
            # quant = quant[~outliers]

            data_point = np.vstack((quant, error)) # shape(2, n_sample)

            df = pd.DataFrame(data_point.T)
            df = df.T.reset_index().T.reset_index(drop=True)
            df = df.astype(float)
            # lw = LedoitWolf().fit(data_point.unsqueeze(dim=0))
            # oas = OAS().fit(data_point.T)

            # shrinkage = get_shrinkage(k)
            # cov1para_cov = cov1Para.cov1Para(df, pred_shrinkage=shrinkage)
            # cov1para_cov = cov1Para.cov1Para(df, pred_shrinkage=0.5)

            # print(f'--------------- t = {k} -------------')
            # print('shrinkage: {}'.format(shrinkage))
            # print(f'quant sigma: {cov1para_cov[0][0]: .8f}      err sigma: {cov1para_cov[1][1]: .8f}')

            # cov2para_cov = cov2Para.cov2Para(df)
            # cov_cor_cov = covCor.covCor(df)
            # cov_diag_cov = covDiag.covDiag(df)
            # cov_market_cov = covMarket.covMarket(df, pred_shrinkage=0.3)
            # gis_cov = GIS.GIS(df)
            # lis_cov = LIS.LIS(df)
            # qis_cov = QIS.QIS(df) 


            # data_mean = np.mean(data_point, axis=1)
            # covariance = np.cov(data_point)
            
            if type == 'quant':
                # 计算并绘制概率密度函数 (PDF)
                density = gaussian_kde(quant)
                xs = np.linspace(min(quant), max(quant), 200)
                axs[i, j].plot(xs, density(xs), linewidth=2 ,label='Gaussian KDE emstimate PDF')

                # unbiased estimation
                mu = np.mean(quant)
                n = len(quant)
                sigma = np.std(quant) * np.sqrt(n / (n-1))
                pdf = norm.pdf(xs, mu, sigma)
                axs[i, j].plot(xs, pdf, linestyle='--', linewidth=2, color='darkorange', label='Empirical estimate PDF')

                empirical_sigma[k] = sigma

                # # vs-estimate
                # def normal_pdf(x, sigma):
                #     return 1/(sigma * np.sqrt(2*np.pi)) * np.exp(-(x-mu)**2/(2*sigma**2))
                
                # popt, _ = curve_fit(normal_pdf, xs, density(xs))
                # vs_sigma = popt[0]
                # vs_pdf = norm.pdf(xs, mu, vs_sigma)
                
                # kde_sigma[k] = vs_sigma

                # axs[i, j].plot(xs, vs_pdf, linestyle='--', linewidth=2, color='navy', label='Variance shrinkage estimate PDF')
                # print(f'vs sigma: {vs_sigma: .8f}')

                # # Ledoit-Wolf variance shrinkage estimation
                # lw_sigma = np.sqrt(lw.covariance_[0][0])
                # lw_pdf = norm.pdf(xs, mu, lw_sigma)
                # axs[i, j].plot(xs, lw_pdf, linestyle='--', linewidth=2, color='navy', label='Ledoit-Wolf variance estimate PDF')

                # # OAS variance shrinkage estimation
                # oas_sigma = np.sqrt(oas.covariance_[0][0])
                # oas_pdf = norm.pdf(xs, mu, oas_sigma)
                # axs[i, j].plot(xs, oas_pdf, linestyle='--', linewidth=2, color='darkorange', label='OAS estimate PDF')

                # cov1para variance shrinkage estimation
                # cov1para_sigma = np.sqrt(cov1para_cov.values[0][0])
                # cov1para_pdf = norm.pdf(xs, mu, cov1para_sigma)
                # axs[i, j].plot(xs, cov1para_pdf, linestyle='--', linewidth=2, color='navy', label='Ledoit-Wolf variance shrinkage estimate PDF')

                # # cov2para variance shrinkage estimation
                # cov2para_sigma = np.sqrt(cov2para_cov.values[0][0])
                # cov2para_pdf = norm.pdf(xs, mu, cov2para_sigma)
                # axs[i, j].plot(xs, cov2para_pdf, linestyle='--', linewidth=2, color='darkorange', label='cov2Para estimate PDF')

                #  # covCor variance shrinkage estimation
                # cov_cor_sigma = np.sqrt(cov_cor_cov.values[0][0])
                # cov_cor_pdf = norm.pdf(xs, mu, cov_cor_sigma)
                # axs[i, j].plot(xs, cov_cor_pdf, linestyle='--', linewidth=2, label='covCor estimate PDF')

                #  # covDiag variance shrinkage estimation
                # cov_diag_sigma = np.sqrt(cov_diag_cov.values[0][0])
                # cov_diag_pdf = norm.pdf(xs, mu, cov_diag_sigma)
                # axs[i, j].plot(xs, cov_diag_pdf, linestyle='--', linewidth=2, label='covDiag estimate PDF')

                #  # covMarket variance shrinkage estimation
                # cov_market_sigma = np.sqrt(cov_market_cov.values[0][0])
                # cov_market_pdf = norm.pdf(xs, mu, cov_market_sigma)
                # axs[i, j].plot(xs, cov_market_pdf, linestyle='--', linewidth=2, label='covMarket estimate PDF')

                #  # GIS variance shrinkage estimation
                # gis_sigma = np.sqrt(gis_cov.values[0][0])
                # gis_pdf = norm.pdf(xs, mu, gis_sigma)
                # axs[i, j].plot(xs, gis_pdf, linestyle='--', linewidth=2, label='GIS estimate PDF')

                #  # LIS variance shrinkage estimation
                # lis_sigma = np.sqrt(lis_cov.values[0][0])
                # lis_pdf = norm.pdf(xs, mu, lis_sigma)
                # axs[i, j].plot(xs, lis_pdf, linestyle='--', linewidth=2, label='LIS estimate PDF')

                #  # QIS variance shrinkage estimation
                # qis_sigma = np.sqrt(qis_cov.values[0][0])
                # qis_pdf = norm.pdf(xs, mu, qis_sigma)
                # axs[i, j].plot(xs, qis_pdf, linestyle='--', linewidth=2, label='QIS estimate PDF')

                axs[i, j].hist(quant, bins=30, density=True, color='r', alpha=0.5, label='Frequency Histogram')
                axs[i, j].set_title(f"Frequency Histogram and Probability Density Curve \n t = {k}")
                axs[i, j].set_xlabel('Quantized $\epsilon_t$')
                axs[i, j].set_ylabel('Frequency / Probability Density')
                axs[i, j].grid(True)
                axs[i, j].legend()
            elif type == 'error':
                # mean = np.mean(error)
                # std = np.std(error)
                # threshold = 4
                # outliers = (np.abs(error - mean) > threshold * std)
                # error = error[~outliers]
                xs = np.linspace(min(error), max(error), 200)

                # 计算并绘制概率密度函数 (PDF)
                density = gaussian_kde(error)
                axs[i, j].plot(xs, density(xs), linewidth=2, label='Gaussian KDE emstimate PDF')
                
                # unbiased estimation
                mu = np.mean(error)
                n = len(error)
                sigma = np.std(error) * np.sqrt(n / (n-1))
                pdf = norm.pdf(xs, mu, sigma)
                axs[i, j].plot(xs, pdf, linestyle='--', linewidth=2, color='darkorange', label='Empirical estimate PDF')
                print(f'unbiased err sigma: {sigma: .8f}')

                empirical_sigma[k] = sigma

                # # vs-estimate
                # def normal_pdf(x, sigma):
                #     return 1/(sigma * np.sqrt(2*np.pi)) * np.exp(-(x-mu)**2/(2*sigma**2))
                
                # popt, _ = curve_fit(normal_pdf, xs, density(xs))
                # vs_sigma = popt[0]
                # vs_pdf = norm.pdf(xs, mu, vs_sigma)
                
                # kde_sigma[k] = vs_sigma

                # axs[i, j].plot(xs, vs_pdf, linestyle='--', linewidth=2, color='navy', label='Variance shrinkage estimate PDF')
                # print(f'vs sigma: {vs_sigma: .8f}')
                
                # # cov1para variance shrinkage estimati
                # cov1para_sigma = np.sqrt(cov1para_cov.values[1][1])
                # cov1para_pdf = norm.pdf(xs, mu, cov1para_sigma)
                # axs[i, j].plot(xs, cov1para_pdf, linestyle='--', linewidth=2, color='navy', label='Ledoit-Wolf variance shrinkage estimate PDF')
                # print(f'unbiased vs lw sigame: {sigma: .8f} - {cov1para_sigma: .8f}')
                
                axs[i, j].hist(error, bins=50, density=True, alpha=0.5, label='Frequency Histogram')
                axs[i, j].set_title(f"Frequency Histogram and Probability Density Curve \n t = {k}")
                axs[i, j].set_xlabel('Quantized error')
                axs[i, j].set_ylabel('Frequency / Probability Density')
                axs[i, j].grid(True)
                axs[i, j].legend()
            else:
                # 计算并绘制概率密度函数 (PDF)
                density = gaussian_kde(data)
                xs = np.linspace(min(data), max(data), 200)
                axs[i, j].plot(xs, density(xs), linewidth=2 ,label='Probability Density Curve')
                mu = np.mean(data)
                n = len(data)
                sigma = np.std(data) * np.sqrt(n / (n-1))
                pdf = norm.pdf(xs, mu, sigma)
                axs[i, j].plot(xs, pdf, linestyle='--', linewidth=2, color='black', label='Estimate PDF')
                axs[i, j].hist(data, bins=30, density=True, color='b', alpha=0.5, label='Frequency Histogram')
                axs[i, j].set_title(f"Frequency Histogram and Probability Density Curve \n t = {k}")
                axs[i, j].set_xlabel('FP32 output')
                axs[i, j].set_ylabel('Frequency / Probability Density')
                axs[i, j].grid(True)
                axs[i, j].legend()

            if step == 19:
                plt.tight_layout()
                if type == 'quant':
                    plt.savefig('observed_results/channel_wise_gaussian/w4a8/quant_out_hist_c{}.jpg'.format(channel), dpi=300)
                elif type == 'error':
                    plt.savefig('observed_results/channel_wise_gaussian/w4a8/quant_error_hist_c{}.jpg'.format(channel), dpi=300)
                else:
                    plt.savefig('observation_result/variance_shrinkage/fp32_out_hist_estimate_pdf_t_w{}a{}_step200_{}.jpg'.format(n_bits_w, n_bits_a, img_num), dpi=300)
                fig, axs = plt.subplots(4, 5, figsize=(40, 50))
                step = 0
                img_num += 1
            else:
                step = (step + 1) % 20
        
        # np.save("reproduce/scale3.0_eta0.0_step20/imagenet/w4a4/variance_shrinkage/linear_invariant/{}_empirical_sigma.npy".format(type), empirical_sigma, allow_pickle=True)
        # np.save("reproduce/scale3.0_eta0.0_step20/imagenet/w4a4/variance_shrinkage/linear_invariant/{}_vs_sigma.npy".format(type), kde_sigma, allow_pickle=True)

def plot_scatter(type):
    data_error_t_list = torch.load('reproduce/scale3.0_eta0.0_step20/imagenet/w4a8/data_error_t_w4a8_scale3.0_eta0.0_step20.pth', map_location='cpu')  ## replace error file here
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
    
    step = 0
    img_num = 0
    
    fig, axs = plt.subplots(4, 5, figsize=(50, 40))

    step = 0
    img_num = 0

    for channel in range(3):
        for k in t_data_dict[channel].keys():
            flatten_data = t_data_dict[channel][k].flatten()
            flatten_error = t_error_dict[channel][k].flatten()
            flatten_quant = t_quant_dict[channel][k].flatten()

            i = step // 5
            j = step % 5 
            data, error, quant = flatten_data.numpy()[:40960], flatten_error.numpy()[:40960], flatten_quant.numpy()[:40960]
            
            data_point = np.vstack((quant, error))
            mean = np.mean(data_point, axis=1)[:, np.newaxis]
            std = np.std(data_point, axis=1)[:, np.newaxis]

            if k <= 50:
                threshold = 12
            elif k <= 480:
                threshold = 8
            elif k <=1000:
                threshold = 4
            
            outliers = ((np.abs(data_point - mean) > threshold * std).any(axis=0))
    
            quant = data_point[0, ~outliers]
            error = data_point[1, ~outliers]

            if type == 'quant_error':
                axs[i, j].scatter(quant, error, label='Original Data', alpha=0.5)
                axs[i, j].set_title(f"Quantized $\epsilon_t$ - Quantized error scatter \n t = {k}")
                axs[i, j].set_xlabel('Quantized $\epsilon_t$')
                axs[i, j].set_ylabel('Quantized error')
                # axs[i, j].set_ylim([-0.6, 0.6])
                # axs[i, j].set_xlim([-4, 4])

                mean_ = np.mean(data_point, axis=1)
                cov = np.cov(quant, error)
                model = multivariate_normal(mean_, cov)
                samples = model.rvs(40960)
                axs[i, j].scatter(samples[:, 0], samples[:, 1], color='pink', alpha=0.5, label='Samples')
                axs[i, j].legend()
            else:
                axs[i, j].scatter(data, error)
                axs[i, j].set_title(f"FP32 $\epsilon_t$ - Quantized error scatter \n t = {k}")
                axs[i, j].set_xlabel('FP32 $\epsilon_t$')
                axs[i, j].set_ylabel('Quantized error')
                # axs[i, j].set_ylim([-0.6, 0.6])
                # axs[i, j].set_xlim([-4, 4])

            if step == 19:
                plt.tight_layout()
                if type == 'quant_error':
                    plt.savefig('observed_results/channel_wise_gaussian/w4a8/quant_error_scatter_c{}_final.jpg'.format(channel))
                else:
                    plt.savefig('observed_results/channel_wise_gaussian/w4a8/fp32_error_scatter_c{}__final.jpg'.format(channel))
                fig, axs = plt.subplots(4, 5, figsize=(50, 40))
                step = 0
                img_num += 1
            else:
                step = (step + 1) % 20

if __name__ == '__main__':
    # plot_estimate_1d_gaussian_channel_wise('error')
    plot_scatter('quant_error')
