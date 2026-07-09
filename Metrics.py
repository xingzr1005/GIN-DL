import numpy as np
from skimage.metrics import structural_similarity as ssim
import torch
import torch.nn.functional as F
import torch.nn as nn
# SSIM相关函数和类
def _fspecial_gauss_1d(size, sigma):
    coords = torch.arange(size).to(dtype=torch.float)
    coords -= size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g /= g.sum()
    return g.unsqueeze(0).unsqueeze(0)

def gaussian_filter(input, win):
    N, C, H, W = input.shape
    out = F.conv2d(input, win, stride=1, padding=0, groups=C)
    out = F.conv2d(out, win.transpose(2, 3), stride=1, padding=0, groups=C)
    return out

def _ssim(X, Y, data_range, win, size_average=True, K=(0.01, 0.03)):
    K1, K2 = K
    batch, channel, height, width = X.shape
    compensation = 1.0
    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2
    win = win.to(X.device, dtype=X.dtype)
    mu1 = gaussian_filter(X, win)
    mu2 = gaussian_filter(Y, win)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = compensation * (gaussian_filter(X * X, win) - mu1_sq)
    sigma2_sq = compensation * (gaussian_filter(Y * Y, win) - mu2_sq)
    sigma12 = compensation * (gaussian_filter(X * Y, win) - mu1_mu2)
    cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
    ssim_map = ((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)) * cs_map
    ssim_per_channel = torch.flatten(ssim_map, 2).mean(-1)
    cs = torch.flatten(cs_map, 2).mean(-1)
    return ssim_per_channel, cs



def calculate_ssim(y_pred: np.array, y_true: np.array):
    ssim_values = []
    for i in range(y_true.shape[0]):  # Iterate over the first dimension
        for j in range(y_true.shape[1]):  # Iterate over the second dimension
            # Extract 2D slices
            true_slice = y_true[i, j, :, :, 0]
            pred_slice = y_pred[i, j, :, :, 0]

            # Compute SSIM for the current slice
            slice_ssim = ssim(true_slice, pred_slice, data_range=pred_slice.max() - pred_slice.min())
            ssim_values.append(slice_ssim)

    # Compute and return the average SSIM
    return np.mean(ssim_values)


def evaluate(y_pred: np.array, y_true: np.array, precision=4):
    mse = MSE(y_pred, y_true)
    rmse = RMSE(y_pred, y_true)
    mae = MAE(y_pred, y_true)
    mape = MAPE(y_pred, y_true)
    pcc = PCC(y_pred, y_true)
    ssim_value = calculate_ssim(y_pred, y_true)
    CPC= common_part_of_commuters(y_pred, y_true)

    # Print statistics
    print('MSE:', round(mse, precision))
    print('RMSE:', round(rmse, precision))
    print('MAE:', round(mae, precision))
    print('MAPE:', round(mape * 100, precision), '%')
    print('PCC:', round(pcc, precision))
    print('SSIM:', round(ssim_value, precision))
    print('CPC:', round(CPC, precision))

    return mse, rmse, mae, mape, ssim_value, CPC


def MSE(y_pred: np.array, y_true: np.array):
    return np.mean(np.square(y_pred - y_true))

def RMSE(y_pred:np.array, y_true:np.array):
    return np.sqrt(MSE(y_pred, y_true))

def MAE(y_pred:np.array, y_true:np.array):
    return np.mean(np.abs(y_pred - y_true))

def MAPE(y_pred:np.array, y_true:np.array, epsilon=1e-0):       # avoid zero division
    return np.mean(np.abs(y_pred - y_true) / (y_true + epsilon))

def PCC(y_pred:np.array, y_true:np.array):      # Pearson Correlation Coefficient
    return np.corrcoef(y_pred.flatten(), y_true.flatten())[0,1]

def common_part_of_commuters(y_pred: np.array, y_true: np.array):
    """
    Calculate the Common Part of Commuters (CPC) metric between
    predicted and true flow matrices.

    Parameters:
    y_pred (np.array): Predicted flow matrix.
    y_true (np.array): Actual flow matrix.

    Returns:
    float: CPC metric.
    """
    intersection_sum = np.sum(np.minimum(y_pred, y_true))
    total_sum = np.sum(y_pred) + np.sum(y_true)

    return (2 * intersection_sum) / total_sum if total_sum != 0 else 0