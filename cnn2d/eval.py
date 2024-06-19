import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from log import log


class MeowEvaluator(object):
    def __init__(self, cacheDir):
        self.cacheDir = cacheDir
        self.predictionCol = "forecast"
        self.ycol = "fret12"

    def eval(self, ydf):
        ydf = ydf.replace([np.inf, -np.inf], np.nan).fillna(0)
        pcor = ydf[[self.predictionCol, self.ycol]].corr().to_numpy()[0, 1]
        r2 = 1 - ((ydf[self.predictionCol] - ydf[self.ycol]) ** 2).sum() / ydf[self.ycol].var() / ydf.shape[0]
        mse = ((ydf[self.predictionCol] - ydf[self.ycol]) ** 2).sum() / ydf.shape[0]
        log.inf("Meow evaluation summary: Pearson correlation={:.8f}, R2={:.8f}, MSE={:.8f}".format(pcor, r2, mse))
        return pcor, r2, mse


def show_(ydf):
    # 直接在 ydf 上操作，不需要使用 .xs
    # 替换无穷值并填充NaN为0
    ydf = ydf.replace([np.inf, -np.inf], np.nan).fillna(0)

    # 绘图代码...
    plt.figure(figsize=(10, 5))
    plt.plot(ydf.index, ydf["fret12"], label='Actual', color='blue')
    plt.plot(ydf.index, ydf["forecast"], label='Forecast', color='red')
    plt.legend()
    plt.title('Forecast vs Actual')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.grid(True)
    plt.show()


import torch


def pearson_loss(output, target):
    mean_output = output.mean()
    mean_target = target.mean()
    output_centered = output - mean_output
    target_centered = target - mean_target
    numerator = (output_centered * target_centered).sum()
    denominator = torch.sqrt((output_centered ** 2).sum()) * torch.sqrt((target_centered ** 2).sum())
    pearson_correlation = numerator / denominator
    return 1 - pearson_correlation


def r2_loss(output, target):
    total_variance = torch.var(target)
    unexplained_variance = torch.var(target - output)
    r2 = 1 - (unexplained_variance / total_variance)
    return 1 - r2


def mse_loss(output, target):
    return torch.mean((output - target) ** 2)


def composite_loss(output, target, alpha=0, beta=0, gamma=1):
    # 计算每个单独的损失
    pearson = pearson_loss(output, target)
    r2 = r2_loss(output, target)
    mse = mse_loss(output, target)

    # 标准化损失（可选）
    # 这里可以添加代码来标准化每个损失值

    # 加权组合损失

    return alpha * (1-pearson) + beta * (1-r2) + gamma * mse
# 使用复合损失函数
# 在你的模型训练循环中，你可以这样使用它：
# loss = composite_loss(outputs, batch_y)
