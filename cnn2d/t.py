import os  # 导入os模块，用于操作文件和目录路径
import numpy as np  # 导入numpy模块，用于进行数值计算
import pandas as pd  # 导入pandas模块，用于数据处理和分析

from feat import FeatureGenerator
from model import  Model
from dl import DataLoader
from tradingcalendar import Calendar
import torch

if __name__ == '__main__':
    epochs = 1
    calendar = Calendar()
    ft = FeatureGenerator(None)
    dloader = DataLoader("../archive")
    dates = calendar.range(20230601, 20230730)
    rawData = dloader.loadDates(dates)

    #ft.plot_correlation_heatmap(rawData)
    print(rawData.columns)
    X, y = ft.genFeatures(rawData)

    #ft.plot_correlation_heatmap(X)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mo = Model(device)
    mo.fit(X, y, epochs)
    mo.save_net('model/model.pth')
