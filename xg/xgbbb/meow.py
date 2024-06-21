import os

import pandas as pd

from log import log
from dl import MeowDataLoader
from feat import FeatureGenerator
from mdl import MeowModel
from eval import MeowEvaluator
from tradingcalendar import Calendar
import warnings
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV

warnings.filterwarnings("ignore")


parameters = {
    'n_estimators': [100, 200, 300, 400],
    'learning_rate': [0.001, 0.005, 0.01, 0.05],
    'max_depth': [8, 10, 12, 15],
    'gamma': [0.001, 0.005, 0.01, 0.02],
    'random_state': [42]
}

class MeowEngine(object):
    def __init__(self, h5dir, cacheDir):
        self.calendar = Calendar()
        self.h5dir = h5dir
        if not os.path.exists(h5dir):
            raise ValueError("Data directory not exists: {}".format(self.h5dir))
        if not os.path.isdir(h5dir):
            raise ValueError("Invalid data directory: {}".format(self.h5dir))
        self.cacheDir = cacheDir # this is not used in sample code
        self.dloader = MeowDataLoader(h5dir=h5dir)
        self.featGenerator = FeatureGenerator(cacheDir=cacheDir)
        self.model = MeowModel(cacheDir=cacheDir)
        self.evaluator = MeowEvaluator(cacheDir=cacheDir)

    def fit(self, startDate, endDate):
        dates = self.calendar.range(startDate, endDate)
        rawData = self.dloader.loadDates(dates)
        log.inf("Running model fitting...")
        xdf, ydf = self.featGenerator.genFeatures(rawData)
        # print(xdf)
        # 以下为先提取特征再groupby的过程，效果不佳
        # grouped_x = xdf.groupby("symbol")
        # grouped_y = ydf.groupby("symbol")
        # x_list = []
        # y_list = []
        # for symbol, group in grouped_x:
        #     x_list.append(group)
        # for symbol, group in grouped_y:
        #     y_list.append(group)
        # for i in range(0, len(x_list) - 1):
        #     self.model.fit(x_list[i], y_list[i])
        self.model.fit(xdf, ydf)

    def predict(self, xdf):
        return self.model.predict(xdf)

    def eval(self, startDate, endDate):
        log.inf("Running model evaluation...")
        dates = self.calendar.range(startDate, endDate)
        rawData = self.dloader.loadDates(dates)
        xdf, ydf = self.featGenerator.genFeatures(rawData)
        ydf.loc[:, "forecast"] = self.predict(xdf)
        self.evaluator.eval(ydf)

    def judge(self):
        dates_train = self.calendar.range(20230601, 20230630)
        dates_valid = self.calendar.range(20230703, 20230731)
        rawData_train = self.dloader.loadDates(dates_train)
        rawData_valid = self.dloader.loadDates(dates_valid)
        xdf_train, ydf_train = self.featGenerator.genFeatures(rawData_train)
        xdf_valid, ydf_valid = self.featGenerator.genFeatures(rawData_valid)
        eval_set = [(xdf_train, ydf_train), (xdf_valid, ydf_valid)]
        model_jugde = xgb.XGBRegressor(eval_set=eval_set, objective='reg:squarederror', verbose=False)
        clf = GridSearchCV(model_jugde, parameters)
        clf.fit(xdf_train, ydf_train)
        print(f'Best params: {clf.best_params_}')
        print(f'Best validation score = {clf.best_score_}')



if __name__ == "__main__":
    engine = MeowEngine(h5dir="../archive", cacheDir=None)
    # engine.fit(20230601, 20230601)
    # engine.judge()
    engine.fit(20230601, 20231130)
    engine.eval(20231201, 20231229)
