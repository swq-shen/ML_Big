import os
import xgboost as xgb
from log import log

class MeowModel(object):
    def __init__(self, cacheDir):
        # 初始化XGBoost的模型
        self.estimator = xgb.XGBRegressor(
            objective='reg:squarederror',  # 对于回归问题
            alpha=0.5,  # L1正则化项
            random_state=None,
            tol=1e-8
        )

    def fit(self, xdf, ydf):
        # 使用XGBoost训练模型
        self.estimator.fit(
            X=xdf,
            y=ydf,
            eval_metric='rmse'  # 可以替换为其他评估指标
        )
        log.inf("Done fitting")

    def predict(self, xdf):
        # 使用XGBoost进行预测
        return self.estimator.predict(xdf)
