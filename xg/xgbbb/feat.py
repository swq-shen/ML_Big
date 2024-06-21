import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import seaborn as sns
from scipy.stats import yeojohnson
from sklearn.preprocessing import MinMaxScaler


class FeatureGenerator(object):  # 定义MeowFeatureGenerator类

    @classmethod  # 类方法，不需要实例化类即可调用
    def featureNames(cls):  # 返回特征名称列表
        return [
            "ob_imb0",  # 特征名称：订单簿不平衡0
            "ob_imb4",  # 特征名称：订单簿不平衡4
            "ob_imb9",  # 特征名称：订单簿不平衡9
            "trade_imb",  # 特征名称：交易不平衡
            "trade_imbema5",  # 特征名称：交易不平衡指数移动平均5
            "lagret12",  # 特征名称：滞后回报12
            "best_bid_change",
            "best_ask_change",
            "trade_activity",
            "trading_cost",
            "lag_bsize0",
            "volume_vwap_diff",
            "cxl_volume_to_mean_ratio",
            "midpx_change",
            "lastpx_change",
            "open_change",
            "high_low_spread",
            "bid_ask_spread4",
            "bid_ask_spread19",
            'trade_high_low_diff',
            'price_movement',
            'bid_ask_mid_spread',
            'turnover_rate',
            'k_mid',
            'k_sft',
            'k_up',
            'k_low',
            'buyVwad',
            'price_grad4',
            'price_grad19','nTradeSell', 'tradeSellQty', 'tradeSellTurnover', 'tradeSellHigh',
       'tradeSellLow', 'sellVwad', 'nAddBuy', 'addBuyQty', 'addBuyTurnover',
       'addBuyHigh', 'addBuyLow', 'nAddSell', 'addSellQty', 'addSellTurnover',
       'addSellHigh', 'addSellLow', 'nCxlBuy', 'cxlBuyQty', 'cxlBuyTurnover',
       'cxlBuyHigh', 'cxlBuyLow', 'nCxlSell', 'cxlSellQty', 'cxlSellTurnover',
       'cxlSellHigh', 'cxlSellLow',
            # 'SMA_5',
            # 'SMA_8',
            # 'SMA_13',
            # 'AO',
            'stock_tag'
            # 'log_return_ask0',
            # 'log_return_ask4',
            # 'log_return_ask9',
            # 'log_return_ask19',
            # 'log_return_bid0',
            # 'log_return_bid4',
            # 'log_return_bid9',
            # 'log_return_bid19',
            # 'RSI',
            # 'MACD',
            # 'MACD_signal'



        ]

    def __init__(self, cacheDir):  # 初始化方法，接收缓存目录作为参数
        self.cacheDir = cacheDir  # 存储缓存目录的属性
        self.ycol = "fret12"  # 目标列的名称
        self.mcols = ["symbol", "date", "interval"]  # 合并列的名称
        self.threshold = 2

    def genFeatures(self, df):  # 定义生成特征的方法，接收DataFrame
        # 计算特征值，这里列出了特征的计算方式：
        # 订单簿不平衡特征，使用买卖订单量差额除以总和
        df.loc[:, "ob_imb0"] = (df["asize0"] - df["bsize0"]) / (df["asize0"] + df["bsize0"])
        df.loc[:, "ob_imb4"] = (df["asize0_4"] - df["bsize0_4"]) / (df["asize0_4"] + df["bsize0_4"])
        df.loc[:, "ob_imb9"] = (df["asize5_9"] - df["bsize5_9"]) / (df["asize5_9"] + df["bsize5_9"])
        # 交易不平衡特征，使用买入交易量与卖出交易量的差额除以总和
        df.loc[:, "trade_imb"] = (df["tradeBuyQty"] - df["tradeSellQty"]) / (df["tradeBuyQty"] + df["tradeSellQty"])
        # 交易不平衡指数移动平均5，使用ewm方法计算半衰期为5的指数移动平均
        df.loc[:, "trade_imbema5"] = df["trade_imb"].ewm(halflife=5).mean()
        # 计算滞后回报12，即过去12个时间单位的回报率
        df.loc[:, "bret12"] = (df["midpx"] - df["midpx"].shift(12)) / df["midpx"].shift(12)
        # 计算交叉间隔的滞后回报平均值，并将其与原始滞后回报合并
        cxbret = df.groupby("interval")[["bret12"]].mean().reset_index().rename(columns={"bret12": "cx_bret12"})
        df = df.merge(cxbret, on="interval", how="left")
        # 计算滞后回报12的差值
        df.loc[:, "lagret12"] = df["bret12"] - df["cx_bret12"]
        # 最佳买价变动
        df['best_bid_change'] = df['bid0'] - df['bid0'].shift(12)  # 买一档价格的变化量
        # 最佳卖价变动
        df['best_ask_change'] = df['ask0'] - df['ask0'].shift(12)  # 卖一档价格的变化量
        # # 交易活跃度特征，使用买入和卖出交易次数的总和
        df.loc[:, "trade_activity"] = df["nTradeBuy"] + df["nTradeSell"]
        # 交易成本指标，使用买卖价差与成交量的乘积
        df.loc[:, "trading_cost"] = (df["ask0"] - df["bid0"]) * (df["tradeBuyQty"] + df["tradeSellQty"]) / 2
        # # 买卖订单的变动率特征，使用当前订单量与前一周期订单量的差比
        df.loc[:, "lag_bsize0"] = df["bsize0"].shift(12)
        # 交易量与成交量加权平均价格的差异特征
        df.loc[:, "volume_vwap_diff"] = df["tradeBuyQty"] - df["buyVwad"]
        # 撤单量与历史平均撤单量的比率特征
        df.loc[:, "cxl_volume_to_mean_ratio"] = df["cxlBuyQty"] / df["cxlBuyQty"].mean()

        # 中间价格变动
        df['midpx_change'] = df['midpx'] - df['midpx'].shift(1)
        # 最后成交价格变动
        df['lastpx_change'] = df['lastpx'] - df['lastpx'].shift(1)
        # 开盘价格变动
        df['open_change'] = df['open'] - df['open'].shift(1)
        # 最高价与最低价的差
        df['high_low_spread'] = df['high'] - df['low']
        # 买卖价差
        df['bid_ask_spread0'] = df['ask0'] - df['bid0']
        df['bid_ask_spread4'] = df['ask4'] - df['bid4']
        df['bid_ask_spread9'] = df['ask9'] - df['bid9']
        df['bid_ask_spread19'] = df['ask19'] - df['bid19']
        # 成交高低价差异
        df['trade_high_low_diff'] = df['tradeBuyHigh'] - df['tradeSellLow']

        # 价格变动趋势
        df['price_movement'] = df['lastpx'].diff()#kykw
        # 买卖价与中间价的平均价差
        df['bid_ask_mid_spread'] = (df['ask0'] + df['bid0']) / 2 - df['midpx']#kykw
        # 成交额转换率
        df['turnover_rate'] = df['tradeBuyTurnover'].div(df['tradeBuyQty'])

        df['k_mid'] = (df['lastpx'] - df['open']) / df['open']
        df['k_mid2'] = (df['lastpx'] - df['open']) / (df['high'] - df['low'])

        df['k_up'] = (df['high'] - df[['open', 'lastpx']].max(axis=1)) / df['open']
        df['k_low'] = (df[['open', 'lastpx']].min(axis=1) - df['low']) / df['open']

        df['k_sft'] = (2 * df['lastpx'] - df['high'] - df['low']) / df['open']
        df['k_sft2'] = (2 * df['lastpx'] - df['high'] - df['low']) / (df['high'] - df['low'])

        df['price_grad4'] = (df['ask4'] - df['bid4']) / 4
        df['price_grad9'] = (df['ask9'] - df['bid9']) / 9
        df['price_grad19'] = (df['ask19'] - df['bid19']) / 19

        df['stock_tag'] = df['symbol']

        df['RSI'] = self.relative_strength_idx(df).fillna(0)

        df['log_return_ask0'] = np.log((df['ask0'] - df['ask0'].shift(1)) / df['ask0'].shift(1) + 1)
        df['log_return_ask4'] = np.log((df['ask4'] - df['ask4'].shift(1)) / df['ask4'].shift(1) + 1)
        df['log_return_ask9'] = np.log((df['ask9'] - df['ask9'].shift(1)) / df['ask9'].shift(1) + 1)
        df['log_return_ask19'] = np.log((df['ask19'] - df['ask19'].shift(1)) / df['ask19'].shift(1) + 1)

        df['log_return_bid0'] = np.log((df['bid0'] - df['bid0'].shift(1)) / df['bid0'].shift(1) + 1)
        df['log_return_bid4'] = np.log((df['bid4'] - df['bid4'].shift(1)) / df['bid4'].shift(1) + 1)
        df['log_return_bid9'] = np.log((df['bid9'] - df['bid9'].shift(1)) / df['bid9'].shift(1) + 1)
        df['log_return_bid19'] = np.log((df['bid19'] - df['bid19'].shift(1)) / df['bid19'].shift(1) + 1)

        EMA_12 = pd.Series(df['lastpx'].ewm(span=12, min_periods=12).mean())
        EMA_26 = pd.Series(df['lastpx'].ewm(span=26, min_periods=26).mean())
        df['MACD'] = pd.Series(EMA_12 - EMA_26)
        df['MACD_signal'] = pd.Series(df.MACD.ewm(span=9, min_periods=9).mean())

        df['SMA_5'] = ((df['high'] + df['low']) / 2).rolling(5).mean().shift()
        df['SMA_8'] = ((df['high'] + df['low']) / 2).rolling(8).mean().shift()
        df['SMA_13'] = ((df['high'] + df['low']) / 2).rolling(8).mean().shift()

        df['AO'] = ((df['high'] + df['low']) / 2).rolling(5).mean().shift() - ((df['high'] + df['low']) / 2).rolling(34).mean().shift()

        df['mid_price0_4'] = (df['atr0_4'] + df['btr0_4']) / (df['asize0_4'] + df['bsize0_4'])
        df['mid_price5_9'] = (df['atr5_9'] + df['btr5_9']) / (df['asize5_9'] + df['bsize5_9'])
        df['mid_price10_19'] = (df['atr10_19'] + df['btr10_19']) / (df['asize10_19'] + df['bsize10_19'])

        # 将特征和目标列设置为DataFrame的索引，并填充缺失值
        xdf = df[self.mcols + self.featureNames()].set_index(self.mcols).fillna(0)
        ydf = df[self.mcols + [self.ycol]].set_index(self.mcols).fillna(0)
        return xdf, ydf  # 返回特征DataFrame和目标DataFrame

    def relative_strength_idx(self, df, n=14):
        close = df['lastpx']
        delta = close.diff()
        delta = delta[1:]
        pricesUp = delta.copy()
        pricesDown = delta.copy()
        pricesUp[pricesUp < 0] = 0
        pricesDown[pricesDown > 0] = 0
        rollUp = pricesUp.rolling(n).mean()
        rollDown = pricesDown.abs().rolling(n).mean()
        rs = rollUp / rollDown
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi

    def plot_correlation_heatmap(self, df):
        # 计算特征列之间的相关系数矩阵
        corr_matrix = df.corr()
        # 使用Seaborn绘制热力图
        plt.figure(figsize=(10, 8))  # 可以根据需要调整图形大小
        sns.heatmap(corr_matrix, annot=False, fmt=".2f", cmap='coolwarm', square=True)
        plt.title('Feature Correlation Heatmap')
        plt.show()
        self.find_high_correlation_with_row(df)

    def deal(self, df):
        # 计算所有列的均值和标准差
        means = df[self.featureNames()].mean()
        stds = df[self.featureNames()].std()
        # 一次性计算所有z-scores
        z_scores = (df[self.featureNames()] - means) / stds
        # 一次性创建离群点的布尔索引
        outlier_mask = (z_scores.abs() > self.threshold).any(axis=1)
        # 使用布尔索引一次性删除所有离群点
        df_cleaned = df[~outlier_mask]
        return df_cleaned

    def find_high_correlation_with_row(self, df, row_name='bret12', threshold=0.2):
        # 计算特征列之间的相关系数矩阵
        corr_matrix = df.corr()
        # 获取特定行的相关系数
        nn_correlations = corr_matrix.loc[row_name]
        # 找出与特定行相关系数绝对值大于阈值的列名
        high_corr_columns = [col for col in nn_correlations.index if abs(nn_correlations[col]) > threshold]
        # 打印这些列名
        print(f"Columns with correlation greater than {threshold} with '{row_name}':")
        for col in high_corr_columns:
            print(col)
