import os
import pandas as pd
from tradingcalendar import Calendar
from log import log


class MeowDataLoader(object):
    def __init__(self, h5dir):
        self.h5dir = h5dir
        self.calendar = Calendar()
        self.threshold = 2

    def loadDates(self, dates):
        if len(dates) == 0:
            raise ValueError("Dates empty")
        log.inf("Loading data of {} dates from {} to {}...".format(len(dates), min(dates), max(dates)))
        df = pd.concat(self.loadDate(x) for x in dates)
        # print(df)
        return self.deal(df)
        #return df

    def loadDate(self, date):
        if not self.calendar.isTradingDay(date):
            raise ValueError("Not a trading day: {}".format(date))
        h5File = os.path.join(self.h5dir, "{}.h5".format(date))
        df = pd.read_hdf(h5File)
        df.loc[:, "date"] = date
        precols = ["symbol", "interval", "date"]
        df = df[precols + [x for x in df.columns if x not in precols]] # re-arrange columns
        # grouped = df.groupby("symbol")
        # print(df)
        # print(grouped.get_group(90301419))
        return df

    def deal(self, df):
        # 假设df是pandas DataFrame
        nn = [
            'midpx', 'lastpx', 'open',
       'high', 'low', 'bid0', 'ask0', 'bid4', 'ask4', 'bid9', 'ask9', 'bid19',
       'ask19', 'bsize0', 'asize0', 'bsize0_4', 'asize0_4', 'bsize5_9',
       'asize5_9', 'bsize10_19', 'asize10_19', 'btr0_4', 'atr0_4', 'btr5_9',
       'atr5_9', 'btr10_19', 'atr10_19', 'nTradeBuy', 'tradeBuyQty',
       'tradeBuyTurnover', 'tradeBuyHigh', 'tradeBuyLow', 'buyVwad',
       'nTradeSell', 'tradeSellQty', 'tradeSellTurnover', 'tradeSellHigh',
       'tradeSellLow', 'sellVwad', 'nAddBuy', 'addBuyQty', 'addBuyTurnover',
       'addBuyHigh', 'addBuyLow', 'nAddSell', 'addSellQty', 'addSellTurnover',
       'addSellHigh', 'addSellLow', 'nCxlBuy', 'cxlBuyQty', 'cxlBuyTurnover',
       'cxlBuyHigh', 'cxlBuyLow', 'nCxlSell', 'cxlSellQty', 'cxlSellTurnover',
       'cxlSellHigh', 'cxlSellLow']

        # 计算所有列的均值和标准差，避免在循环中重复计算
        means = df[nn].mean()
        stds = df[nn].std()

        # 一次性计算所有z-scores
        z_scores = (df[nn] - means) / stds

        # 一次性创建离群点的布尔索引
        outlier_mask = (z_scores.abs() > self.threshold).any(axis=1)

        # 使用布尔索引一次性删除所有离群点
        df_cleaned = df[~outlier_mask]

        return df_cleaned

