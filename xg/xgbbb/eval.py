import os
import numpy as np
import pandas as pd
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
        log.inf("Meow evaluation summary: Pearson correlation={:.4f}, R2={:.5f}, MSE={:.8f}".format(pcor, r2, mse))
