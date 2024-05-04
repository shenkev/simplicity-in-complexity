import numpy as np
import scipy.stats as stats
from sklearn.metrics import r2_score, mean_squared_error


def _pearson(x, y):
    return stats.pearsonr(x, y).statistic

def _spearman(x, y):
    return stats.spearmanr(x, y).statistic

def _loglikelihood(model):
    return model.llf

def _aic(x, y, n, k):
    rss = np.sum((x - y) ** 2)
    return n * np.log(rss/n) + 2 * k

def _bic(x, y, n, k):
    rss = np.sum((x - y) ** 2)
    return n * np.log(rss/n) + k * np.log(n)

def _r2(x, y):
    return r2_score(x, y)

def _rmse(x, y):
    return np.sqrt(mean_squared_error(x, y))
