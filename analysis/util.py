import numpy as np
from statistics import NormalDist


def confidence_interval(data, confidence=0.95):
  dist = NormalDist.from_samples(data)
  z = NormalDist().inv_cdf((1 + confidence) / 2.)
  h = dist.stdev * z / ((len(data) - 1) ** .5)
  return dist.mean - h, dist.mean + h


def results_to_mean_confidence_interval(results):
  return {k1: {k2: {k3: [np.mean(v3), confidence_interval(v3)] for k3, v3 in v2.items()} for k2, v2 in v1.items()} for k1, v1 in results.items()}