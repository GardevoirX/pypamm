from numpy.linalg import norm
from .dist import pammr2

def _euclidean_distance(x, y, period = None):
    return norm(x - y, axis=1)

def _euclidean_distance_period(x, y, period):
    return pammr2(period, x, y)
