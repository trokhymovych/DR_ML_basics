import numpy as np


class Kernel(object):
    """Check kernels here https://en.wikipedia.org/wiki/Support_vector_machine"""
    @staticmethod
    def linear():
        return lambda x, y: np.inner(x, y)

    @staticmethod
    def gaussian(sigma=5):
        return lambda x, y: np.exp(-(np.linalg.norm(x - y))**2 / (2 * sigma**2))
    
    @staticmethod
    def polynomial_kernel(p=3):
        return lambda x, y: np.power((1 + np.dot(x, y)), p)
