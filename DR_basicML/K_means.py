# necessary imports

import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np

points = np.vstack(((np.random.randn(150, 2) * 0.75 + np.array([1, 0])),
                    (np.random.randn(50, 2) * 0.25 + np.array([-0.5, 0.5])),
                    (np.random.randn(50, 2) * 0.5 + np.array([-0.5, -0.5]))))


def initialize_centroids(points, k):
    '''
        Selects k random points as initial
        points from dataset
    '''
    centroids = points.copy()
    np.random.shuffle(centroids)
    return centroids[:k]


def closest_centroid(points, centroids):
    '''
        Returns an array containing the index to the nearest centroid for each point
    '''
    distances = np.sqrt(((points - centroids[:, np.newaxis]) ** 2).sum(axis=2))
    return np.argmin(distances, axis=0)


def move_centroids(points, closest, centroids):
    '''
        Returns the new centroids assigned from the points closest to them
    '''
    return np.array([points[closest == k].mean(axis=0) for k in range(centroids.shape[0])])


def main(points):
    num_iterations = 100
    k = 3

    # Initialize centroids
    centroids = initialize_centroids(points, 3)

    # Run iterative process
    for i in range(num_iterations):
        closest = closest_centroid(points, centroids)
        centroids = move_centroids(points, closest, centroids)

    return centroids



centroids = main(points)
centroids = initialize_centroids(points, 3)
plt.ion()

n=0
while (n<10):
    closest = closest_centroid(points, centroids)
    centroids = move_centroids(points, closest, centroids)
    plt.scatter(points[:, 0], points[:, 1])
    plt.scatter(centroids[:, 0], centroids[:, 1], c='r', s=100)
    ax = plt.gca()
    plt.draw()
    plt.pause(0.9)
    plt.clf()
    n+=1
