import matplotlib.pyplot as plt
import numpy as np


# Normalization
def normalize_dataset(dataset):
    minmax = list()
    for i in range(len(dataset[0])):
        col_values = [row[i] for row in dataset]
        value_min = min(col_values)
        value_max = max(col_values)
        minmax.append([value_min, value_max])
    for row in dataset:
        for i in range(len(row)):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])


def linear_regression():
    # Extract data
    points = np.genfromtxt('data.csv', delimiter=',')

    # Define hyperparameters

    # Learning rate
    learning_rate = 0.0001
    k = points.shape[1]
    theta=np.zeros(k)

    # number of iterations
    num_iterations = 10000

    # model training
    print(
        'Start learning at theta = {0}, error = {1}'.format(
            theta,
            compute_error(theta, points)
        )
    )

    theta = gradient_descent(theta, points, learning_rate, num_iterations)

    print(
        'End learning at theta = {0}, error = {1}'.format(
            theta,
            compute_error(theta, points)
        )
    )
    return theta


def compute_error(theta, points):  # Computes Error = 1/N * sum((y - (theta*x))^2)
    error = 0
    N = len(points)
    k = points.shape[1]
    x = np.ones(k)
    for i in range(N):
        for h in range(k - 1):
            x[h] = points[i, h]
        y = points[i, k-1]

        error += (y - (np.dot(x, theta))) ** 2
    return error / N


def gradient_descent(theta_s, points, learning_rate, num_iterations):
    '''
        Performs gradient step num_iterations times
        in order to find optimal a, b values
    '''
    theta = theta_s
    for i in range(num_iterations):
        theta = gradient_step(theta, points, learning_rate)
    return theta


def gradient_step(theta_c, points, learning_rate):
    '''
        Updates a and b in antigradient direction
        with given learning_rate
    '''
    theta = theta_c
    k = points.shape[1]
    grad_theta = np.zeros(k)
    x = np.ones(k)
    N = len(points)

    for i in range(N):
        for h in range(k - 1):
            x[h] = points[i, h]
        y = points[i, k-1]
        for j in range(k-1):
            grad_theta[j] += -(2 / N) * (y - np.dot(theta, x)) * x[j]
        grad_theta[k-1] += -(2 / N) * (y - np.dot(theta, x))



    theta = theta_c - learning_rate * grad_theta
    return theta


theta = linear_regression()


'''
# Learned function
points = np.genfromtxt('data.csv', delimiter=',')
X = points[:, 0]
Y = points[:, 1]

plt.xlim(0, 80)
plt.ylim(0, 150)
plt.scatter(X, Y)

params = np.linspace(0, 150, 10)

# plt.plot(params, a * params + b)

plt.show()
'''