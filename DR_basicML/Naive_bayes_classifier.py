﻿import numpy as np


def loadCsv(filename):

    dataset=np.genfromtxt(filename, delimiter=',')
    for row in dataset:
        for i in range(len(row)):
            row[i] += 0.0001 #zero frequincy problem
    return dataset


def splitDataset(dataset, splitRatio):
    # Training set size
    trainSize = int(dataset.shape[0] * splitRatio)

    # List of randomly chosen indicies
    indices = np.random.permutation(dataset.shape[0])  # shuffle indexes

    # Split indicies for training and test set by trainSize
    training_idx, test_idx = indices[:trainSize], indices[trainSize:]

    # Create training and test sets by indicies
    training, test = dataset[training_idx, :], dataset[test_idx, :]

    return training, test

def separateByClass(dataset):
    separate = {}
    separated = {}
    for i in dataset[:,-1]:
        separate[i] = dataset[np.where(dataset[:, -1]==i), :]
    for i in dataset[:, -1]:
        separated[i] = (separate[i]
            ,separate[i].shape[1] / dataset.shape[0])
    return separated


def summarize(dataset):
    # Calculate means and standart deviations with one degree of freedom for each attribute
    # We do it by column which is axis 1
    # Also we remove last elements
    means = dataset[0].mean(axis=1)[0][:-1]  # WHY 0?
    stds = dataset[0].std(axis=1, ddof=1)[0][:-1]
    p_c = dataset[1]
    return means, stds, p_c

def summarizeByClass(dataset):
    # Divide dataset by class and summarize it
    separated = separateByClass(dataset)
    summaries = {}
    for classValue, instances in separated.items():
        summaries[classValue] = summarize(instances)
    return summaries

def calculateProbability(x, mean, stdev):
    return np.exp(-(x - mean)**2/(2 * stdev**2)) / (np.sqrt(2 * np.pi) * stdev)


def calculateClassProbabilities(summaries, inputVector):
    # Calculate probabilities for input vector from test set
    probabilities = {}

    for classValue, classSummaries in summaries.items():
        means = classSummaries[0]
        stds = classSummaries[1]
        p_c = classSummaries[2]

        # Calculate corresonding probabilities and multiply them
        probabilities[classValue] = np.prod(calculateProbability(inputVector[:-1], means, stds))*p_c

    return probabilities


def predict(summaries, inputVector):
    # Calculate probabilities
    probabilities = calculateClassProbabilities(summaries, inputVector)

    # Init values of probability and label
    bestLabel, bestProb = None, -1

    # Check probability of which class is better
    for classValue, probability in probabilities.items():
        if bestLabel is None or probability > bestProb:
            bestProb = probability
            bestLabel = classValue

    return bestLabel


def getPredictions(summaries, testSet):
    # For each probability find optimal labels
    predictions = []

    for i in range(len(testSet)):
        result = predict(summaries, testSet[i])
        predictions.append(result)

    return predictions


def getAccuracy(testSet, predictions):
    # Check accuracy
    correct = 0

    for i in range(len(testSet)):
        if testSet[i][-1] == predictions[i]:
            correct += 1
    return (correct / float(len(testSet))) * 100.0


def main():
    # Set initial data
    filename = 'data_baes.csv'

    # Set split ratio
    splitRatio = 0.67

    # Load dataset and return numpy array
    dataset = loadCsv(filename)

    # Split dataset
    trainingSet, testSet = splitDataset(dataset, splitRatio)

    # Log row amounts
    print('Split {0} rows into train={1} and test={2} rows'.format(len(dataset), len(trainingSet), len(testSet)))

    # Prepare model
    summaries = summarizeByClass(trainingSet)

    # Test model
    predictions = getPredictions(summaries, testSet)

    accuracy = getAccuracy(testSet, predictions)

    print('Accuracy: {0}%'.format(accuracy))

main()