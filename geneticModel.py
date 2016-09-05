import numpy as np
from numpy.random import choice
from sklearn import linear_model
import operator


def get_partition(numNodes, numSubsets):
    """
    Function to return a partition of a set of nodes

    Parameters
    ----------
    numNodes: int
        Total number of nodes
    numSubsets: int
        Number of subsets per partition

    Returns
    ----------
    partitionData: list of list
        Partition on the nodes

    """

    nodes = np.arange(numNodes)
    np.random.shuffle(nodes)

    lenSplit = int(np.floor(numNodes / numSubsets))

    partitionData = []

    for part in range(0, numNodes, lenSplit):
        try:
            partitionData.append(nodes[part: part + lenSplit])
        except IndexError:
            partitionData.append(nodes[part: -1])

    return partitionData


def createGeneticTrainingData(data, dataTrans, nodesToPred, nodes, startTimeStep, trainingWindow, offset=24,
                              hoursBack=0):
    """
    Function returns training data for a node

    Parameters
    ----------
    data: Pandas DataFrame
        Dataset we want run the algorithm on
    dataTrans: Pandas DataFrame
        Transformed dataset used to get generate pattern target pairs
    nodesToPred: list
        Nodes current model predicts on
    nodes: list
        Nodes current model is trained on
    startTimeStep: int
        Start time of when we want to run the algorithm on the data. This should be a value within the index on the
        data.
    trainingWindow: int
        Length of the training window
    offset: int
        Difference between the pattern and target pairs
    hoursBack: int or list
        The previous hours we want to concatenate in the data


    Returns
    ----------
    X: numpy array
        Array of patterns
    Y: numpy array
        Array of targets

    """

    # select correct nodes from hours back
    if isinstance(hoursBack, list):
        nodesBack = np.arange(len([0] + hoursBack))
        origDataNodes = dataTrans.shape[1] / len(nodesBack)
        nodes = [node + origDataNodes * nb for node in nodes for nb in nodesBack]
    elif not hoursBack == 0:
        nodesBack = np.arange(len([0] + np.arange(hoursBack)))
        origDataNodes = dataTrans.shape[1] / len(nodesBack)
        nodes = [node + origDataNodes * nb for node in nodes for nb in nodesBack]

    # Each element in array is one training X point with dimension equal to numNodes
    X = np.array(dataTrans.iloc[startTimeStep: startTimeStep + trainingWindow, nodes].values)
    # Each element in array now only one dimensional. Offset added here to predict ahead
    Y = np.array(data.iloc[startTimeStep + offset: startTimeStep + offset + trainingWindow, nodesToPred].values)

    return np.array(X), np.array(Y)


def trainGeneticExperts(data, dataTrans, numModels, alpha, startTimeStep=0, trainingWindow=100, offset=24,
                        hoursBack=0):
    """
    Function generates a partition and trains model on it

    Parameters
    ----------
    data: Pandas DataFrame
        Dataset we want run the algorithm on
    dataTrans: Pandas DataFrame
        Transformed dataset used to get generate pattern target pairs
    numModels: int
        Number of partitions we want
    alpha: float
        Ridge regression parameter, lambda in report
    startTimeStep: int
        Start time of when we want to run the algorithm on the data. This should be a value within the index on the
        data.
    trainingWindow: int
        Length of the training window
    offset: int
        Difference between the pattern and target pairs
    hoursBack: int or list
        The previous hours we want to concatenate in the data


    Returns
    ----------
    experts: list
        List of sklearn models which correspond to a subset of a partition
    paritition: list of list
        List of subsets of partition

    """

    experts = []
    numNodes = data.shape[1]
    partition = get_partition(numNodes, numModels)

    for i in range(numModels):

        # nodes randomly chosen in the createTrainingData function for this model
        X, Y = createGeneticTrainingData(data, dataTrans, partition[i], partition[i], startTimeStep, trainingWindow,
                                         offset, hoursBack)

        model = linear_model.Ridge(alpha=alpha)  # sklearn model
        model.fit(X, Y)  # fit mode

        experts.append(model)  # append to models list

    return experts, partition


def get_testing_data(data, nodes, startTimeStep, trainingWindow, hoursBack=0, fixedShare=False):

    """
    Function to get testing data for current expert in partition

    Parameters
    ----------
    data: Pandas DataFrame
        Transformed data set  Note: we do not need the orginal data set here.
    nodes: list
        which current expert predicts on
    startTimeStep: int
        Start time of when we want to run the algorithm on the data. This should be a value within the index on the
        data.
    trainingWindow: int
        Length of the training window
    hoursBack: int or list
        The previous hours we want to concatenate in the data
    fixedShare: Boolean
        True, if we are using this with the fixed share algorithm, false otherwise.

    Returns
    ----------
    X: numpy array
        testing data

    """

    if isinstance(hoursBack, list):
        nodesBack = np.arange(len([0] + hoursBack))
        origDataNodes = data.shape[1] / len(nodesBack)
        nodes = [node + origDataNodes * nb for node in nodes for nb in nodesBack]
    elif not hoursBack == 0:
        nodesBack = np.arange(len([0] + np.arange(hoursBack)))
        origDataNodes = data.shape[1] / len(nodesBack)
        nodes = [node + origDataNodes * nb for node in nodes for nb in nodesBack]

    if not fixedShare:
        X = data.iloc[startTimeStep + trainingWindow: startTimeStep + 2 * trainingWindow, nodes].values
    else:
        X = data.iloc[startTimeStep: startTimeStep + trainingWindow, nodes].values

    return X


def dataTransform(data, startTimeStep, endTimeStep, hoursBack=0):
    """
    Function that transforms the data by concatenating previous hours together.

    Parameters
    ----------
    data: Pandas DataFrame
        Transformed data set  Note: we do not need the original data set here.
    startTimeStep: int
        Start time of when we want to run the algorithm on the data. This should be a value within the index on the
        data.
    endTimeStep: int
        Final time step we want in the transformed data set.
    hoursBack: int or list
        The previous hours we want to concatenate in the data

    Returns
    ----------
    dataTrans: numpy array
        Transformed data set

    """

    numNodes = data.shape[1]

    # Each element in array is one training X point with dimension equal to numNodes
    X = np.array(data.iloc[startTimeStep: startTimeStep + endTimeStep, :].values)

    try:
        hours_length = len(hoursBack)
    except TypeError:
        hours_length = hoursBack

    if not np.all(hoursBack == 0):
        X_temp = X.copy()
        X_oldShape = X.shape
        X = np.zeros((X_oldShape[0], X_oldShape[1] + hours_length * numNodes))
        X[:, :-hours_length * numNodes] = X_temp

        if not isinstance(hoursBack, list):  # hoursBack not a list
            for X_index, _ in enumerate(X):
                X[X_index, X_oldShape[1]:] = data.iloc[startTimeStep + X_index - hours_length:
                                                       startTimeStep + X_index, :].stack().values
        else:  # hoursBack is a list
            for X_index, _ in enumerate(X):
                hours = [startTimeStep + X_index - h for h in hoursBack]
                X[X_index, X_oldShape[1]:] = data.iloc[hours, :].stack().values

    return np.array(X)


def geneticSelection(data, dataTrans, numModels, numParts, numOffspring, numSwap, alpha, startTimeStep,
                     trainingWindow, lossFun, threshold=1, offset=24, numGenerations=10, expertsReturn=5,
                     hoursBack=0):
    """
    Function to using a genetic algorithm to search a space of partitions.

    Parameters
    ----------
    data: Pandas DataFrame
        Data set we want run the algorithm on
    dataTrans: Pandas DataFrame
        Transformed data set used to get generate pattern target pairs
    numModels: int
        Number of models we want per partition
    numParts: int
        Number of partitions we want
    numOffspring: int
        Number of offspring per generation for each partition
    numSwap: int
        Number of elements swapped when mutating a partition
    alpha: float
        Ridge regression parameter
    startTimeStep: int
        Start time of when we want to run the algorithm on the data. This should be a value within the index on the
        data.
    trainingWindow: int
        Length of the training window
    lossFun: func
        Loss function to be used
    threshold: float
        Parameter used to drop models.
    offset: int
        Difference between the pattern and target pairs
    numGenerations: int
        Number of generations we are willing to run the algorithm for
    expertsReturn: int
        Number of experts we want to return. For values greater than 1 we return expertsReturn-1 best experts 1
        and the worst expert, hoping that it will have overfit less it we ran this for too many generations.
    hoursBack: int or list
        The previous hours we want to concatenate in the data


    Returns
    ----------
    experts_error: list
        Error of experts returns on the testing set
    partitions: list of list
        List of partitions of the experts
    experts: list
        List of sklearn models.

    """
    numNodes = data.shape[1]
    partitions = []
    experts = []
    experts_error = []
    error_check = []  # list to check if we have already calculated the error for a partition

    if numNodes / numModels < numSwap:
        print('Number of nodes to swap greater than number of nodes per expert!')

    # inital models
    for j in range(numParts):
        #  generate initial partitions
        expert, expertNodes = trainGeneticExperts(data, dataTrans, numModels, alpha, startTimeStep, trainingWindow,
                                                  offset, hoursBack)
        experts.append(expert)
        partitions.append(expertNodes)

        error_check.append(False)

    print("Initial experts trained.")

    for i in range(numGenerations):

        # create offsprings to current models
        temp_experts = []
        temp_partitions = []
        for partIndex, _ in enumerate(partitions):
                for offspring in range(numOffspring):
                    newExpert_nodes = np.copy(partitions[partIndex])

                    partsToCross = choice(len(partitions[0]), size=2, replace=False)

                    newExpert1_nodes = np.copy(partitions[partIndex][partsToCross[0]])  # take first expert for
                    # crossover
                    newExpert2_nodes = np.copy(partitions[partIndex][partsToCross[1]])  # second expert for crossover

                    if len(newExpert1_nodes) == len(newExpert2_nodes):
                        if len(newExpert1_nodes) > numSwap:
                            nodesToSwap = choice(len(newExpert1_nodes), size=numSwap, replace=False)  # index of nodes to swap
                            dummy = newExpert1_nodes[nodesToSwap]
                            newExpert1_nodes[nodesToSwap] = newExpert2_nodes[nodesToSwap]
                            newExpert2_nodes[nodesToSwap] = dummy

                    elif len(newExpert1_nodes) < len(newExpert2_nodes):
                        if len(newExpert1_nodes) > numSwap:  # length is less than numSwap
                            nodesToSwap = choice(len(newExpert1_nodes), size=numSwap, replace=False)
                            dummy = newExpert1_nodes[nodesToSwap]
                            newExpert1_nodes[nodesToSwap] = newExpert2_nodes[nodesToSwap]
                            newExpert2_nodes[nodesToSwap] = dummy
                        elif len(newExpert1_nodes) < numSwap:
                            nodesToSwap = np.arange(len(newExpert1_nodes))  # swap all nodes of the smaller partition
                            dummy = newExpert1_nodes[nodesToSwap]
                            newExpert1_nodes[nodesToSwap] = newExpert2_nodes[nodesToSwap]
                            newExpert2_nodes[nodesToSwap] = dummy

                    elif len(newExpert2_nodes) < len(newExpert1_nodes):
                        if len(newExpert2_nodes) > numSwap:  # length is less than numSwap
                            nodesToSwap = choice(len(newExpert2_nodes), size=numSwap, replace=False)
                            dummy = newExpert2_nodes[nodesToSwap]
                            newExpert2_nodes[nodesToSwap] = newExpert1_nodes[nodesToSwap]
                            newExpert1_nodes[nodesToSwap] = dummy
                        elif len(newExpert2_nodes) < numSwap:
                            nodesToSwap = np.arange(len(newExpert2_nodes))  # swap alll nodes of the smaller partition
                            dummy = newExpert2_nodes[nodesToSwap]
                            newExpert2_nodes[nodesToSwap] = newExpert1_nodes[nodesToSwap]
                            newExpert1_nodes[nodesToSwap] = dummy
                    # take care of edge cases when the partitions are not the same size
                    # Do not need to check if the nodes are unique in each partition as this how a partition is
                    # defined so they will always be unique.

                    # replace the nodes into the partitions
                    newExpert_nodes[partsToCross[0]] = newExpert1_nodes  # take first expert for crossover
                    newExpert_nodes[partsToCross[1]] = newExpert2_nodes  # second expert for crossover

                    temp_partitions.append(newExpert_nodes)

                    newExpert = []
                    for k in range(len(newExpert_nodes)):

                        # nodes randomly chosen in the createTrainingData function for this model
                        X, Y = createGeneticTrainingData(data, dataTrans, newExpert_nodes[k], newExpert_nodes[k],
                                                         startTimeStep, trainingWindow, offset, hoursBack)

                        # Use Lasso to find sparse representation of the data
                        # TODO: Using ridge or other model here
                        model = linear_model.Ridge(alpha=alpha)  # sklearn model
                        # model = linear_model.Lasso(alpha=alpha)  # sklearn model
                        model.fit(X, Y)  # fit model

                        newExpert.append(model)  # append to models list

                    temp_experts.append(newExpert)  # append the new experts to the array

                    error_check.append(False)

                    # TODO: move error cal here

                    print('Expert {} offspring {} complete.'.format(partIndex, offspring + 1))

        experts = experts + temp_experts
        partitions = partitions + temp_partitions

        forecast = np.zeros([trainingWindow, numNodes])
        for partitionsIndex, partition in enumerate(experts):
            if not error_check[partitionsIndex]:  # if we have not calculate error already
                for expertIndex, expert in enumerate(partition):
                    X_test = get_testing_data(dataTrans, partitions[partitionsIndex][expertIndex],
                                              startTimeStep, trainingWindow, hoursBack)

                    forecast[:, partitions[partitionsIndex][expertIndex]] = expert.predict(X_test)

                error = lossFun(data.iloc[startTimeStep + offset: startTimeStep + trainingWindow + offset, :].values,
                                forecast)
                experts_error.append(error)
                error_check[partitionsIndex] = True

        print(experts_error)

        print('Experts Trained from {} to {} and tested from {} to {}'.format(startTimeStep,
                                                                              startTimeStep + trainingWindow,
                                                                              startTimeStep + trainingWindow,
                                                                              startTimeStep + 2*trainingWindow))

        # Remove partitions that are below threshold.

        best_error = min(experts_error)

        temp_expert_error = []
        temp_partitions = []
        temp_experts = []
        temp_error_check = []

        for expertIndex, expert_error in enumerate(experts_error):
            if expert_error < threshold * best_error:  # keeps nodes that have good error
                temp_expert_error.append(experts_error[expertIndex])
                temp_partitions.append(partitions[expertIndex])
                temp_experts.append(experts[expertIndex])
                temp_error_check.append(True)

        experts_error = temp_expert_error
        partitions = temp_partitions
        experts = temp_experts
        error_check = temp_error_check

        print("Generation ", str(i + 1), " complete.")

    # return best experts
    temp_expert_error = []
    temp_partitions = []
    temp_experts = []

    i = 0  # only one counter for both for loops
    for error, part, expert in sorted(zip(experts_error, partitions, experts), key=operator.itemgetter(0)):
        if i < expertsReturn - 1:
            temp_expert_error.append(error)
            temp_partitions.append(part)
            temp_experts.append(expert)
        elif expertsReturn == 1 and i == 0:
            temp_expert_error.append(error)
            temp_partitions.append(part)
            temp_experts.append(expert)
        i += 1

    for error, part, expert in sorted(zip(experts_error, partitions, experts), reverse=True, key=operator.itemgetter(0)):
        if i < expertsReturn:
            temp_expert_error.append(error)
            temp_partitions.append(part)
            temp_experts.append(expert)
        i += 1

    experts_error = temp_expert_error
    partitions = temp_partitions
    experts = temp_experts

    print(experts_error)

    return experts_error, partitions, experts

