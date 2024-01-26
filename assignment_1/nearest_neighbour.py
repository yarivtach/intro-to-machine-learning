import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt



def gensmallm(x_list: list, y_list: list, m: int):
    """
    gensmallm generates a random sample of size m along side its labels.

    :param x_list: a list of numpy arrays, one array for each one of the labels
    :param y_list: a list of the corresponding labels, in the same order as x_list
    :param m: the size of the sample
    :return: a tuple (X, y) where X contains the examples and y contains the labels
    """
    assert len(x_list) == len(y_list), 'The length of x_list and y_list should be equal'

    x = np.vstack(x_list)
    y = np.concatenate([y_list[j] * np.ones(x_list[j].shape[0]) for j in range(len(y_list))])

    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)

    rearranged_x = x[indices]
    rearranged_y = y[indices]

    return rearranged_x[:m], rearranged_y[:m]


# todo: complete the following functions, you may add auxiliary functions or define class to help you

def learnknn(k: int, x_train: np.array, y_train: np.array):
    """

    :param k: value of the nearest neighbour parameter k
    :param x_train: numpy array of size (m, d) containing the training sample
    :param y_train: numpy array of size (m, 1) containing the labels of the training sample
    :return: classifier data structure
    """
    map = {}
    for i in range(len(x_train)):
        map[tuple(x_train[i])] = y_train[i]
    return [map, k]

def predictknn(classifier, x_test: np.array):
    """

    :param classifier: data structure returned from the function learnknn
    :param x_test: numpy array of size (n, d) containing test examples that will be classified
    :return: numpy array of size (n, 1) classifying the examples in x_test
    """

    ans= np.array([predict_helper(classifier, x_test[i]) for i in range(len(x_test))])
    #reshape to (n,1)
    return ans.reshape(-1,1)
   
        
def predict_helper(classifier, x_test: np.array):
    learning = classifier[0]
    k = classifier[1]
    map_X_to_distance = {}
    map_Label_to_count = {}
    
    for x in learning:
        map_X_to_distance[x] = distance.euclidean(x, x_test)
        if learning[x] in map_Label_to_count:
            map_Label_to_count[learning[x]] += 1
        else:
            map_Label_to_count[learning[x]] = 1

        if len(map_X_to_distance) > k:
            sorted_map = sorted(map_X_to_distance.items(), key=lambda x: x[1])
            key = sorted_map[-1][0]
            del map_X_to_distance[key]
            map_Label_to_count[learning[key]] -= 1

    sorted_map = sorted(map_Label_to_count.items(), key=lambda x: x[1])
    #return as array 
    return np.array(sorted_map[-1][0])

def simple_test():
    data = np.load('mnist_all.npz')

    train0 = data['train0']
    train1 = data['train1']
    train2 = data['train2']
    train3 = data['train3']

    test0 = data['test0']
    test1 = data['test1']
    test2 = data['test2']
    test3 = data['test3']

    x_train, y_train = gensmallm([train0, train1, train2, train3], [0, 1, 2, 3], 100)

    x_test, y_test = gensmallm([test0, test1, test2, test3], [0, 1, 2, 3], 50)

    classifer = learnknn(5, x_train, y_train)

    preds = predictknn(classifer, x_test)

    # tests to make sure the output is of the intended class and shape
    assert isinstance(preds, np.ndarray), "The output of the function predictknn should be a numpy array"
    assert preds.shape[0] == x_test.shape[0] and preds.shape[
        1] == 1, f"The shape of the output should be ({x_test.shape[0]}, 1)"

    # get a random example from the test set
    i = np.random.randint(0, x_test.shape[0])

    # this line should print the classification of the i'th test sample.
    print(f"The {i}'th test sample was classified as {preds[i]}")


def run_test():
    data = np.load('mnist_all.npz')

    train0 = data['train0']
    train1 = data['train1']
    train2 = data['train2']
    train3 = data['train3']

    test0 = data['test0']
    test1 = data['test1']
    test2 = data['test2']
    test3 = data['test3']

    error_array = []
    m_array = [10,25,55,80,95]
    for i in range(5):
        error = 0
        for j in range(10):
            x_train, y_train = gensmallm([train0, train1, train2, train3], [2,3,5,6], m_array[i])
            x_test, y_test = gensmallm([test0, test1, test2, test3], [2,3,5,6], 50)
            classifer = learnknn(5, x_train, y_train)
            preds = predictknn(classifer, x_test)
            error += np.mean(np.vstack(y_test)!= np.vstack(preds))
    
        print(f"The average error rate over 10 trials is {error/10}")
        error_array.append(error/10)    
    
    #plot
    #add axes 0-1 and 0-100
    
    plt.figure()
    plt.title('Error rate as a function of m')
    plt.xlim(0, 100)
    plt.ylim(0, 1)
    
    plt.scatter(m_array, error_array)
    plt.xlabel('m')
    plt.ylabel('error rate')

    #plot max line
    max_error = max(error_array)+0.01
    plt.plot([0,100],[max_error,max_error], color='red',linestyle='dashed') 

    min_error = min(error_array)-0.01
    plt.plot([0,100],[min_error,min_error], color='green',linestyle='dashed')

    plt.show()


if __name__ == '__main__':

    # before submitting, make sure that the function simple_test runs without errors
    run_test()


