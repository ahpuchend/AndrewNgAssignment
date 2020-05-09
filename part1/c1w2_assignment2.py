import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage


def load_dataset():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels
    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels
    classes = np.array(test_dataset["list_classes"][:])  # the list of classes
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

def sigmoide(z):
    s = 1. / (1 + np.exp(-z))
    return s

def tanh(z):
    s = (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))
    return s

def initializing_Parameters(ndim):
    # w = np.zeros([ndim, 1])
    w = np.random.randn(ndim, 1) * 0.01
    b = 0.001
    return w, b

def propagate(w, b, X, Y):
    """
        Implement the cost function and its gradient for the propagation explained above

        Arguments:
        w -- weights, a numpy array of size (num_px * num_px * 3, 1)
        b -- bias, a scalar
        X -- data of size (num_px * num_px * 3, number of examples)
        Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)

        Return:
        cost -- negative log-likelihood cost for logistic regression
        dw -- gradient of the loss with respect to w, thus same shape as w
        db -- gradient of the loss with respect to b, thus same shape as b

        Tips:
        - Write your code step by step for the propagation. np.log(), np.dot()
    """
    m = X.shape[1]
    A = sigmoide(np.dot(w.T, X) + b)
    cost = -1. * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A)) / m
    # 梯度在这里计算
    dw = 1./m * np.dot(X, (A-Y).T)
    db = 1./m * np.sum(A-Y)
    grads = {"dw": dw,"db": db}
    return grads,cost


def optimize(w, b, X, Y, num_iterations, learning_rate):
    """
       This function optimizes w and b by running a gradient descent algorithm
       Arguments:
       w -- weights, a numpy array of size (num_px * num_px * 3, 1)
       b -- bias, a scalar
       X -- data of shape (num_px * num_px * 3, number of examples)
       Y -- true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
       num_iterations -- number of iterations of the optimization loop
       learning_rate -- learning rate of the gradient descent update rule
       print_cost -- True to print the loss every 100 steps
       Returns:
       params -- dictionary containing the weights w and bias b
       grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
       costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.

       Tips:
       You basically need to write down two steps and iterate through them:
           1) Calculate the cost and the gradient for the current parameters. Use propagate().
           2) Update the parameters using gradient descent rule for w and b.
       """

    costs = []
    for i in range(num_iterations):
        # froward propagation
        grads,cost = propagate(w,b,X,Y)
        dw = grads["dw"]
        db = grads["db"]

        # backward propagation
        w = w - learning_rate*dw
        b = b - learning_rate*db

        if i != 0 and i % 100 == 0:
            costs.append(cost)
            # Print the cost every 100 training examples
        if i!= 0 and  i % 100 == 0:
            print("Cost after iteration i {},f {}".format(i, cost))

    # record final weights and grads
    params = {"w": w, "b": b}
    grads = {"dw": dw, "db": db}

    return params, grads, costs

def predict(w, b, X):
    # The previous function will output the learned w and b.
    # We are able to use w and b to predict the labels for a dataset X.
    # Implement the predict() function. There is two steps to computing predictions:
    '''
       Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)
       Arguments:
       w -- weights, a numpy array of size (num_px * num_px * 3, 1)
       b -- bias, a scalar
       X -- data of size (num_px * num_px * 3, number of examples)

       Returns:
       Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
    '''
    m = X.shape[1]
    Z = np.dot(w.T,X) + b
    Y_prediction = np.zeros([1,m])
    A = sigmoide(Z)
    # np.where(Y_prediction>0.5? 1:0)
    for i in range(A.shape[1]):
        if A[0, i] <= 0.5:
            Y_prediction[0, i] = 0
        else:
            Y_prediction[0, i] = 1
    return Y_prediction

def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.05):
    """
       Builds the logistic regression model by calling the function you've implemented previously
       Arguments:
       X_train -- training set represented by a numpy array of shape (num_px * num_px * 3, m_train)
       Y_train -- training labels represented by a numpy array (vector) of shape (1, m_train)
       X_test -- test set represented by a numpy array of shape (num_px * num_px * 3, m_test)
       Y_test -- test labels represented by a numpy array (vector) of shape (1, m_test)
       num_iterations -- hyperparameter representing the number of iterations to optimize the parameters
       learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()
       print_cost -- Set to true to print the cost every 100 iterations
       Returns:
       d -- dictionary containing information about the model.
    """
    w,b = initializing_Parameters(X_train.shape[0])
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate)

    w = parameters["w"]
    b = parameters["b"]

    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train": Y_prediction_train,
         "w": w,
         "b": b,
         "learning_rate": learning_rate,
         "num_iterations": num_iterations
         }
    return d

if __name__ == '__main__':
    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
    train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
    test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
    train_set_x = train_set_x_flatten / 255.
    test_set_x = test_set_x_flatten / 255.
    d = model(train_set_x,train_set_y,test_set_x,test_set_y,4000,0.005)
    np.set_printoptions(threshold=np.inf)

    print(classes[0][0])
    print(classes[0][1])
    print(classes[0][2])

    index = 1
    print(test_set_x.shape)
    plt.imshow(test_set_x[:, index].reshape((64, 64, 3)))
    print(d['Y_prediction_test'][0,index])
    print ("y = " + str(test_set_y[0,index]) + ", you predicted that it is a \""
           + classes[int(d["Y_prediction_test"][0,index])].decode("utf-8") +  "\" picture.")
    plt.show()

    # # Plot learning curve (with costs)
    costs = np.squeeze(d['costs'])
    print(costs)
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(d["learning_rate"]))
    plt.show()

    learning_rates = [0.01, 0.001, 0.0001,]
    models = {}
    for i in learning_rates:
        print("learning rate is: " + str(i))
        models[str(i)] = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=1500, learning_rate=i)
        print('\n' + "-------------------------------------------------------" + '\n')

    for i in learning_rates:
        plt.plot(np.squeeze(models[str(i)]["costs"]), label=str(models[str(i)]["learning_rate"]))
    plt.ylabel('cost')
    plt.xlabel('iterations')
    legend = plt.legend(shadow=True)
    frame = legend.get_frame()
    frame.set_facecolor('0.90')
    plt.show()

