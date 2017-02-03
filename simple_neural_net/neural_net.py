# neural_net.py

import numpy as np

def nonlin(x,deriv=False):
    ''' Nonlinear neuron activation function'''
    if (deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))


if __name__ == "__main__":

    # input dataset
    X = np.array([  [0,0,1],
                    [0,1,1],
                    [1,0,1],
                    [1,1,1] ])
    
    # output dataset (ground-truth labels)            
    y = np.array([[0,0,1,1]]).T

    # seed random numbers to make calculation deterministic
    np.random.seed(1)

    # initialize weights randomly with mean 0
    syn0 = 2*np.random.random((3,1)) - 1

    for iter in xrange(10000):

        # forward propagation
        layer_0 = X
        layer_1 = nonlin(np.dot(layer_0,syn0))

        # how much did we miss?
        layer_1_error = y - layer_1

        # multiply how much we missed by the 
        # slope of the sigmoid at the values in l1
        layer_1_delta = layer_1_error * nonlin(layer_1,True)

        # update weights
        syn0 += np.dot(layer_0.T, layer_1_delta)

    print "Output After Training:"
    print layer_1
    print 'Done!'