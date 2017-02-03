# cluster_features.py
#
# Based on snippets here:
# http://scikit-learn.org/dev/auto_examples/cluster/plot_cluster_iris.html#sphx-glr-auto-examples-cluster-plot-cluster-iris-py
from __future__ import print_function

import time
import datetime

import numpy                as np
import pandas               as pd
import matplotlib.pyplot    as plt
from mpl_toolkits.mplot3d       import Axes3D
from sklearn.cluster            import KMeans
from sklearn.cross_validation   import train_test_split
from sklearn.preprocessing      import StandardScaler  
from sklearn.metrics            import accuracy_score
from sklearn.decomposition      import PCA

def get_payment_data(csv_filename):
    df = pd.read_csv(csv_filename, header = 0)

    # put the original column names in a python list
    feature_names = list(df.columns.values)

    # create a numpy array with the numeric values for input into scikit-learn
    numpy_array = df.as_matrix()

    Y = numpy_array[:,24]
    X = numpy_array[:, [i for i in xrange(np.shape(numpy_array)[1]-1)]]
    
    return (X, Y, feature_names)

if __name__ == "__main__":

    (X, Y, feature_names) = get_payment_data("default_on_payment.csv")
    print('Shape of the inputs: %s, shape of the labels: %s' % (str(X.shape), str(Y.shape)))

    # split into a training and testing set
    # Training instances: 22,500
    # Test instances: 7500
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42)
    print('Train set inputs: %s' % (str(X_train.shape)))
    print('Test set inputs %s'   % (str(X_test.shape)))
    print('Train set labels: %s' % (str(Y_train.shape)))
    print('Test set labels: %s'  % (str(Y_test.shape)))

    # ---------------------------------------------------------------------------
    # Scaling 
    # ----------------------------------------------------------------------------
    scaler = StandardScaler()  
    scaler.fit(X_train)  
    X_train = scaler.transform(X_train) 
    X_test = scaler.transform(X_test)

    # ---------------------------------------------------------------------------
    # PCA Transformation of Features
    # ----------------------------------------------------------------------------
    pca = PCA(n_components=3)
    X_train_new = pca.fit_transform(X_train, y=None)
    
    fig = plt.figure(1)
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    #ax.scatter(X_train_new[:, 0], X_train_new[:, 1], X_train_new[:, 2],c=labels.astype(np.float))
    ax.scatter(X_train_new[:, 0], X_train_new[:, 1], X_train_new[:, 2])


    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    ax.set_xlabel('PC0')
    ax.set_ylabel('PC1')
    ax.set_zlabel('PC2')

    plt.show()

    # ---------------------------------------------------------------------------
    # K-Means Clustering 
    # ----------------------------------------------------------------------------
    num_clusters = 2
    classifier_KMC = KMeans(n_clusters = num_clusters, n_jobs=-1, random_state=1)

    start_time = time.time()
    classifier_KMC.fit(X_train, y=None)
    end_time = time.time()

    labels1 = classifier_KMC.labels_

    # Classify the train and test set vectors
    train_labels = classifier_KMC.predict(X_train)
    test_labels = classifier_KMC.predict(X_test)

    # Returns 68.9% on training set
    accuracy_KMC_train = accuracy_score(Y_train, train_labels)
    accuracy_KMC_test = accuracy_score(Y_test, test_labels)




    # Plotting
    fig = plt.figure(1)
    plt.clf()
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    plt.cla()
    classifier_KMC.fit(X_train)
    labels = classifier_KMC.labels_

    #ax.scatter(X_train[:, 1], X_train[:, 2], X_train[:, 3], X_train[:, 3],c=labels.astype(np.float))
    ax.scatter(X_train[:, 0], X_train[:, 1],c=labels.astype(np.float))

    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    ax.set_xlabel('F0')
    ax.set_ylabel('F1')
    ax.set_zlabel('F2')

    plt.show()

    ## Plot the ground truth
    #fig = plt.figure(1)
    #plt.clf()
    #ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

    #plt.cla()



    # predictions_KMC   = classifier_KMC.predict(X_test)