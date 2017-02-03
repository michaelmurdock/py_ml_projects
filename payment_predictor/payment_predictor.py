# payment_predictor.py
#
# Based on snippets here:
# http://stackoverflow.com/questions/11023411/how-to-import-csv-data-file-into-scikit-learn
from __future__ import print_function

import time
import datetime

import numpy as np
import pandas as pd

from sklearn.cross_validation import train_test_split
#from sklearn.cross_validation import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import make_hastie_10_2
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.cluster import KMeans

from sklearn.preprocessing import StandardScaler  


from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

import my_cm


def get_payment_data(csv_filename):
    '''
    '''
    #input_file = "default_on_payment.csv"

    # comma delimited is the default
    df = pd.read_csv(csv_filename, header = 0)

    # put the original column names in a python list
    feature_names = list(df.columns.values)

    # remove the non-numeric columns
    #df = df._get_numeric_data()

    # put the numeric column names in a python list
    #numeric_headers = list(df.columns.values)

    # create a numpy array with the numeric values for input into scikit-learn
    numpy_array = df.as_matrix()
    #print(numpy_array.shape)

    Y = numpy_array[:,24]
    #Y = numpy_array[:,24].astype(float)
    #print(y.shape)

    X = numpy_array[:, [i for i in xrange(np.shape(numpy_array)[1]-1)]]
    #print(X.shape)

    return (X, Y, feature_names)

if __name__ == "__main__":

    DO_SCALING                  = True
    DO_K_MEANS_CLUSTERING       = False
    DO_LOGISTIC_REGRESSION      = False
    DO_ADABOOST                 = False
    DO_GRADIENT_BOOSTING        = False
    DO_GAUSSIAN_NAIVE_BAYES     = False
    DO_MLP                      = True
    DO_SVM                      = False
    DO_LINEAR_SVM               = False




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


    my_labels = ['Payment Late', 'Payment On-Time']

    # ---------------------------------------------------------
    # Scaling the input features
    # http://scikit-learn.org/dev/modules/neural_networks_supervised.html#mlp-tips
    # Multi-layer Perceptron is sensitive to feature scaling, so it is highly recommended to scale your data. 
    # For example, scale each attribute on the input vector X to [0, 1] or [-1, +1], or standardize it to have 
    # mean 0 and variance 1. Note that you must apply the same scaling to the test set for meaningful results. 
    # You can use StandardScaler for standardization.
    # ---------------------------------------------------------

    if DO_SCALING:
        scaler = StandardScaler()  
    
        # Don't cheat - fit only on training data
        scaler.fit(X_train)  
        X_train = scaler.transform(X_train)  
    
        # apply same transformation to test data
        X_test = scaler.transform(X_test)  

    # ---------------------------------------------------------
    # K-Means Clustering
    # ---------------------------------------------------------
    if DO_K_MEANS_CLUSTERING:
        num_clusters = 2
        classifier_KMC = KMeans(n_clusters = num_clusters, n_jobs=-1, random_state=1)

        start_time = time.time()
        classifier_KMC.fit(X_train, y=None)
        end_time = time.time()

        predictions_KMC   = classifier_KMC.predict(X_test)


    # ---------------------------------------------------------
    # Gaussian Naive Bayes
    # ---------------------------------------------------------
    if DO_GAUSSIAN_NAIVE_BAYES:
        classifier_GNB = GaussianNB()

        start_time = time.time()
        classifier_GNB.fit(X_train, Y_train)
        end_time = time.time()

        predictions_GNB   = classifier_GNB.predict(X_test)
        #probabilities_SVM = classifier_GNB.predict_proba(X_test)

        accuracy_GNB = accuracy_score(Y_test, predictions_GNB)
        print('Model: %s' % ('G. Naive Bayes'))
        print('Accuracy for GNB classifier: %f' % (accuracy_GNB))
        print('Time to train: %s' % (datetime.timedelta(seconds=(end_time - start_time))))
        print(classification_report(Y_test, predictions_GNB))
        cm = confusion_matrix(Y_test, predictions_GNB)
        my_cm.show_confusion_matrix(cm, class_labels = my_labels)


    # ---------------------------------------------------------
    # Linear Regression classifier
    # ---------------------------------------------------------
    #classifier_LinR = LinearRegression()
    #classifier_LinR.fit(X_train, Y_train)
    #predictions_LinR = classifier_LinR.predict(X_test)
    
    #accuracy_LinR = accuracy_score(Y_test, predictions_LinR)
    #print('Accuracy for LinearRegression classifier: %f' % (accuracy_LinR))
    #print(classification_report(Y_test, predictions_LinR))


    # ---------------------------------------------------------
    # Logistic Regression classifier
    # ---------------------------------------------------------
    if DO_LOGISTIC_REGRESSION:
        classifier_LogR = LogisticRegression()
        start_time = time.time()
        classifier_LogR.fit(X_train, Y_train)
        end_time = time.time()
        predictions_LogR   = classifier_LogR.predict(X_test)
        probabilities_LogR = classifier_LogR.predict_proba(X_test)
        accuracy_LogR = accuracy_score(Y_test, predictions_LogR)
        print('Model: %s' % ('Logistic Regression'))
        print('Accuracy: %f' % (accuracy_LogR))
        print('Time to train: %s' % (datetime.timedelta(seconds=(end_time - start_time))))
        print(classification_report(Y_test, predictions_LogR))
        cm = confusion_matrix(Y_test, predictions_LogR)
        my_cm.show_confusion_matrix(cm, class_labels = my_labels)

    # ---------------------------------------------------------
    # AdaBoost Classifier
    # ---------------------------------------------------------
    if DO_ADABOOST:
        classifier_AB = AdaBoostClassifier(n_estimators=25)
        start_time = time.time()
        classifier_AB.fit(X_train, Y_train)
        end_time = time.time()
        # scores = cross_val_score(classifier_AB, X_train, Y_train)
        # scores.mean()
        predictions_AB = classifier_AB.predict(X_test)
        accuracy_AB = accuracy_score(Y_test, predictions_AB)
        print('Model: %s' % ('AdaBoost Classifier'))
        #print('Score for AB classifier: %f' % (score))
        print('Accuracy for AB classifier: %f' % (accuracy_AB))
        print('Time to train: %s' % (datetime.timedelta(seconds=(end_time - start_time))))
        print(classification_report(Y_test, predictions_AB))


    # ---------------------------------------------------------
    # Gradient Boosting Classifier
    # ---------------------------------------------------------
    if DO_GRADIENT_BOOSTING:
        #X, y = make_hastie_10_2(random_state=0)
        #X_train, X_test = X[:2000], X[2000:]
        #Y_train, Y_test = y[:2000], y[2000:]

        start_time = time.time()
        classifier_GB = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0).fit(X_train, Y_train)
        end_time = time.time()

        score = classifier_GB.score(X_test, Y_test) 
        predictions_GB   = classifier_GB.predict(X_test)
        accuracy_GB = accuracy_score(Y_test, predictions_GB)        
        print('Model: %s' % ('Gradient Boosting Classifier'))
        print('Score for GB classifier: %f' % (score))
        print('Accuracy for GB classifier: %f' % (accuracy_GB))
        print('Time to train: %s' % (datetime.timedelta(seconds=(end_time - start_time))))
        print(classification_report(Y_test, predictions_GB))


    # ---------------------------------------------------------
    # Gaussian Naive Bayes
    # ---------------------------------------------------------
    #if DO_GAUSSIAN_NAIVE_BAYES:
    #    classifier_GNB = GaussianNB()

    #    start_time = time.time()
    #    classifier_GNB.fit(X_train, Y_train)
    #    end_time = time.time()

    #    predictions_GNB   = classifier_GNB.predict(X_test)
    #    #probabilities_SVM = classifier_GNB.predict_proba(X_test)

    #    accuracy_GNB = accuracy_score(Y_test, predictions_GNB)
    #    print('Model: %s' % ('G. Naive Bayes'))
    #    print('Accuracy for GNB classifier: %f' % (accuracy_GNB))
    #    print('Time to train: %s' % (datetime.timedelta(seconds=(end_time - start_time))))
    #    print(classification_report(Y_test, predictions_GNB))

    # ---------------------------------------------------------
    # Linear Support Vector Machine
    # ---------------------------------------------------------
    if DO_LINEAR_SVM:

        classifier_LSVM = svm.LinearSVC()

        start_time = time.time()
        classifier_LSVM.fit(X_train, Y_train)
        end_time = time.time()

        predictions_LSVM   = classifier_LSVM.predict(X_test)
        #probabilities_SVM = classifier_SVM.predict_proba(X_test)

        accuracy_LSVM = accuracy_score(Y_test, predictions_LSVM)
        print('Model: %s' % ('Linear SVM'))
        print('Accuracy for LSVM classifier: %f' % (accuracy_LSVM))
        print('Time to train: %s' % (datetime.timedelta(seconds=(end_time - start_time))))
        print(classification_report(Y_test, predictions_LSVM))


    # ---------------------------------------------------------
    # Support Vector Machine
    # ---------------------------------------------------------
    if DO_SVM:

        classifier_SVM = svm.SVC()

        start_time = time.time()
        classifier_SVM.fit(X_train, Y_train)
        end_time = time.time()

        predictions_SVM   = classifier_SVM.predict(X_test)
        #probabilities_SVM = classifier_SVM.predict_proba(X_test)

        accuracy_SVM = accuracy_score(Y_test, predictions_SVM)
        print('Model: %s' % ('SVM'))
        print('Accuracy for SVM classifier: %f' % (accuracy_SVM))
        print('Time to train: %s' % (datetime.timedelta(seconds=(end_time - start_time))))
        print(classification_report(Y_test, predictions_SVM))

    # ---------------------------------------------------------
    # MLP classifier
    # ---------------------------------------------------------
    if DO_MLP:

        num_hidden_nodes = 8
        classifier_MLP = MLPClassifier(algorithm='l-bfgs', alpha=1e-5, hidden_layer_sizes=(8, 2), random_state=1)
        #classifier_MLP = MLPClassifier(activation='logistic', algorithm='l-bfgs', alpha=1e-5, hidden_layer_sizes=(num_hidden_nodes,), random_state=1)
        #classifier_MLP = MLPClassifier(algorithm='adam', alpha=1e-5, hidden_layer_sizes=(num_hidden_nodes,), random_state=1)
        #classifier_MLP = MLPClassifier(algorithm='sgd', alpha=1e-5, hidden_layer_sizes=(num_hidden_nodes,), random_state=1)

        start_time = time.time()
        classifier_MLP.fit(X_train, Y_train)
        end_time = time.time()

        predictions_MLP   = classifier_MLP.predict(X_test)
        probabilities_MLP = classifier_MLP.predict_proba(X_test)

        accuracy_MLP = accuracy_score(Y_test, predictions_MLP)
        print('Model: %s' % ('MLP'))
        print('Num hidden nodes: %d' % (num_hidden_nodes))
        print('Accuracy for MLP classifier: %f' % (accuracy_MLP))
        print('Time to train: %s' % (datetime.timedelta(seconds=(end_time - start_time))))
        print(classification_report(Y_test, predictions_MLP))
        cm = confusion_matrix(Y_test, predictions_MLP)
        my_cm.show_confusion_matrix(cm, class_labels = my_labels)



    #for i, prediction in enumerate(predictions[:25]):
    #    print('Prediction: %d. Ground-Truth: %d' % (prediction, y_test[i]))

    # 78.31%
    accuracy_LinR = accuracy_score(Y_test, predictions1)
    print('Accuracy for LinearRegression classifier: %f' % (accuracy_LinR))
    print(classification_report(Y_test, predictions_LinR))

    print('Cross-Validation Scores')
    scores = cross_val_score(classifier, X_train, y_train, cv=5)
    print(np.mean(scores), scores)

    print('Cross-Validation scores using precision')
    precisions = cross_val_score(classifier, X_train, y_train, cv=5, scoring='precision')
    print('Precision', np.mean(precisions), precisions)

    print('Cross-Validation scores using recall')
    recalls = cross_val_score(classifier, X_train, y_train, cv=5, scoring='recall')
    print('Recalls', np.mean(recalls), recalls)


    confusion_matrix = confusion_matrix(y_test, predictions)
    print(confusion_matrix)
    plt.matshow(confusion_matrix)
    plt.title('Confusion matrix')
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


    print('Done.')

    # -------------------- Results ------------------------
    # Accuracy: 0.783067
    #C:\Users\mmurdock\AppData\Local\Continuum\Anaconda\lib\site-packages\sklearn\metrics\classification.py:958:
    # UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.
    # 'precision', 'predicted', average, warn_for)

    #             precision    recall  f1-score   support

    #          0       0.78      1.00      0.88      5873
    #          1       0.00      0.00      0.00      1627

    #avg / total       0.61      0.78      0.69      7500

    #Cross-Validation Scores
    #0.777288894369 [ 0.7773828   0.77711111  0.77711111  0.77733333  0.77750611]

    #Cross-Validation scores using precision
    #C:\Users\mmurdock\AppData\Local\Continuum\Anaconda\lib\site-packages\sklearn\metrics\classification.py:958: 
    #    UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 due to no predicted samples.
    #'precision', 'predicted', average, warn_for)
    #Precision 0.0 [ 0.  0.  0.  0.  0.]
    #Cross-Validation scores using recall
    #Recalls 0.0 [ 0.  0.  0.  0.  0.]




    # reverse the order of the columns
    #numeric_headers.reverse()
    #reverse_df = df[numeric_headers]

    ## write the reverse_df to an excel spreadsheet
    #reverse_df.to_excel('path_to_file.xls')

  


    # First, we load the .csv file using pandas and split the data set into training and
    # test sets. By default, train_test_split() assigns 75 percent of the samples to the
    # training set and allocates the remaining 25 percent of the samples to the test set:
    #df = pd.read_csv('data/SMSSpamCollection', delimiter='\t', header=None)
    #X_train_raw, X_test_raw, y_train, y_test = train_test_split(df[1], df[0])

   
    
    #for i, prediction in enumerate(predictions[:5]):
      