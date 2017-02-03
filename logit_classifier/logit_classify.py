# logit_classify.py
# This code snippet came from sklearn_classifier.py in the andarlib_service project
from __future__ import print_function
from __future__ import division


import pickle
import numpy as np
#import numpy.linalg as linalg
import sklearn.linear_model



def get_classifier_info(classifier):
    '''
    Returns a dictionary with info about this LR model
    '''
    d = {}
    d['num_classes'] = len(classifier.coef_)
    d['num_inputs'] = len(classifier.coef_.T)
    d['num_training_iters'] = classifier.n_iter_
    d['coefficients'] = classifier.coef_
    d['coefficients_transpose'] = classifier.coef_.T
    d['intercept'] = classifier.intercept_
    
    return d

def read_classifier(classifier_model_filename):
    classifier, label_encoder = pickle.load(open(classifier_model_filename, 'rb'))
    return classifier, label_encoder


def read_classifier_safe(classifier_model_filename):
    '''
    Reads the LogisticRegression classifier and the LabelEncoder from the 
    specified classifier_model_filename.
    Returns: (result_flag, err_msg, (classifier, label_encoder))

    '''
    try:
        classifier, label_encoder = pickle.load(open(classifier_model_filename, 'rb'))
    except Exception as e:
        return (False, 'Exception: %s' % (str(e)), None)
    return (True, '', (classifier, label_encoder))


def augment_coef_matrix(coefs_matrix, intercept_array):
    (num_arrays, num_elements) = coefs_matrix.shape
    new_coeffs_matrix = np.append(coefs_matrix, intercept_array)

    #foo = intercept_array.shape
    for i in range(num_arrays):
        intercept_value = intercept_array[i]
        foo = np.append(coefs_matrix[i], intercept_value)
        #coefs_matrix[i] = np.append(coefs_matrix[i], intercept_value)

    return coefs_matrix
    
     
def softmax(x):
    '''
    Compute softmax values for each sets of scores in x.
    '''
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def logistic_calculator(input_array, coeff_array, intercept_array):
    '''
    Demonstrates how to use the LR classifier to process an input vecotr, input_list.

    Note that the length of input_list must be 4096 to match the length of the 
    model's coefficient vector.

    This classifier's coef array has 9 elements, each with 4096 floats
    The transpose of the coef array, classifer.coef_.T,  has 4096 elements, each with 9 floats
    The intercept array, classifier.intercept_, has 9 floats
    '''
    # Append 1.0 to the input_array so the dot-product can be done against coefficients, 
    # which includes the intercept term.
    # augmented array has 4097 terms
    # input_array only has 4096 (why?!)
    augmented_inputs = np.append(input_array, 1.0)
    augmented_inputs = input_array

    # compute dot-product of input vector and coefficients
    dot_product = np.dot(augmented_inputs, coeff_array)

    # Add the offset
    dot_product_with_offset = dot_product + intercept_array
    
    # Run through the sigmoid
    # output = sigmoid(dot_product_with_offset)

    # Run through the logit
    output = logit(dot_product_with_offset)

    return output


def logit(X):
    '''
    Computes the inverse of the logistic (sigmoid) function for ndarray, X
    '''
    numer = sigmoid(X)
    denom = 1.0 - numer
    combined = numer / denom
    nat_log = np.log(combined)
    return nat_log

def sigmoid(X):
    return 1.0 / (1.0 + np.exp(-X))

def sigmoid_orig(X):
    return 1.0 / (1.0 + np.exp(-1.0 * X))
    

def cosine_distance(v1, v2):
    return 1.0 - (np.dot(v1, v2) / ( np.linalg.norm(v1) * np.linalg.norm(v2)))
  
def euclidean_distance(v1, v2):
  return np.sqrt(np.sum((v1-v2)**2))

def get_winning_class(class_probabilities):
    '''
    class_probabilities is a list of tuples (label, probability), with one element per class.
    RETURNS: tuple: (label, score), corresponding to the class with the highest score.
    '''
    max_score = 0.0
    for item in class_probabilities:
        if item[1] > max_score:
            winner = item
            max_score = item[1]
    return winner

def classify_this_feature_vector(feat_filename, classifier_model_filename):
    '''
    '''
    # Load the classifier model
    classifier, label_encoder = pickle.load(open(classifier_model_filename, 'rb'))

    # Create the list of class labels
    labels = label_encoder.classes_.tolist()

    # Read feature vector
    features = np.loadtxt(feat_filename)

    # Process this feature vector with our classifier 
    raw_probabilities = classifier.predict_proba(features)
    xprobabilities = raw_probabilities[0].tolist()
    
    # Create the list of (class_label, prob) tuples
    computed_class_probabilities = []
    for idx in xrange(0, len(xprobabilities)):
        computed_class_probabilities.append((labels[idx], xprobabilities[idx]))
        
    (winning_class_label, winning_class_score) = get_winning_class(computed_class_probabilities)

    return (winning_class_label, winning_class_score)


if __name__ == "__main__":

    # ---------------------------------------------------------------------------------
    # Feature Vectors
    # ---------------------------------------------------------------------------------

    # 10 feature vectors, each with 4096 elements
    feature_vectors = np.loadtxt('feats_4096_10.txt')
    print(feature_vectors.shape)

    first_feature_vector = feature_vectors[0]
    second_feature_vector = feature_vectors[1]

    cos_dist = cosine_distance(first_feature_vector, second_feature_vector)
    print('cosine distance: %f' % (cos_dist))

    euc_dist = euclidean_distance(first_feature_vector, second_feature_vector)
    print('Euclidiean distance: %f' % (euc_dist))


    # 1 feature vector with 4096 elements
    feature_vector = np.loadtxt('feats_4096.txt')
    print(feature_vector.shape)
    
    # ---------------------------------------------------------------------------------
    # Classifier
    # ---------------------------------------------------------------------------------

    # Read in a classifier, c - Trained to predict probablity for each of 9 classes  
    (c, label_encoder) = read_classifier('latest_model.pckl')

    print('Array shapes:')

    # Transpose of coefficients: (4096L, 9L)
    print('Transpose of coefficients: ' + str(c.coef_.T.shape))

    # Intercept: (9L,)
    print('Intercept: ' + str(c.intercept_.shape))

    # Result from dot-product: (9L,)
    print('Result from dot-product: ' + str(np.dot(feature_vector, c.coef_.T).shape))
    
    # The confidence score for a sample is the signed distance of that sample to the hyperplane.
    # [-4.66011321 -6.86733935 -3.85433604 -2.3590917  -5.26464047 -6.94190295 -6.36456544 -6.33818804 -6.62427644]
    output = c.decision_function(feature_vector)
    print(output)

    # Get the probabilities using the trained classifier's predict method
    # [ 0.07300987  0.00809907  0.16155188  0.67231963  0.04005808  0.0075177 0.0133811   0.01373812  0.01032456]
    probs1 = c.predict_proba(feature_vector)
    print(probs1)

   
    # Get the probabilities using the weights and offsets:
    # [ 0.00937664  0.00104016  0.02074806  0.08634582  0.00514465  0.0009655 0.00171853  0.00176438  0.00132598]
    probs2 = sigmoid(np.dot(feature_vector, c.coef_.T) + c.intercept_)
    print(probs2)

    # Computing the softmax on the probs2 vector
    # [ 0.11053004  0.10961244  0.1117941   0.11937342  0.11006327  0.10960426 0.10968683  0.10969186  0.10964378]
    probs3 = softmax(probs2)
    print(probs3)



    #foo = augment_coef_matrix(c.coef_.T, c.intercept_)



    # Features - 1 feature vector with 4096 elements
    feature_vector = np.loadtxt('feats_4096.txt')
    print(feature_vector.shape)
    
    # classifier, c - Trained to predict probablity for each of 9 classes  
    (c, label_encoder) = read_classifier('latest_model.pckl')
    
    # Getting the probabilities using the trained classifier's predict method
    probs1 = c.predict_proba(feature_vector)
    print(probs1)

  


    # Distance measures
    v1 = [1.0,  0.0, 0.0]
    v2 = [0.0,  1.0, 0.0]
    v3 = [1.0,  0.0, 0.0]
    v4 = [-1.0, 0.0, 0.0]

    cos_dist2 = cosine_distance(v1, v2)
    print(cos_dist2)

    cos_dist3 = cosine_distance(v1, v3)
    print(cos_dist3)

    cos_dist4 = cosine_distance(v1, v4)
    print(cos_dist4)


    # Features - 1 feature vector with 4096 elements
    feature_filename = r'C:\pyDev\__My Scripts\logit_classify\feats_4096.txt'
    feature_vector = np.loadtxt(feature_filename)
    
    # classifier_model_filename  
    classifier_model_filename = r'C:\pyDev\__My Scripts\logit_classify\latest_model.pckl'
    
    # Classify this feature vector
    (winning_class_label, winning_class_score) = classify_this_feature_vector(feature_filename, classifier_model_filename)
    print('Class label: %s, Score: %s' % (str(winning_class_label), str(winning_class_score)))

    # Read the classifier model from the pickle file
    (results, errmsg, (classifier, label_encoder)) = read_classifier(classifier_model_filename)
    if not results:
        print('Error in read_classifier. Details: %s' % errmsg)
        exit(0)

    # Get the dictionary with the info about this classifier
    d = get_classifier_info(classifier)

    coeff_array = d['coefficients_transpose'] 
    intercept_array = d['intercept']

    # Process this feature_vector with the model parameters to generate an output vector
    output = logistic_calculator(feature_vector, coeff_array, intercept_array)

    print(output)

    #LogisticRegression( C=1.0, 
    #                   class_weight=None, 
    #                   dual=False, 
    #                   fit_intercept=True,
    #                   intercept_scaling=1, 
    #                   max_iter=100, 
    #                   multi_class='ovr',
    #                   penalty='l2', 
    #                   random_state=None, 
    #                   solver='liblinear', 
    #                   tol=0.0001,
    #                   verbose=0)

 