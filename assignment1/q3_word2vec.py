#!/usr/bin/env python
# coding=utf-8

import numpy as np
import random

from q1_softmax import softmax
from q2_gradcheck import gradcheck_naive
from q2_sigmoid import sigmoid, sigmoid_grad


def normalize_rows(x):
    """
    Implement a function that normalizes each row of a matrix to have
    unit length.
    """
    return x / np.reshape(np.linalg.norm(x, axis=1), [x.shape[0], 1])


def test_normalize_rows():
    print "Testing normalizeRows..."
    x = normalize_rows(np.array([[3.0, 4.0], [1, 2]]))
    print x
    ans = np.array([[0.6,0.8],[0.4472136,0.89442719]])
    assert np.allclose(x, ans, rtol=1e-05, atol=1e-06)
    print ""


def softmax_cost_and_gradient(predicted, target, output_vectors, dataset):
    """ Softmax cost function for word2vec models

    Implement the cost and gradients for one predicted word vector
    and one target word vector as a building block for word2vec
    models, assuming the softmax prediction function and cross
    entropy loss.

    Arguments:
    predicted -- numpy ndarray, predicted word vector (\hat{v} in
                 the written component)

    target -- integer, the index of the target word
    output_vectors -- "output" vectors (as rows) for all tokens
    dataset -- needed for negative sampling, unused here.

    Return:
    cost -- cross entropy cost for the softmax word prediction
    grad_pred -- the gradient with respect to the predicted word
           vector
    grad -- the gradient with respect to all the other word
           vectors

    We will not provide starter code for this function, but feel
    free to reference the code you previously wrote for this
    assignment!
    """

    num, dim = output_vectors.shape

    # u ~ output_vectors ~ [N x D] / v ~ predicted ~ [D, ]
    score = np.dot(output_vectors, predicted)  # (N, )
    prob = softmax(score)  # (N, )
    cost = - np.log(prob[target])  # 1,

    # score (N x 1)= output_vectors (N x D) x predicted (D x 1)
    # doutput_vectors = dscore (N x 1) x predicted^T (1 x D)
    dscore = prob - np.eye(num)[target]  # (N, )
    grad = np.outer(dscore, predicted)
    grad_pred = np.dot(output_vectors.T, dscore)

    return cost, grad_pred, grad


def get_negative_samples(target, dataset, K):
    """ Samples K indexes which are not the target """

    indices = [None] * K
    for k in xrange(K):
        newidx = dataset.sample_token_idx()
        while newidx == target:
            newidx = dataset.sample_token_idx()
        indices[k] = newidx
    return indices


def neg_sampling_cost_and_gradient(predicted, target, output_vectors, dataset, K=10):
    """ Negative sampling cost function for word2vec models

    Implement the cost and gradients for one predicted word vector
    and one target word vector as a building block for word2vec
    models, using the negative sampling technique. K is the sample
    size.

    Note: See test_word2vec below for dataset's initialization.

    Arguments/Return Specifications: same as softmaxCostAndGradient
    """

    # Sampling of indices is done for you. Do not modify this if you
    # wish to match the autograder and receive points!
    indices = [target]
    indices.extend(get_negative_samples(target, dataset, K))

    ### YOUR CODE HERE
    _, dim = output_vectors.shape
    u = output_vectors[indices, :]  # (K+1, D)
    score = np.dot(u, np.reshape(predicted, [dim, 1]))  # (K+1, 1)
    prob = sigmoid(score)  # (K+1, 1)
    cost = - np.log(prob[0]) - np.sum(np.log(1 - prob[1:]))

    # sigmoid(-x) = 1 - sigmoid(x) 인거 사용하면 좋음
    # prob 에 대한 local gradient 는 구하기 쉬우므로, 거길 들렸다 가면 좋음
    dprob = np.reshape(np.append(np.array([- 1 / prob[0]]), 1 / (1 - prob[1:])), [K + 1, 1])  # (K+1, 1)
    dscore = sigmoid_grad(dprob)  # (K+1, 1)
    du = np.dot(dscore, np.reshape(predicted, [1, dim]))  # (K+1, D)
    dv = np.dot(u.T, dscore)
    grad = np.zeros_like(output_vectors)
    grad[indices, :] = du  # (N, D)
    grad_pred = dv
    ### END YOUR CODE

    return cost, grad_pred, grad


def skipgram(current_word, C, context_words, tokens, input_vectors, output_vectors,
             dataset, word2vec_cost_and_gradient=softmax_cost_and_gradient):
    """ Skip-gram model in word2vec

    Implement the skip-gram model in this function.

    Arguments:
    current_word -- a string of the current center word
    C -- integer, context size
    context_words -- list of no more than 2*C strings, the context words
    tokens -- a dictionary that maps words to their indices in
              the word vector list
    input_vectors -- "input" word vectors (as rows) for all tokens
    output_vectors -- "output" word vectors (as rows) for all tokens
    word2vec_cost_and_gradient -- the cost and gradient function for
                               a prediction vector given the target
                               word vectors, could be one of the two
                               cost functions you implemented above.

    Return:
    cost -- the cost function value for the skip-gram model
    grad -- the gradient with respect to the word vectors
    """

    cost = 0.0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)

    ### YOUR CODE HERE
    raise NotImplementedError
    ### END YOUR CODE

    return cost, gradIn, gradOut


def cbow(currentWord, C, contextWords, tokens, inputVectors, outputVectors,
         dataset, word2vecCostAndGradient=softmax_cost_and_gradient):
    """CBOW model in word2vec

    Implement the continuous bag-of-words model in this function.

    Arguments/Return specifications: same as the skip-gram model

    Extra credit: Implementing CBOW is optional, but the gradient
    derivations are not. If you decide not to implement CBOW, remove
    the NotImplementedError.
    """

    cost = 0.0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)

    ### YOUR CODE HERE
    raise NotImplementedError
    ### END YOUR CODE

    return cost, gradIn, gradOut


#############################################
# Testing functions below. DO NOT MODIFY!   #
#############################################

def word2vec_sgd_wrapper(word2vecModel, tokens, wordVectors, dataset, C,
                         word2vecCostAndGradient=softmax_cost_and_gradient):
    batchsize = 50
    cost = 0.0
    grad = np.zeros(wordVectors.shape)
    N = wordVectors.shape[0]
    inputVectors = wordVectors[:N/2,:]
    outputVectors = wordVectors[N/2:,:]
    for i in xrange(batchsize):
        C1 = random.randint(1,C)
        centerword, context = dataset.getRandomContext(C1)

        if word2vecModel == skipgram:
            denom = 1
        else:
            denom = 1

        c, gin, gout = word2vecModel(
            centerword, C1, context, tokens, inputVectors, outputVectors,
            dataset, word2vecCostAndGradient)
        cost += c / batchsize / denom
        grad[:N/2, :] += gin / batchsize / denom
        grad[N/2:, :] += gout / batchsize / denom

    return cost, grad


def test_word2vec():
    """ Interface to the dataset for negative sampling """
    dataset = type('dummy', (), {})()
    def dummySampleTokenIdx():
        return random.randint(0, 4)

    def getRandomContext(C):
        tokens = ["a", "b", "c", "d", "e"]
        return tokens[random.randint(0,4)], \
            [tokens[random.randint(0,4)] for i in xrange(2*C)]
    dataset.sample_token_idx = dummySampleTokenIdx
    dataset.getRandomContext = getRandomContext

    random.seed(31415)
    np.random.seed(9265)
    dummy_vectors = normalize_rows(np.random.randn(10, 3))
    dummy_tokens = dict([("a",0), ("b",1), ("c",2),("d",3),("e",4)])
    print "==== Gradient check for skip-gram ===="
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        skipgram, dummy_tokens, vec, dataset, 5, softmax_cost_and_gradient),
                    dummy_vectors)
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        skipgram, dummy_tokens, vec, dataset, 5, neg_sampling_cost_and_gradient),
                    dummy_vectors)
    print "\n==== Gradient check for CBOW      ===="
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        cbow, dummy_tokens, vec, dataset, 5, softmax_cost_and_gradient),
                    dummy_vectors)
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        cbow, dummy_tokens, vec, dataset, 5, neg_sampling_cost_and_gradient),
                    dummy_vectors)

    print "\n=== Results ==="
    print skipgram("c", 3, ["a", "b", "e", "d", "b", "c"],
        dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset)
    print skipgram("c", 1, ["a", "b"],
                   dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset,
                   neg_sampling_cost_and_gradient)
    print cbow("a", 2, ["a", "b", "c", "a"],
        dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset)
    print cbow("a", 2, ["a", "b", "a", "c"],
               dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset,
               neg_sampling_cost_and_gradient)


if __name__ == "__main__":
    test_normalize_rows()
    test_word2vec()
