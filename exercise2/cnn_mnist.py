from __future__ import print_function

import argparse
import gzip
import json
import os
import pickle

import numpy as np
import tensorflow as tf


def one_hot(labels):
    """this creates a one hot encoding from a flat vector:
    i.e. given y = [0,2,1]
     it creates y_one_hot = [[1,0,0], [0,0,1], [0,1,0]]
    """
    classes = np.unique(labels)
    n_classes = classes.size
    one_hot_labels = np.zeros(labels.shape + (n_classes,))
    for c in classes:
        one_hot_labels[labels == c, c] = 1
    return one_hot_labels


def mnist(datasets_dir='./data'):
    if not os.path.exists(datasets_dir):
        os.mkdir(datasets_dir)
    data_file = os.path.join(datasets_dir, 'mnist.pkl.gz')
    if not os.path.exists(data_file):
        print('... downloading MNIST from the web')
        try:
            import urllib
            urllib.urlretrieve('http://google.com')
        except AttributeError:
            import urllib.request as urllib
        url = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        urllib.urlretrieve(url, data_file)

    print('... loading data')
    # Load the dataset
    f = gzip.open(data_file, 'rb')
    try:
        train_set, valid_set, test_set = pickle.load(f, encoding="latin1")
    except TypeError:
        train_set, valid_set, test_set = pickle.load(f)
    f.close()

    test_x, test_y = test_set
    test_x = test_x.astype('float32')
    test_x = test_x.astype('float32').reshape(test_x.shape[0], 28, 28, 1)
    test_y = test_y.astype('int32')
    valid_x, valid_y = valid_set
    valid_x = valid_x.astype('float32')
    valid_x = valid_x.astype('float32').reshape(valid_x.shape[0], 28, 28, 1)
    valid_y = valid_y.astype('int32')
    train_x, train_y = train_set
    train_x = train_x.astype('float32').reshape(train_x.shape[0], 28, 28, 1)
    train_y = train_y.astype('int32')
    print('... done loading data')
    return train_x, one_hot(train_y), valid_x, one_hot(valid_y), test_x, one_hot(test_y)


def model_fn(features, labels, mode, params):
    '''
    The model_fn is a parameter needed for tf.estimator.Estimator.
    It is a function that define the deep Learning model (its layout and output), as well as
    how to train it (the loss function and the optimizer).
    It returns a EstimatorSpec which is able to handle different modes (TRAIN | EVAL | PREDICT).
    '''
    # Layer 1: Conv (ReLU) + Pooling (2x2)
    conv1 = tf.layers.conv2d(
            inputs=features['x'],
            filters=params.num_filters,
            kernel_size=[params.filter_size, params.filter_size],
            padding="same",
            activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Layer 2: Conv (ReLU) + Pooling (2x2)
    conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=params.num_filters,
            kernel_size=[params.filter_size, params.filter_size],
            padding="same",
            activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Layer 3: Dense (128 units)
    shape = pool2.get_shape().as_list()
    pool2_flat = tf.reshape(pool2, [-1, shape[1] * shape[2] * shape[3]])
    dense = tf.layers.dense(inputs=pool2_flat, units=128)

    # Output Layer
    y_hat = tf.layers.dense(inputs=dense, units=10)
    y_hat_label = tf.argmax(y_hat, axis=1)

    # Loss (Cross-Entropy), Optimizer (SGD)
    labels=tf.argmax(labels, axis=1)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=y_hat)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=params.lr)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(labels=labels, predictions=y_hat_label)}

    return tf.estimator.EstimatorSpec(
            mode,
            predictions=y_hat_label,         # for mode = PREDICT
            loss=loss,                       # for mode = TRAIN, EVAL
            train_op=train_op,               # for mode = TRAIN
            eval_metric_ops=eval_metric_ops) # for mode = EVAL


def train_and_validate(x_train, y_train, x_valid, y_valid, num_epochs, lr, num_filters, batch_size, filter_size):
    '''
    A function to train and evaluate the CNN with given parameters.
    The final model and the evaluation errors during each epoch of training are returned.
    '''
    params = tf.contrib.training.HParams(
            lr=lr,
            num_filters=num_filters,
            filter_size=filter_size)
    model = tf.estimator.Estimator(model_fn, params=params)

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'x':x_train},
            y=y_train,
            batch_size=batch_size,
            num_epochs=None,
            shuffle=True)
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'x':x_valid},
            y=y_valid,
            num_epochs=1,
            shuffle=False)

    # Make the number of steps per training to be number of training data / batch_size,
    # so that in each training epoch, the entire training data would be seen once.
    num_steps = np.ceil(x_train.shape[0] / batch_size)
    learning_curve = np.zeros(num_epochs)

    # Loop the number of epochs here so that we can record the evaluation loss for each epoch.
    for e in range(num_epochs):
        model.train(input_fn=train_input_fn, steps=num_steps)
        eval_results = model.evaluate(input_fn=eval_input_fn)
        learning_curve[e] = eval_results['loss']

    return learning_curve.tolist(), model


def test(x_test, y_test, model):
    '''
    Use the given trained model to test on the test data and return the test error.
    '''
    test_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'x':x_test},
            y=y_test,
            num_epochs=1,
            shuffle=False)

    # Since we need to return the test error, use evaluate instead of predict.
    test_results = model.evaluate(input_fn=test_input_fn)
    print("Test Accuracy:", test_results['accuracy'])

    return float(test_results['loss'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", default="./", type=str, nargs="?",
                        help="Path where the results will be stored")
    parser.add_argument("--input_path", default="./", type=str, nargs="?",
                        help="Path where the data is located. If the data is not available it will be downloaded first")
    parser.add_argument("--learning_rate", default=1e-3, type=float, nargs="?", help="Learning rate for SGD")
    parser.add_argument("--num_filters", default=32, type=int, nargs="?",
                        help="The number of filters for each convolution layer")
    parser.add_argument("--batch_size", default=128, type=int, nargs="?", help="Batch size for SGD")
    parser.add_argument("--epochs", default=12, type=int, nargs="?",
                        help="Determines how many epochs the network will be trained")
    parser.add_argument("--run_id", default=0, type=int, nargs="?",
                        help="Helps to identify different runs of an experiments")
    parser.add_argument("--filter_size", default=3, type=int, nargs="?",
                        help="Filter width and height")
    args = parser.parse_args()

    # hyperparameters
    lr = args.learning_rate
    num_filters = args.num_filters
    batch_size = args.batch_size
    epochs = args.epochs
    filter_size = args.filter_size

    # train and test convolutional neural network
    x_train, y_train, x_valid, y_valid, x_test, y_test = mnist(args.input_path)

    learning_curve, model = train_and_validate(x_train, y_train, x_valid, y_valid, epochs, lr, num_filters, batch_size, filter_size)

    test_error = test(x_test, y_test, model)

    # save results in a dictionary and write them into a .json file
    results = dict()
    results["lr"] = lr
    results["num_filters"] = num_filters
    results["batch_size"] = batch_size
    results["filter_size"] = filter_size
    results["learning_curve"] = learning_curve
    results["test_error"] = test_error

    path = os.path.join(args.output_path, "results")
    os.makedirs(path, exist_ok=True)

    fname = os.path.join(path, "results_run_%d.json" % args.run_id)

    fh = open(fname, "w")
    json.dump(results, fh)
    fh.close()
