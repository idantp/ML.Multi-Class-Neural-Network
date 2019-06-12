# Author: Idan Twito
# ID: 311125249

import numpy as np

# relu = lambda x: np.maximum(x, 0, x)

def relu(vec):
    return np.maximum(vec,0)


# todo - delete.
def get_accuracy(W1, b1, W2, b2, x_valid, y_valid):
    true = 0
    for x, y in zip(x_valid, y_valid):
        x = np.reshape(x, (1, 784))
        y_hat, z1, z2, h1 = forward_propagation(W1, b1, W2, b2, x, y)
        max_y = y_hat.argmax(axis=0)
        if max_y[0] == int(y):
            true += 1
    return true / float(len(y_valid))


def update_weights(W1, W2, b1, b2, W1_grad, W2_grad, b1_grad, b2_grad):
    eta = 0.02
    W1 = W1 - (eta * W1_grad)
    W2 = W2 - (eta * W2_grad)
    b1 = b1 - (eta * b1_grad)
    b2 = b2 - (eta * b2_grad)
    return W1, W2, b1, b2


def get_file(file_path):
    return np.loadtxt(file_path)


def softmax(z2):
    denominator = np.sum(np.exp(z2), axis=0)
    return np.exp(z2) / denominator


def forward_propagation(W1, b1, W2, b2, x, y):
    x_temp = np.reshape(x, (-1, 1))
    z1 = np.dot(W1, x_temp) + b1
    h1 = relu(z1)
    # todo - what is this if? is it needed? is it normalization?
    if h1.max() != 0:
        h1 /= h1.max()
    z2 = np.dot(W2, h1) + b2
    y_hat = softmax(z2)
    return y_hat, z1, z2, h1


def b1_gradient_calc(w2, h2, z1):
    return np.dot(np.transpose(w2), h2) * z1


def w1_gradient_calc(b1_gradient, x):
    return np.dot(b1_gradient, x)


def w2_gradient_calc(h2, h1):
    return np.dot(h2, np.transpose(h1))


def backward_propagation(y_hat, z1, z2, h1, W1, b1, W2, b2, x, y):
    h2 = y_hat
    correct_answer = int(y)
    h2[correct_answer] = h2[correct_answer] - 1
    b2_gradient = h2
    # relu Derivative
    z1[z1 <= 0] = 0
    z1[z1 > 0] = 1
    b1_gradient = b1_gradient_calc(W2, h2, z1)
    w1_gradient = w1_gradient_calc(b1_gradient, x)
    w2_gradient = w2_gradient_calc(h2, h1)
    return w1_gradient, w2_gradient, b1_gradient, b2_gradient


def train_func(W1, b1, W2, b2, epochs, train_x, train_y, valid_x, valid_y):
    train_x /= 255
    valid_x /= 255
    for i in range(epochs):
        arr_zipped = list(zip(train_x, train_y))
        np.random.shuffle(arr_zipped)
        train_x, train_y = zip(*arr_zipped)
        for x, y in zip(train_x, train_y):
            x = np.reshape(x, (1, 784))
            y_hat, z1, z2, h1 = forward_propagation(W1, b1, W2, b2, x, y)
            W1_grad, W2_grad, b1_grad, b2_grad = backward_propagation(y_hat, z1, z2, h1, W1, b1, W2, b2, x, y)
            W1, W2, b1, b2 = update_weights(W1, W2, b1, b2,W1_grad, W2_grad, b1_grad, b2_grad)
        acc = get_accuracy(W1, b1, W2, b2, valid_x, valid_y)
        print(acc * 100)
    return W1,  W2, b1, b2


def main():
    epochs = 20
    max = 0.05
    min = -0.05
    hidden_layers = 142
    # todo - uniform - random numbers between to numbers.
    W1 = np.random.uniform(min, max, [hidden_layers, 784])
    W2 = np.random.uniform(min, max, [10, hidden_layers])
    b1 = np.random.uniform(min, max, [hidden_layers, 1])
    b2 = np.random.uniform(min, max, [10, 1])
    x_training_set = get_file("train_x.txt")
    y_training_set = get_file("train_y.txt")
    x_testing_set = get_file("test_x.txt")
    train_zipped = list(zip(x_training_set, y_training_set))
    np.random.shuffle(train_zipped)
    training_size = x_training_set.shape[0]
    validation_size = (int)(training_size * (1 / 6))
    valid_x, valid_y = x_training_set[-validation_size:, :], y_training_set[-validation_size:]
    x_training_set, y_training_set = x_training_set[:-validation_size, :], y_training_set[:-validation_size]
    train_func(W1, b1, W2, b2, epochs, x_training_set, y_training_set, valid_x, valid_y)


if __name__ == "__main__":
    main()