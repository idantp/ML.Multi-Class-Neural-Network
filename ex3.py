import numpy as np


# Relu calculation in accordance to what was learned in class.
def relu(vec):
    return np.maximum(vec, 0)


# Updates the weights and biases using the learning rate we chose, for each iteration.
def update_weights(W1, W2, b1, b2, W1_grad, W2_grad, b1_grad, b2_grad):
    eta = 0.02
    W1 = W1 - (eta * W1_grad)
    W2 = W2 - (eta * W2_grad)
    b1 = b1 - (eta * b1_grad)
    b2 = b2 - (eta * b2_grad)
    return W1, W2, b1, b2


# Returns what was read from the file as an nd array.
def get_file(file_path):
    return np.loadtxt(file_path)


# Softmax function calculation, according to what we learned in class.
def softmax(z2):
    denominator = np.sum(np.exp(z2), axis=0)
    return np.exp(z2) / denominator


# Forward Propagation.
def forward_propagation(W1, b1, W2, b2, x):
    # Change the layout of the vector in order to properly multiply with the matrix.
    x_temp = np.reshape(x, (-1, 1))
    z1 = np.dot(W1, x_temp) + b1
    # Call the Activation function ReLU
    h1 = relu(z1)
    # Normalize the result from ReLU
    if h1.max() != 0:
        h1 /= h1.max()
    # Multiply it by the weights and add the bias.
    z2 = np.dot(W2, h1) + b2
    y_hat = softmax(z2)
    return y_hat, z1, z2, h1


# Returns the gradient of the first bias.
def b1_gradient_calc(w2, h2, z1):
    return np.dot(np.transpose(w2), h2) * z1


# Returns the gradient of the first weights matrix.
def w1_gradient_calc(b1_gradient, x):
    return np.dot(b1_gradient, x)


# Returns the gradient of the second wieghts matrix.
def w2_gradient_calc(h2, h1):
    return np.dot(h2, np.transpose(h1))


# Backwards Propagation function.
def backward_propagation(y_hat, z1, h1, W2, x, y):
    h2 = y_hat
    correct_answer = int(y)
    # Derivative of H2
    h2[correct_answer] = h2[correct_answer] - 1
    b2_gradient = h2
    # Calculate the ReLU Derivative
    z1[z1 <= 0] = 0
    z1[z1 > 0] = 1
    # Calculate the necessary gradients.
    b1_gradient = b1_gradient_calc(W2, h2, z1)
    w1_gradient = w1_gradient_calc(b1_gradient, x)
    w2_gradient = w2_gradient_calc(h2, h1)
    return w1_gradient, w2_gradient, b1_gradient, b2_gradient


# Function that trains the algorithm and returns updated weights when done.
def train_func(W1, b1, W2, b2, epochs, train_x, train_y):
    # Normalize the training set to be less than 1.
    train_x /= 255
    for i in range(epochs):
        # Zip each training example with its correct value in order to train correctly.
        arr_zipped = list(zip(train_x, train_y))
        # Shuffle for better results.
        np.random.shuffle(arr_zipped)
        train_x, train_y = zip(*arr_zipped)
        # For each example and its respective value
        for x, y in zip(train_x, train_y):
            # Reshape in order to make the multiplication possible.
            x = np.reshape(x, (1, 784))
            y_hat, z1, z2, h1 = forward_propagation(W1, b1, W2, b2, x)
            W1_grad, W2_grad, b1_grad, b2_grad = backward_propagation(y_hat, z1, h1, W2, x, y)
            # Update the weights each time.
            W1, W2, b1, b2 = update_weights(W1, W2, b1, b2, W1_grad, W2_grad, b1_grad, b2_grad)
    return W1, W2, b1, b2


def main():
    epochs = 20
    max = 1
    min = -1
    hidden_layers = 150
    # Initialize the weights and biases with random numbers between negative one and one.
    W1 = np.random.uniform(min, max, [hidden_layers, 784])
    W2 = np.random.uniform(min, max, [10, hidden_layers])
    b1 = np.random.uniform(min, max, [hidden_layers, 1])
    b2 = np.random.uniform(min, max, [10, 1])
    # Get each of the nd arrays.
    x_training_set = get_file("train_x.txt")
    y_training_set = get_file("train_y.txt")
    x_testing_set = get_file("test_x.txt")
    W1_trained, W2_trained, b1_trained, b2_trained = train_func(W1, b1, W2, b2, epochs, x_training_set, y_training_set,)
    # get_y_file(W1_trained, W2_trained, b1_trained, b2_trained, x_testing_set)


if __name__ == "__main__":
    main()
