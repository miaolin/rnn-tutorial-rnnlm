import numpy as np
import operator


class rnn_numpy:
    def __init__(self, word_dim, hidden_dim=100, bptt_truncate=4):
        # Assign instance variables
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        # Randomly initialize the network parameters
        self.U = np.random.uniform(-np.sqrt(1. / word_dim), np.sqrt(1. / word_dim), (hidden_dim, word_dim))
        self.V = np.random.uniform(-np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim), (word_dim, hidden_dim))
        self.W = np.random.uniform(-np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim), (hidden_dim, hidden_dim))

    def forward_propagation(self, x):
        # The total number of time steps
        T = len(x)

        # During forward propagation we save all hidden states in s because need them later.
        # We add one additional element for the initial hidden, which we set to 0
        s = np.zeros((T + 1, self.hidden_dim))
        s[-1] = np.zeros(self.hidden_dim)

        # The outputs at each time step. Again, we save them for later.
        o = np.zeros((T, self.word_dim))

        for t in np.arange(T):
            # Note that we are indexing U by x[t]. This is the same as multiplying U with a one-hot vector.
            s[t] = np.tanh(self.U[:, x[t]] + self.W.dot(s[t - 1]))
            o[t] = softmax(self.V.dot(s[t]))
        return [o, s]

    def predict(self, x):
        # perform forward propagation and return index of the hightest score
        o, s = self.forward_propagation(x)
        return np.argmax(o, axis=1)

    def calculate_total_loss(self, x, y):
        L = 0

        for i in np.arange(len(y)):
            o, s = self.forward_propagation(x[i])
            corrected_word_predictions = o[np.arange(len(y[i])), y[i]]
            L += -1 * np.sum(np.log(corrected_word_predictions))
        return L

    def calculate_loss(self, x, y):
        N = np.sum(len(y_i) for y_i in y)
        return self.calculate_total_loss(x, y) / N

    def bptt(self, x, y):
        T = len(y)

        o, s = self.forward_propagation(x)
        dLdU = np.zeros(self.U.shape)
        dLdV = np.zeros(self.V.shape)
        dLdW = np.zeros(self.W.shape)
        delta_o = o
        delta_o[np.arange(len(y)), y] -= 1

        # for each output backwards.
        for t in np.arange(T)[::-1]:
            dLdV += np.outer(delta_o[t], s[t].T)

            # initial delta calculation
            delta_t = self.V.T.dot(delta_o[t]) * (1 - (s[t] ** 2))

            # backpropagation through time
            for bptt_step in np.arange(max(0, t-self.bptt_truncate), t+1)[::-1]:
                dLdW += np.outer(delta_t, s[bptt_step-1])
                dLdU[:, x[bptt_step]] += delta_t

                # update delta for the next step
                delta_t = self.W.T.dot(delta_t) * (1 - s[bptt_step-1] ** 2)
        return [dLdU, dLdV, dLdW]

    def gradient_check(self, x, y, h=0.001, error_threshold=0.01):
        # Calculate the gradients using backpropagation. We want to checker if these are correct.
        bptt_gradients = self.bptt(x, y)

        # List of all parameters we want to check.
        model_parameters = ['U', 'V', 'W']
        # Gradient check for each parameter
        for pidx, pname in enumerate(model_parameters):
            # Get the actual parameter value from the mode, e.g. model.W

            # parameter = operator.attrgetter(pname)(self)
            parameter = self.__getattribute__(pname)
            print("Performing gradient check for parameter {} with size {}.".format(pname, np.prod(parameter.shape)))

            # Iterate over each element of the parameter matrix, e.g. (0,0), (0,1), ...
            it = np.nditer(parameter, flags=['multi_index'], op_flags=['readwrite'])
            while not it.finished:
                ix = it.multi_index
                # Save the original value so we can reset it later
                original_value = parameter[ix]

                # Estimate the gradient using (f(x+h) - f(x-h))/(2*h)
                parameter[ix] = original_value + h
                gradplus = self.calculate_total_loss([x], [y])
                parameter[ix] = original_value - h
                gradminus = self.calculate_total_loss([x], [y])
                estimated_gradient = (gradplus - gradminus) / (2 * h)
                # Reset parameter to original value
                parameter[ix] = original_value

                # The gradient for this parameter calculated using backpropagation
                backprop_gradient = bptt_gradients[pidx][ix]
                error_value = relative_error(backprop_gradient, estimated_gradient)
                # If the error is to large fail the gradient check´
                if error_value > error_threshold:
                    print("Gradient Check ERROR: parameter={} ix={}".format(pname, ix))
                    print("+h Loss: {}".format(gradplus))
                    print("-h Loss: {}".format(gradminus))
                    print("Estimated_gradient: {}".format(estimated_gradient))
                    print("Backpropagation gradient: {}".format(backprop_gradient))
                    print("Relative Error: {}".format(error_value))
                    return
                it.iternext()
            print("Gradient check for parameter {} passed.".format(pname))


def softmax(values):
    exp_values = np.exp(values - np.max(values))
    return exp_values / np.sum(exp_values)


def relative_error(x, y):
    # calculate The relative error: (|x - y|/(|x| + |y|))
    return np.abs(x - y) / (np.abs(x) + np.abs(y))