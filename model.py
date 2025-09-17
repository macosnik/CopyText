import numpy as np
import json


def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return (x > 0).astype(float)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def compute_loss(y_true, y_prediction):
    m = y_true.shape[0]
    log_likelihood = -np.log(y_prediction[range(m), y_true])
    return np.sum(log_likelihood) / m


class Net:
    def __init__(self, layers):
        self.layers = layers
        self.weights = []
        self.biases = []

        for i in range(len(layers) - 1):
            w = np.random.randn(layers[i], layers[i+1]) / np.sqrt(layers[i])
            b = np.zeros((1, layers[i+1]))
            self.weights.append(w)
            self.biases.append(b)

    def forward(self, x):
        activations = [x]
        pre_acts = []

        for i in range(len(self.weights) - 1):
            z = activations[-1] @ self.weights[i] + self.biases[i]
            pre_acts.append(z)
            a = relu(z)
            activations.append(a)

        z = activations[-1] @ self.weights[-1] + self.biases[-1]
        pre_acts.append(z)
        a = softmax(z)
        activations.append(a)

        return activations, pre_acts

    def backward(self, activations, pre_acts, y_true):
        grads_w = []
        grads_b = []

        m = y_true.shape[0]
        y_onehot = np.zeros_like(activations[-1])
        y_onehot[np.arange(m), y_true] = 1

        dz = activations[-1] - y_onehot

        for i in reversed(range(len(self.weights))):
            dw = activations[i].T @ dz / m
            db = np.sum(dz, axis=0, keepdims=True) / m
            grads_w.insert(0, dw)
            grads_b.insert(0, db)

            if i > 0:
                da = dz @ self.weights[i].T
                dz = da * relu_deriv(pre_acts[i-1])

        return grads_w, grads_b

    def update(self, grads_w, grads_b, lr):
        for i in range(len(self.weights)):
            self.weights[i] -= lr * grads_w[i]
            self.biases[i] -= lr * grads_b[i]

    def train(self, x_train, y_train, epochs, lr):
        for epoch in range(1, epochs+1):
            activations, pre_acts = self.forward(x_train)
            loss_train = compute_loss(y_train, activations[-1])

            grads_w, grads_b = self.backward(activations, pre_acts, y_train)
            self.update(grads_w, grads_b, lr)

            val_prediction, _ = self.forward(x_train)
            loss_val = compute_loss(y_train, val_prediction[-1])

            print(f"Epoch {epoch}/{epochs} - Train loss: {loss_train:.4f}, Val loss: {loss_val:.4f}")

    def predict(self, x):
        activations, _ = self.forward(x)
        return np.argmax(activations[-1], axis=1)

    def predict_proba(self, x):
        activations, _ = self.forward(x)
        return activations[-1]

    def save(self, filename):
        model_data = {
            "layers": self.layers,
            "weights": [w.tolist() for w in self.weights],
            "biases": [b.tolist() for b in self.biases]
        }
        with open(filename, "w") as f:
            json.dump(model_data, f)

    @classmethod
    def load(cls, filename):
        with open(filename, "r") as f:
            model_data = json.load(f)

        model = cls(model_data["layers"])
        model.weights = [np.array(w) for w in model_data["weights"]]
        model.biases = [np.array(b) for b in model_data["biases"]]
        return model
