import numpy as np
import json

def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return (x > 0).astype(float)

def softmax(x):
    e = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

def loss(y, p):
    y = np.array(y, dtype=int)
    return -np.log(p[np.arange(len(y)), y]).mean()

class Net:
    def __init__(self, layers):
        self.layers = layers
        self.weights = [
            np.random.randn(layers[i], layers[i + 1]) / np.sqrt(layers[i])
            for i in range(len(layers) - 1)
        ]
        self.biases = [
            np.zeros((1, layers[i + 1]))
            for i in range(len(layers) - 1)
        ]

    def forward(self, x):
        x = np.array(x, float)
        a, acts, zs = x, [x], []
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            z = a @ w + b
            a = relu(z)
            zs.append(z)
            acts.append(a)
        z = a @ self.weights[-1] + self.biases[-1]
        a = softmax(z)
        zs.append(z)
        acts.append(a)
        return acts, zs

    def backward(self, acts, zs, y):
        y = np.array(y, dtype=int)
        m = len(y)
        y_onehot = np.zeros_like(acts[-1])
        y_onehot[np.arange(m), y] = 1
        dz = acts[-1] - y_onehot
        gw, gb = [], []
        for i in reversed(range(len(self.weights))):
            gw.insert(0, acts[i].T @ dz / m)
            gb.insert(0, dz.mean(axis=0, keepdims=True))
            if i > 0:
                dz = (dz @ self.weights[i].T) * relu_deriv(zs[i - 1])
        return gw, gb

    def update(self, gw, gb, lr):
        for i in range(len(self.weights)):
            self.weights[i] -= lr * gw[i]
            self.biases[i] -= lr * gb[i]

    def train(self, x, y, epochs, lr):
        x = np.array(x, float)
        y = np.array(y, int)
        for e in range(1, epochs + 1):
            acts, zs = self.forward(x)
            l = loss(y, acts[-1])
            gw, gb = self.backward(acts, zs, y)
            self.update(gw, gb, lr)
            print(f"\rEpoch {e}/{epochs} - Loss: {l:.4f}", end="")

    def predict(self, x):
        x = np.array(x, float)
        return self.forward(x)[0][-1].argmax(axis=1)

    def predict_proba(self, x):
        x = np.array(x, float)
        return self.forward(x)[0][-1]

    def save(self, path):
        data = {
            "layers": self.layers,
            "weights": [w.tolist() for w in self.weights],
            "biases": [b.tolist() for b in self.biases],
        }
        with open(path, "w") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            data = json.load(f)
        net = cls(data["layers"])
        net.weights = [np.array(w) for w in data["weights"]]
        net.biases = [np.array(b) for b in data["biases"]]
        return net
