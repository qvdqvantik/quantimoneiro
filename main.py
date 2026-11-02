import os

import numpy as np
import matplotlib.pyplot as plt

import utils


HIDDEN = 64
EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 0.5
SEED = 42


rng = np.random.default_rng(SEED)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    x = x - x.max(axis=1, keepdims=True)  
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def forward(x, params):
    w1, b1, w2, b2 = params
    hidden = sigmoid(x @ w1 + b1)
    probs = softmax(hidden @ w2 + b2)
    return hidden, probs


def evaluate(x, y, params):
    """Возвращает (cross-entropy loss, accuracy в %)."""
    _, probs = forward(x, params)
    loss = -np.mean(np.sum(y * np.log(probs + 1e-9), axis=1))
    acc = np.mean(probs.argmax(axis=1) == y.argmax(axis=1)) * 100
    return loss, acc


def main():
    x_train, y_train, x_test, y_test = utils.load_dataset()

    w1 = rng.uniform(-1, 1, (784, HIDDEN)) * np.sqrt(6 / (784 + HIDDEN))
    b1 = np.zeros(HIDDEN)
    w2 = rng.uniform(-1, 1, (HIDDEN, 10)) * np.sqrt(6 / (HIDDEN + 10))
    b2 = np.zeros(10)
    params = [w1, b1, w2, b2]

    n = len(x_train)
    for epoch in range(1, EPOCHS + 1):
        order = rng.permutation(n)
        for start in range(0, n, BATCH_SIZE):
            idx = order[start:start + BATCH_SIZE]
            x, y = x_train[idx], y_train[idx]

            hidden, probs = forward(x, params)

            delta_out = (probs - y) / len(x)
            delta_hidden = (delta_out @ params[2].T) * hidden * (1 - hidden)

            params[2] -= LEARNING_RATE * hidden.T @ delta_out
            params[3] -= LEARNING_RATE * delta_out.sum(axis=0)
            params[0] -= LEARNING_RATE * x.T @ delta_hidden
            params[1] -= LEARNING_RATE * delta_hidden.sum(axis=0)

        tr_loss, tr_acc = evaluate(x_train, y_train, params)
        te_loss, te_acc = evaluate(x_test, y_test, params)
        print(f"Epoch {epoch:2d} | train loss {tr_loss:.4f} acc {tr_acc:.2f}% " f"| test loss {te_loss:.4f} acc {te_acc:.2f}%")

    if os.path.exists("custom.jpg"):
        custom = utils.load_custom_image("custom.jpg")
        _, probs = forward(custom.reshape(1, -1), params)
        guess = probs.argmax()
        print(f"Своя цифра: {guess} (уверенность {probs.max() * 100:.1f}%)")

        plt.imshow(custom, cmap="Greys")
        plt.title(f"NN suggests the CUSTOM number is: {guess}")
        plt.show()


if __name__ == "__main__":
    main()