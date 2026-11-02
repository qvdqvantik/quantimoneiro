"""
    python main.py              - обучить сеть, сохранить (digits_01.ГГГГ.ММ.ДД.npz) и открыть окно рисования
    python main.py draw         - окно рисования с самой свежей сохранённой моделью
    python main.py draw ФАЙЛ    - окно рисования с конкретной моделью
    python main.py image        - распознать картинку custom.jpg
"""
import datetime
import glob
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

import utils

# ---------- настройки ----------
MODEL_PREFIX = "digits_01"
HIDDEN = 64
EPOCHS = 15
BATCH_SIZE = 64
LEARNING_RATE = 0.5
AUGMENT = True      # во время обучения случайно сдвигать цифры
MAX_SHIFT = 2       # на сколько пикселей максимум
SEED = 42
# --------------------------------

rng = np.random.default_rng(SEED)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def forward(x, p):
    hidden = sigmoid(x @ p["w1"] + p["b1"])
    return hidden, softmax(hidden @ p["w2"] + p["b2"])


def evaluate(x, y, p):
    """Возвращает (cross-entropy loss, accuracy в %)."""
    _, probs = forward(x, p)
    loss = -np.mean(np.sum(y * np.log(probs + 1e-9), axis=1))
    acc = np.mean(probs.argmax(axis=1) == y.argmax(axis=1)) * 100
    return loss, acc


def shift_batch(x):
    """Каждую картинку в пачке случайно сдвигает на +-MAX_SHIFT пикселей."""
    imgs = x.reshape(-1, 28, 28)
    out = np.empty_like(imgs)
    shifts = rng.integers(-MAX_SHIFT, MAX_SHIFT + 1, (len(imgs), 2))
    for i, (img, (dy, dx)) in enumerate(zip(imgs, shifts)):
        out[i] = np.roll(img, (dy, dx), axis=(0, 1))   # у MNIST поля по 4 px, ничего не обрежется
    return out.reshape(len(x), -1)


# ---------- сохранение / загрузка ----------
def new_model_path():
    """digits_01.2026.11.02.npz; если файл уже есть - добавляется _2, _3, ..."""
    base = f"{MODEL_PREFIX}.{datetime.date.today():%Y.%m.%d}"
    path, n = f"{base}.npz", 2
    while os.path.exists(path):
        path = f"{base}_{n}.npz"
        n += 1
    return path


def latest_model_path():
    files = glob.glob(f"{MODEL_PREFIX}.*.npz")
    if not files:
        sys.exit("Нет сохранённых моделей. Сначала запустите: python main.py")
    return max(files, key=os.path.getmtime)


def save(p):
    path = new_model_path()
    np.savez(path, **p)
    return path


def load(path=None):
    path = path or latest_model_path()
    print(f"Загружаю модель {path}")
    d = np.load(path)
    return {k: d[k] for k in ("w1", "b1", "w2", "b2")}


# ---------- обучение ----------
def train():
    x_train, y_train, x_test, y_test = utils.load_dataset()

    # Инициализация Xavier
    p = {
        "w1": rng.uniform(-1, 1, (784, HIDDEN)) * np.sqrt(6 / (784 + HIDDEN)),
        "b1": np.zeros(HIDDEN),
        "w2": rng.uniform(-1, 1, (HIDDEN, 10)) * np.sqrt(6 / (HIDDEN + 10)),
        "b2": np.zeros(10),
    }

    n = len(x_train)
    for epoch in range(1, EPOCHS + 1):
        order = rng.permutation(n)
        for start in range(0, n, BATCH_SIZE):
            idx = order[start:start + BATCH_SIZE]
            x, y = x_train[idx], y_train[idx]
            if AUGMENT:
                x = shift_batch(x)

            hidden, probs = forward(x, p)

            # градиенты считаем ДО обновления весов
            delta_out = (probs - y) / len(x)
            delta_hidden = (delta_out @ p["w2"].T) * hidden * (1 - hidden)

            p["w2"] -= LEARNING_RATE * hidden.T @ delta_out
            p["b2"] -= LEARNING_RATE * delta_out.sum(axis=0)
            p["w1"] -= LEARNING_RATE * x.T @ delta_hidden
            p["b1"] -= LEARNING_RATE * delta_hidden.sum(axis=0)

        tr_loss, tr_acc = evaluate(x_train, y_train, p)
        te_loss, te_acc = evaluate(x_test, y_test, p)
        print(f"Epoch {epoch:2d} | train loss {tr_loss:.4f} acc {tr_acc:.2f}% "
              f"| test loss {te_loss:.4f} acc {te_acc:.2f}%")

    path = save(p)
    print(f"Модель сохранена в {path}")
    return p


# ---------- окно рисования ----------
class DrawApp:
    N = 112      # холст 112x112 (в 4 раза больше, чем 28x28, чтобы линии были гладкими)
    BRUSH = 5    # радиус кисти

    def __init__(self, params):
        self.p = params
        self.canvas = np.zeros((self.N, self.N), dtype="float32")
        self.drawing = False
        self.last = None

        self.fig, (self.ax_c, self.ax_s, self.ax_b) = plt.subplots(
            1, 3, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1.3, 0.8, 1.3]})
        self.fig.suptitle("Рисуйте мышкой  |  Enter - распознать  |  C - очистить  |  Q - выход")

        self.img = self.ax_c.imshow(self.canvas, cmap="Greys", vmin=0, vmax=1)
        self.ax_c.set_title("Рисунок")
        self.ax_c.set_xticks([])
        self.ax_c.set_yticks([])

        self.small = self.ax_s.imshow(np.zeros((28, 28)), cmap="Greys", vmin=0, vmax=1)
        self.ax_s.set_title("Что видит сеть (28x28)")
        self.ax_s.set_xticks([])
        self.ax_s.set_yticks([])

        self.bars = self.ax_b.bar(range(10), np.zeros(10))
        self.ax_b.set_ylim(0, 1)
        self.ax_b.set_xticks(range(10))
        self.ax_b.set_title("Вероятности")

        c = self.fig.canvas
        c.mpl_connect("button_press_event", self.on_press)
        c.mpl_connect("motion_notify_event", self.on_move)
        c.mpl_connect("button_release_event", self.on_release)
        c.mpl_connect("key_press_event", self.on_key)

    # --- рисование ---
    def stamp(self, x, y):
        yy, xx = np.ogrid[:self.N, :self.N]
        self.canvas[(yy - y) ** 2 + (xx - x) ** 2 <= self.BRUSH ** 2] = 1

    def stroke(self, x, y):
        if self.last is None:
            points = [(x, y)]
        else:
            x0, y0 = self.last
            k = int(max(abs(x - x0), abs(y - y0))) + 1   # без дыр между точками
            points = zip(np.linspace(x0, x, k + 1), np.linspace(y0, y, k + 1))
        for px, py in points:
            self.stamp(px, py)
        self.last = (x, y)
        self.img.set_data(self.canvas)
        self.fig.canvas.draw_idle()

    def on_press(self, ev):
        if ev.inaxes is self.ax_c and ev.button == 1:
            self.drawing, self.last = True, None
            self.stroke(ev.xdata, ev.ydata)

    def on_move(self, ev):
        if not self.drawing:
            return
        if ev.inaxes is self.ax_c:
            self.stroke(ev.xdata, ev.ydata)
        else:
            self.last = None

    def on_release(self, ev):
        if self.drawing:
            self.drawing, self.last = False, None
            self.predict()

    def on_key(self, ev):
        if ev.key == "enter":
            self.predict()
        elif ev.key == "c":
            self.clear()

    # --- распознавание ---
    def predict(self):
        small = self.canvas.reshape(28, 4, 28, 4).mean(axis=(1, 3))
        if small.max() < 0.05:
            return None
        digit = utils.preprocess_digit(small)
        _, probs = forward(digit.reshape(1, -1), self.p)
        probs = probs[0]
        guess = int(probs.argmax())
        self.small.set_data(digit)
        for bar, value in zip(self.bars, probs):
            bar.set_height(value)
        self.ax_b.set_title(f"Это {guess}  ({probs[guess] * 100:.0f}%)")
        self.fig.canvas.draw_idle()
        return guess

    def clear(self):
        self.canvas[:] = 0
        self.img.set_data(self.canvas)
        self.small.set_data(np.zeros((28, 28)))
        for bar in self.bars:
            bar.set_height(0)
        self.ax_b.set_title("Вероятности")
        self.fig.canvas.draw_idle()


def predict_file(p, path="custom.jpg"):
    custom = utils.load_custom_image(path)
    _, probs = forward(custom.reshape(1, -1), p)
    guess = probs.argmax()
    print(f"Своя цифра: {guess} (уверенность {probs.max() * 100:.1f}%)")
    plt.imshow(custom, cmap="Greys")
    plt.title(f"Сеть думает, что это: {guess}")
    plt.show()


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "train"
    if mode == "draw":
        params = load(sys.argv[2] if len(sys.argv) > 2 else None)
    elif mode == "image":
        params = load(sys.argv[2] if len(sys.argv) > 2 else None)
        predict_file(params)
        sys.exit()
    else:
        params = train()
    DrawApp(params)
    plt.show()