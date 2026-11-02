import numpy as np
from PIL import Image


def load_dataset(path="mnist.npz"):
    """Возвращает (x_train, y_train, x_test, y_test).
    Картинки: (N, 784), значения 0..1. Метки: one-hot (N, 10)."""
    with np.load(path) as f:
        x_train = f["x_train"].astype("float32") / 255
        x_test = f["x_test"].astype("float32") / 255
        y_train = np.eye(10, dtype="float32")[f["y_train"]]
        y_test = np.eye(10, dtype="float32")[f["y_test"]]
    return x_train.reshape(len(x_train), -1), y_train, x_test.reshape(len(x_test), -1), y_test


def load_custom_image(path="custom.jpg"):
    """Готовит свою картинку так же, как делали в MNIST:
    1) серый + инверсия (тёмная цифра на светлом фоне -> светлая на тёмном),
    2) обрезка по границам цифры,
    3) масштаб так, чтобы большая сторона = 20 пикселей (пропорции сохраняются),
    4) размещение в кадре 28x28 по центру масс.
    """
    img = Image.open(path).convert("L")
    arr = 1 - np.asarray(img, dtype="float32") / 255

    ys, xs = np.where(arr > 0.1)
    if len(ys) == 0:
        return np.zeros((28, 28), dtype="float32")
    arr = arr[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

    h, w = arr.shape
    scale = 20 / max(h, w)
    new_h, new_w = max(1, round(h * scale)), max(1, round(w * scale))
    digit = Image.fromarray((arr * 255).astype("uint8")).resize((new_w, new_h), Image.LANCZOS)
    digit = np.asarray(digit, dtype="float32") / 255

    canvas = np.zeros((28, 28), dtype="float32")
    top, left = (28 - new_h) // 2, (28 - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = digit

    total = canvas.sum()
    cy = (canvas.sum(axis=1) * np.arange(28)).sum() / total
    cx = (canvas.sum(axis=0) * np.arange(28)).sum() / total
    canvas = np.roll(canvas, (round(13.5 - cy), round(13.5 - cx)), axis=(0, 1))
    return canvas