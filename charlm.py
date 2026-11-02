import glob
import os
import sys
import datetime

import numpy as np

# ---------- настройки ----------
TEXTS_DIR = "texts"
MODEL_PREFIX = "quantum_01"
CONTEXT = 24        # сколько предыдущих букв модель "видит"
EMB = 24            # размер вектора одной буквы
HIDDEN = 256
BATCH = 128
STEPS = 20000
LR = 0.003          
TEMPERATURE = 0.8   
GEN_LEN = 300
SEED = 42
# --------------------------------

"""
    python charlm.py                 - обучение (тексты берутся из папки texts/*.txt), модель сохраняется как quantum_01.ГГГГ.ММ.ДД.npz, потом чат
    python charlm.py chat            - чат с самой свежей моделью
    python charlm.py chat ФАЙЛ.npz   - чат с конкретной моделью
"""

rng = np.random.default_rng(SEED)


def load_corpus():
    files = sorted(glob.glob(os.path.join(TEXTS_DIR, "**", "*.txt"), recursive=True))
    if not files:
        sys.exit(f"Положите .txt файлы (UTF-8) в папку '{TEXTS_DIR}/'.")
    parts = []
    for path in files:
        with open(path, encoding="utf-8", errors="ignore") as f:
            parts.append(f.read().replace("\r\n", "\n"))
        print(f"  загружен {path}")
    return "\n\n".join(parts)


def init_params(vocab_size):
    def uni(fan_in, fan_out, shape):
        return rng.uniform(-1, 1, shape) * np.sqrt(6 / (fan_in + fan_out))
    return {
        "E": rng.normal(0, 0.1, (vocab_size + 1, EMB)),
        "W1": uni(CONTEXT * EMB, HIDDEN, (CONTEXT * EMB, HIDDEN)),
        "b1": np.zeros(HIDDEN),
        "W2": uni(HIDDEN, vocab_size, (HIDDEN, vocab_size)),
        "b2": np.zeros(vocab_size),
    }


def forward(x, p):
    """x: (B, CONTEXT) номера букв -> вероятности следующей буквы (B, V)."""
    flat = p["E"][x].reshape(len(x), -1)
    h = np.tanh(flat @ p["W1"] + p["b1"])
    logits = h @ p["W2"] + p["b2"]
    logits -= logits.max(axis=1, keepdims=True)
    e = np.exp(logits)
    return flat, h, e / e.sum(axis=1, keepdims=True)


def loss_fn(probs, y):
    return -np.mean(np.log(probs[np.arange(len(y)), y] + 1e-9))


def backward(x, y, flat, h, probs, p):
    B = len(x)
    d_logits = probs.copy()
    d_logits[np.arange(B), y] -= 1
    d_logits /= B

    g = {"W2": h.T @ d_logits, "b2": d_logits.sum(axis=0)}
    d_pre = (d_logits @ p["W2"].T) * (1 - h ** 2)  
    g["W1"] = flat.T @ d_pre
    g["b1"] = d_pre.sum(axis=0)
    d_flat = d_pre @ p["W1"].T
    g["E"] = np.zeros_like(p["E"])
    np.add.at(g["E"], x, d_flat.reshape(B, CONTEXT, EMB))
    return g


def get_batch(ids, size, pad, augment=True):
    starts = rng.integers(0, len(ids) - CONTEXT - 1, size)
    x = np.stack([ids[s:s + CONTEXT] for s in starts])
    y = ids[starts + CONTEXT]
    if augment:
        for i in np.where(rng.random(size) < 0.33)[0]:
            x[i, :rng.integers(1, CONTEXT)] = pad
    return x, y


def generate(p, chars, prompt, length=GEN_LEN, temperature=TEMPERATURE):
    stoi = {c: i for i, c in enumerate(chars)}
    pad = len(chars)
    ctx = [pad] * CONTEXT + [stoi[c] for c in prompt if c in stoi]
    ctx = ctx[-CONTEXT:]
    out = []
    for _ in range(length):
        _, _, probs = forward(np.array([ctx]), p)
        q = np.power(probs[0] + 1e-9, 1 / temperature)
        q /= q.sum()
        nxt = int(rng.choice(len(q), p=q))
        out.append(chars[nxt])
        ctx = ctx[1:] + [nxt]
    return prompt + "".join(out)


def new_model_path():
    """quantum_01.2026.11.02.npz; если файл уже есть - добавляется _2, _3, ..."""
    base = f"{MODEL_PREFIX}.{datetime.date.today():%Y.%m.%d}"
    path, n = f"{base}.npz", 2
    while os.path.exists(path):
        path = f"{base}_{n}.npz"
        n += 1
    return path


def latest_model_path():
    files = glob.glob(f"{MODEL_PREFIX}.*.npz")
    if not files:
        sys.exit("Нет сохранённых моделей. Сначала запустите: python charlm.py")
    return max(files, key=os.path.getmtime)


def save(p, chars):
    path = new_model_path()
    np.savez(path, chars="".join(chars), **p)
    return path


def load(path=None):
    path = path or latest_model_path()
    print(f"Загружаю модель {path}")
    d = np.load(path)
    return {k: d[k] for k in ("E", "W1", "b1", "W2", "b2")}, list(str(d["chars"]))


def chat(p, chars):
    print("\nВведите начало текста').")
    while True:
        prompt = input("> ").replace("\\n", "\n")
        if not prompt:
            break
        print(generate(p, chars, prompt), "\n")


def train():
    print("Загружаю тексты:")
    text = load_corpus()
    chars = sorted(set(text) | {"\n"})
    stoi = {c: i for i, c in enumerate(chars)}
    ids = np.array([stoi[c] for c in text])
    pad = len(chars)
    print(f"Всего {len(ids)} символов, {len(chars)} уникальных.")
    if len(ids) < CONTEXT * 20:
        sys.exit("Текста слишком мало, добавьте больше файлов.")
    if len(ids) < 200_000:
        print("ВНИМАНИЕ: текста мало (<200 тыс. символов). Модель будет скорее заучивать, " "чем учиться. Добавьте книги/стихи в папку texts/.")

    split = int(len(ids) * 0.9) if len(ids) > 5000 else len(ids)
    train_ids, val_ids = ids[:split], (ids[split:] if split < len(ids) else None)

    p = init_params(len(chars))
    m = {k: np.zeros_like(v) for k, v in p.items()}
    v = {k: np.zeros_like(x) for k, x in p.items()}
    b1, b2, eps = 0.9, 0.999, 1e-8

    run = 0.0
    for step in range(1, STEPS + 1):
        x, y = get_batch(train_ids, BATCH, pad)
        flat, h, probs = forward(x, p)
        run = 0.98 * run + 0.02 * loss_fn(probs, y) if step > 1 else loss_fn(probs, y)
        g = backward(x, y, flat, h, probs, p)

        for k in p:  # Adam
            m[k] = b1 * m[k] + (1 - b1) * g[k]
            v[k] = b2 * v[k] + (1 - b2) * g[k] ** 2
            p[k] -= LR * (m[k] / (1 - b1 ** step)) / (np.sqrt(v[k] / (1 - b2 ** step)) + eps)

        if step % 500 == 0 or step == 1:
            msg = f"шаг {step:5d} | train loss {run:.3f}"
            if val_ids is not None and len(val_ids) > CONTEXT + 1:
                vx, vy = get_batch(val_ids, 512, pad, augment=False)
                msg += f" | val loss {loss_fn(forward(vx, p)[2], vy):.3f}"
            print(msg)
        if step % 2000 == 0 or step == STEPS:
            print("--- пример ---")
            print(generate(p, chars, "Вопрос:", 150))
            print("--------------")

    path = save(p, chars)
    print(f"Модель сохранена в {path}")
    return p, chars


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "chat":
        params, vocab = load(sys.argv[2] if len(sys.argv) > 2 else None)
    else:
        params, vocab = train()
    chat(params, vocab)