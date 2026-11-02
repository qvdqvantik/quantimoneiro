"""
    python charlm.py                 - обучение (тексты берутся из папки texts/*.txt), модель сохраняется как quantum_01.ГГГГ.ММ.ДД.npz, потом чат
    python charlm.py chat            - чат с самой свежей моделью
    python charlm.py chat ФАЙЛ.npz   - чат с конкретной моделью

В чате пишите просто вопрос. Чтобы продолжить произвольный текст, начните с /raw
"""
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
ANSWER_LEN = 400    # максимальная длина ответа в чате
ANSWER_TEMPERATURE = 0.3  # для ответов лучше низкая: меньше выдумывания (0.2-0.5)
KEEP_BEST = False   # True: сохранять версию с лучшим val loss (и рано останавливаться), False: обучать все STEPS и сохранить последнюю
EVAL_EVERY = 500    # как часто считать val loss
PATIENCE = 5        # при KEEP_BEST=True: остановиться, если val loss не улучшался столько проверок подряд
SEED = 42
# --------------------------------

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


def generate(p, chars, prompt, length=GEN_LEN, temperature=TEMPERATURE, stop=()):
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
        text = "".join(out)
        hit = next((st for st in stop if text.endswith(st)), None)
        if hit:  # дошли до стоп-строки: ответ закончился
            return prompt + text[:-len(hit)]
    return prompt + "".join(out)


def answer(p, chars, question):
    """Оформляет вопрос как в обучающем тексте и останавливается в конце ответа."""
    prompt = f"Вопрос: {question.strip()}\nОтвет:"
    full = generate(p, chars, prompt, length=ANSWER_LEN, temperature=ANSWER_TEMPERATURE,
                    stop=("\n\n", "\nВопрос:"))
    return full[len(prompt):].strip()


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
    print("\nЗадайте вопрос. Начните с /raw, чтобы продолжить произвольный текст. Пустая строка - выход.")
    while True:
        q = input("> ").strip()
        if not q:
            break
        if q.startswith("/raw"):
            print(generate(p, chars, q[4:].lstrip().replace("\\n", "\n")), "\n")
        else:
            print("Ответ:", answer(p, chars, q), "\n")


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

    # фиксированная отложенная выборка - чтобы val loss сравнивался честно
    vx = vy = None
    if val_ids is not None and len(val_ids) > CONTEXT + 1:
        n_val = min(4096, len(val_ids) - CONTEXT - 1)
        starts = np.linspace(0, len(val_ids) - CONTEXT - 2, n_val).astype(int)
        vx = np.stack([val_ids[s0:s0 + CONTEXT] for s0 in starts])
        vy = val_ids[starts + CONTEXT]

    p = init_params(len(chars))
    m = {k: np.zeros_like(v) for k, v in p.items()}
    v = {k: np.zeros_like(x) for k, x in p.items()}
    b1, b2, eps = 0.9, 0.999, 1e-8

    best_loss, best_step, best_p, bad = float("inf"), 0, None, 0
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

        if step % EVAL_EVERY == 0 or step == 1:
            msg = f"шаг {step:5d} | train loss {run:.3f}"
            if vx is not None:
                val = loss_fn(forward(vx, p)[2], vy)
                msg += f" | val loss {val:.3f}"
                if val < best_loss:
                    best_loss, best_step, bad = val, step, 0
                    best_p = {k: a.copy() for k, a in p.items()}
                    msg += "  <- лучшая"
                elif step > 1:
                    bad += 1
            print(msg)
            if vx is not None and KEEP_BEST and bad >= PATIENCE:
                print(f"val loss не улучшается {PATIENCE} проверок подряд, останавливаю обучение.")
                break
        if step % 2000 == 0:
            print("--- пример ---")
            print("Вопрос: Как тебя зовут?\nОтвет:", answer(p, chars, "Как тебя зовут?"))
            print("--------------")

    if KEEP_BEST and best_p is not None:
        p = best_p
        print(f"Сохраняю лучшую версию: шаг {best_step}, val loss {best_loss:.3f}")

    path = save(p, chars)
    print(f"Модель сохранена в {path}")
    return p, chars


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "chat":
        params, vocab = load(sys.argv[2] if len(sys.argv) > 2 else None)
    else:
        params, vocab = train()
    chat(params, vocab)