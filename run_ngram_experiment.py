from __future__ import annotations

import gc
import json
import math
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "raw_data"
OUT_DIR = ROOT / "ngram_experiment"
CLEAN_DIR = OUT_DIR / "data"
RESULT_DIR = OUT_DIR / "results"
FIG_DIR = OUT_DIR / "figures"
MAX_ORDER = 8
MIN_COUNT = 2
BOS = "<BOS>"
EOS = "<EOS>"
UNK = "<UNK>"
PROMPT = "在阳光明媚的五月，我们学校胜利召开了……"


def log(message: str) -> None:
    print(message, flush=True)
    with (RESULT_DIR / "training.log").open("a", encoding="utf-8") as f:
        f.write(message + "\n")


def source_paths() -> list[Path]:
    if not DATA_DIR.is_dir():
        raise FileNotFoundError(
            f"未找到原始语料目录：{DATA_DIR}。请在项目根目录下创建"
            " `raw_data`，并放入六个按月保存的 GBK 文本文件。"
        )
    paths = sorted(DATA_DIR.glob("*.txt"))
    if len(paths) != 6:
        raise FileNotFoundError(
            f"语料目录 {DATA_DIR} 中需要六个 .txt 文件，当前找到 {len(paths)} 个。"
        )
    expected_months = [f"1998-{month:02d}" for month in range(1, 7)]
    actual_months = [path.name[:7] for path in paths]
    if actual_months != expected_months:
        raise FileNotFoundError(
            "语料文件名需按月份以 1998-01 至 1998-06 开头，当前排序为："
            + ", ".join(path.name for path in paths)
        )
    return paths


def clean_sources() -> tuple[list[list[list[str]]], dict]:
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    monthly: list[list[list[str]]] = []
    stats = {}
    for month, path in enumerate(source_paths(), 1):
        records: list[list[str]] = []
        source_lines = record_removed = pronunciation_removed = angle_removed = (
            replacement
        ) = corrupt_tokens = 0
        for raw in path.read_bytes().splitlines():
            source_lines += 1
            text = raw.decode("gbk", errors="replace")
            replacement += text.count("�")
            text = text.strip()
            if not text:
                continue
            label = re.match(r"^\d{8}-\d{2}-\d{3}-\d{3}\s+", text)
            if label:
                text = text[label.end() :]
                record_removed += 1
            pronunciation_removed += len(re.findall(r"\{[^{}]*\}", text))
            text = re.sub(r"\{[^{}]*\}", "", text)
            angle_removed += len(re.findall(r"<[^>]*>", text))
            text = re.sub(r"<[^>]*>", "", text)
            tokens = text.split()
            corrupt_tokens += sum("�" in token for token in tokens)
            tokens = [UNK if "�" in token else token for token in tokens]
            if tokens:
                records.append(tokens)
        monthly.append(records)
        clean_path = CLEAN_DIR / f"1998-{month:02d}.txt"
        clean_path.write_text(
            "\n".join(" ".join(r) for r in records) + "\n", encoding="utf-8"
        )
        row = {
            "source_lines": source_lines,
            "records_kept": len(records),
            "record_labels_removed": record_removed,
            "pronunciation_tags_removed": pronunciation_removed,
            "angle_tags_removed": angle_removed,
            "decode_replacement_chars": replacement,
            "corrupt_tokens_replaced_with_unk": corrupt_tokens,
            "tokens_after_cleaning": sum(map(len, records)),
            "source_bytes": path.stat().st_size,
        }
        stats[f"{month:02d}"] = row
        log(
            f"清理 {path.name}: records={len(records):,}, tokens={row['tokens_after_cleaning']:,}"
        )
    total = {
        k: sum(stats[m][k] for m in stats)
        for k in [
            "source_lines",
            "records_kept",
            "record_labels_removed",
            "pronunciation_tags_removed",
            "angle_tags_removed",
            "decode_replacement_chars",
            "corrupt_tokens_replaced_with_unk",
            "tokens_after_cleaning",
            "source_bytes",
        ]
    }
    return monthly, {"monthly": stats, "total": total}


def build_vocabulary(monthly: list[list[list[str]]]):
    train = [r for month in monthly[:4] for r in month]
    frequencies = Counter(t for r in train for t in r)
    vocab_words = [
        w
        for w, c in frequencies.items()
        if c >= MIN_COUNT and w not in {UNK, BOS, EOS} and "�" not in w
    ]
    vocab_words.sort(key=lambda w: (-frequencies[w], w))
    id_to_word = [UNK, BOS, EOS] + vocab_words
    word_to_id = {w: i for i, w in enumerate(id_to_word)}
    rare_types = sum(
        1 for w, c in frequencies.items() if c < MIN_COUNT and "�" not in w
    )
    monthly_ids: list[list[list[int]]] = []
    corrupt_occurrences = 0
    for mi, month in enumerate(monthly):
        converted = []
        for r in month:
            out = []
            for w in r:
                if "�" in w:
                    if mi < 4:
                        corrupt_occurrences += 1
                    out.append(word_to_id[UNK])
                else:
                    out.append(word_to_id.get(w, word_to_id[UNK]))
            converted.append(out)
        monthly_ids.append(converted)
    rare_occurrences = sum(
        c for w, c in frequencies.items() if c < MIN_COUNT and "�" not in w
    )
    return (
        word_to_id,
        id_to_word,
        frequencies,
        monthly_ids,
        {
            "minimum_count": MIN_COUNT,
            "vocabulary_size_including_specials": len(id_to_word),
            "training_tokens_before_oov_mapping": sum(map(len, train)),
            "rare_token_types_mapped_to_unk": rare_types,
            "rare_token_occurrences_mapped_to_unk": rare_occurrences,
            "corrupt_placeholder_occurrences_mapped_to_unk": corrupt_occurrences,
        },
    )


def pack(ids: list[int], bits: int) -> int:
    key = 0
    for x in ids:
        key = (key << bits) | x
    return key


def count_ngrams(month_records: list[list[int]], order: int, bits: int):
    counts: dict[int, dict[int, int]] = {
        n: defaultdict(int) for n in range(1, order + 1)
    }
    for rec in month_records:
        seq = [1] * (order - 1) + rec + [2]
        for i, target in enumerate(seq):
            if target == 1:
                continue
            for n in range(1, min(order, i + 1) + 1):
                key = pack(seq[i - n + 1 : i + 1], bits)
                counts[n][key] += 1
    return counts


def count_of_counts(distribution: dict[int, int]) -> tuple[float, float, float]:
    nc = Counter(distribution.values())
    n1, n2, n3, n4 = (nc.get(i, 0) for i in range(1, 5))
    if n1 == 0 or n2 == 0:
        return (0.75, 1.0, 1.25)
    y = n1 / (n1 + 2 * n2)
    d1 = 1 - 2 * y * n2 / n1
    d2 = 2 - 3 * y * n3 / n2 if n3 else 1.0
    d3 = 3 - 4 * y * n4 / n3 if n4 else 1.5
    return tuple(max(0.05, min(1.99, x)) for x in (d1, d2, d3))


def discount(discounts, c):
    return discounts[0] if c == 1 else discounts[1] if c == 2 else discounts[2]


class MKNModel:
    def __init__(
        self, order, bits, distributions, totals, masses, discounts, unigram_total
    ):
        self.order, self.bits = order, bits
        self.distributions, self.totals, self.masses = distributions, totals, masses
        self.discounts, self.unigram_total = discounts, unigram_total

    def probability(self, key: int, n: int | None = None) -> float:
        n = self.order if n is None else n
        if n == 1:
            return self.distributions[1].get(key, 0) / self.unigram_total
        history = key >> self.bits
        c = self.distributions[n].get(key, 0)
        denom = self.totals[n].get(history, 0)
        if c and denom:
            first = max(c - discount(self.discounts[n], c), 0.0) / denom
        else:
            first = 0.0
        # An unseen history has no discounted high-order mass and therefore
        # backs off completely to the lower-order distribution.
        back = self.masses[n].get(history, 1.0)
        lower_mask = (1 << (self.bits * (n - 1))) - 1
        lower = self.probability(key & lower_mask, n - 1)
        return first + back * lower


def build_mkn(raw, order, bits):
    distributions = {n: raw[n] for n in range(1, order + 1)}
    for n in range(order - 1, 0, -1):
        source = raw[n + 1]
        cont = defaultdict(int)
        suffix_mask = (1 << (bits * n)) - 1
        for key in source:
            cont[key & suffix_mask] += 1
        distributions[n] = dict(cont)
    discounts = {n: count_of_counts(distributions[n]) for n in range(2, order + 1)}
    totals = {n: defaultdict(int) for n in range(2, order + 1)}
    masses = {n: defaultdict(float) for n in range(2, order + 1)}
    for n in range(2, order + 1):
        for key, c in distributions[n].items():
            h = key >> bits
            totals[n][h] += c
    for n in range(2, order + 1):
        for key, c in distributions[n].items():
            h = key >> bits
            denom = totals[n][h]
            masses[n][h] += (
                max(discount(discounts[n], c), 0.0) / denom if denom else 0.0
            )
    return MKNModel(
        order,
        bits,
        distributions,
        totals,
        masses,
        discounts,
        sum(distributions[1].values()),
    )


def evaluate(model, records, bits):
    total_log = 0.0
    count = zeros = 0
    for rec in records:
        seq = [1] * (model.order - 1) + rec + [2]
        for i, target in enumerate(seq[model.order - 1 :], model.order - 1):
            hist = seq[i - model.order + 1 : i]
            key = pack(hist + [target], bits) if model.order > 1 else target
            p = model.probability(key, model.order)
            if p <= 0:
                zeros += 1
                p = 1e-12
            total_log += math.log(p)
            count += 1
    ppl = math.exp(-total_log / count)
    return {
        "perplexity": ppl,
        "tokens_scored_including_eos": count,
        "zero_probability_events": zeros,
    }


def build_context_totals(raw, order, bits):
    totals = {n: defaultdict(int) for n in range(2, order + 1)}
    for n in range(2, order + 1):
        for key, c in raw[n].items():
            totals[n][key >> bits] += c
    return totals


class JMModel:
    def __init__(self, raw, totals, order, bits, lam):
        self.raw, self.totals, self.order, self.bits, self.lam = (
            raw,
            totals,
            order,
            bits,
            lam,
        )
        self.uni = sum(raw[1].values())

    def probability(self, key, n=None):
        n = self.order if n is None else n
        if n == 1:
            return self.raw[1].get(key, 0) / self.uni
        h = key >> self.bits
        p = self.raw[n].get(key, 0) / self.totals[n].get(h, 1)
        lower_mask = (1 << (self.bits * (n - 1))) - 1
        return self.lam * p + (1 - self.lam) * self.probability(key & lower_mask, n - 1)


class KatzModel:
    def __init__(self, raw, totals, order, bits, discounts, alphas):
        self.raw, self.totals, self.order, self.bits = raw, totals, order, bits
        self.discounts, self.alphas = discounts, alphas
        self.unigram_total = sum(raw[1].values())

    def probability(self, key, n=None):
        n = self.order if n is None else n
        if n == 1:
            return self.raw[1].get(key, 0) / self.unigram_total
        history = key >> self.bits
        count = self.raw[n].get(key, 0)
        if count:
            denom = self.totals[n].get(history, 0)
            return (
                (1 - self.discounts[n].get(count, 0.99)) * count / denom
                if denom
                else 0.0
            )
        lower_mask = (1 << (self.bits * (n - 1))) - 1
        return self.alphas[n].get(history, 1.0) * self.probability(
            key & lower_mask, n - 1
        )


def build_katz(raw, order, bits):
    totals = build_context_totals(raw, order, bits)
    discounts = {}
    for n in range(2, order + 1):
        n_counts = Counter(raw[n].values())
        discounts[n] = {}
        for c in n_counts:
            if 1 <= c <= 6 and n_counts.get(c, 0) and n_counts.get(c + 1, 0):
                estimate = 1 - ((c + 1) * n_counts[c + 1] / n_counts[c]) / c
                discounts[n][c] = min(0.99, max(0.05, estimate))
            else:
                # Counts outside the Good–Turing estimation range are kept
                # unchanged; unseen continuations receive the backoff mass.
                discounts[n][c] = 0.0 if c > 6 else 0.5
    alphas = {n: defaultdict(float) for n in range(2, order + 1)}
    model = KatzModel(raw, totals, order, bits, discounts, alphas)
    for n in range(2, order + 1):
        mass = defaultdict(lambda: [0.0, 0.0])
        for key, c in raw[n].items():
            h = key >> bits
            denom = totals[n][h]
            mass[h][0] += (1 - discounts[n][c]) * c / denom if denom else 0.0
            lower_mask = (1 << (bits * (n - 1))) - 1
            mass[h][1] += model.probability(key & lower_mask, n - 1)
        for h, (discounted_mass, lower_mass) in mass.items():
            remaining = 1.0 - lower_mass
            alphas[n][h] = (
                max(0.0, (1.0 - discounted_mass) / remaining)
                if remaining > 1e-12
                else 1.0
            )
    return model


def generate(
    model,
    prompt_ids,
    id_to_word,
    candidate_ids,
    bits,
    temperature=1.0,
    top_k=20,
    seed=7,
    length=60,
):
    rng = random.Random(seed)
    context = [1] * (model.order - 1) + prompt_ids[:]
    out = []
    for _ in range(length):
        scored = []
        hist = context[-(model.order - 1) :] if model.order > 1 else []
        for w in candidate_ids:
            if w == 1:
                continue
            key = pack(hist + [w], bits) if model.order > 1 else w
            p = model.probability(key, model.order)
            if p > 0:
                scored.append((w, p))
        scored.sort(key=lambda x: x[1], reverse=True)
        if len(out) < 12:
            scored = [(w, p) for w, p in scored if w != 2]
        scored = scored[:top_k]
        if not scored:
            break
        weights = [max(p, 1e-12) ** (1 / max(temperature, 1e-3)) for _, p in scored]
        choice = rng.choices([w for w, _ in scored], weights=weights, k=1)[0]
        if choice == 2:
            break
        out.append(choice)
        context.append(choice)
    return " ".join(id_to_word[i] for i in out)


def oov_rate(records, word_to_id):
    total = unknown = 0
    for rec in records:
        for w in rec:
            total += 1
            unknown += w not in word_to_id
    return {
        "tokens": total,
        "oov_tokens": unknown,
        "oov_rate": unknown / total if total else 0,
    }


def make_plots(result):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    rows = result["order_sweep"]
    n = [r["n"] for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    ax[0].plot(n, [r["training_seconds"] for r in rows], marker="o")
    ax[0].set(xlabel="Order n", ylabel="Training time (s)", title="Training time")
    ax[1].plot(n, [r["test"]["perplexity"] for r in rows], marker="o", color="#d9534f")
    ax[1].set_yscale("log")
    ax[1].set(
        xlabel="Order n", ylabel="Test perplexity (log scale)", title="Test perplexity"
    )
    for a in ax:
        a.set_xticks(n)
        a.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "order_time_perplexity.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(
        n, [r["model_ngram_types"] / 1e6 for r in rows], marker="o", color="#59a14f"
    )
    ax.set(
        xlabel="Order n", ylabel="Unique n-gram types (millions)", title="Model size"
    )
    ax.set_xticks(n)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "order_model_size.png", dpi=180)
    plt.close(fig)
    comp = result["fourgram_comparison"]
    names = ["Jelinek–Mercer", "Katz", "Modified Kneser–Ney"]
    val = [comp[x]["validation"]["perplexity"] for x in ["jm", "katz", "mkn"]]
    test = [comp[x]["test"]["perplexity"] for x in ["jm", "katz", "mkn"]]
    pos = range(3)
    w = 0.36
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.bar([i - w / 2 for i in pos], val, w, label="Validation")
    ax.bar([i + w / 2 for i in pos], test, w, label="Test")
    ax.set_xticks(list(pos), names)
    ax.set_ylabel("Perplexity")
    ax.set_title("Smoothing comparison on the 4-gram model")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "smoothing_comparison.png", dpi=180)
    plt.close(fig)


def main():
    OUT_DIR.mkdir(exist_ok=True)
    RESULT_DIR.mkdir(exist_ok=True)
    (RESULT_DIR / "training.log").write_text("", encoding="utf-8")
    monthly, cleaning = clean_sources()
    word_to_id, id_to_word, frequencies, ids, vocab = build_vocabulary(monthly)
    bits = max(1, (len(id_to_word) - 1).bit_length())
    train = [r for m in ids[:4] for r in m]
    valid, test = ids[4], ids[5]
    rows = []
    prompt_tokens = [
        "在",
        "阳光",
        "明媚",
        "的",
        "五月",
        "，",
        "我们",
        "学校",
        "胜利",
        "召开",
        "了",
    ]
    prompt_ids = [word_to_id.get(w, word_to_id[UNK]) for w in prompt_tokens]
    candidate_ids = [
        word_to_id[w] for w, _ in frequencies.most_common(1000) if w in word_to_id
    ]
    candidate_ids.extend([word_to_id[UNK], word_to_id[EOS]])
    for order in range(1, MAX_ORDER + 1):
        log(f"开始训练 n={order}")
        started = time.perf_counter()
        raw = count_ngrams(train, order, bits)
        count_s = time.perf_counter() - started
        model = build_mkn(raw, order, bits)
        train_s = time.perf_counter() - started
        val = evaluate(model, valid, bits)
        te = evaluate(model, test, bits)
        generation = []
        if order in (1, 2, 4, 6, 8):
            for temp, topk in (
                ((0.8, 20), (1.0, 1), (1.3, 40)) if order == 4 else ((0.8, 20),)
            ):
                generation.append(
                    {
                        "temperature": temp,
                        "top_k": topk,
                        "continuation": generate(
                            model,
                            prompt_ids,
                            id_to_word,
                            candidate_ids,
                            bits,
                            temp,
                            topk,
                            17,
                        ),
                    }
                )
        unique = {str(k): len(raw[k]) for k in raw}
        row = {
            "n": order,
            "unique_ngrams_by_order": unique,
            "model_ngram_types": sum(unique.values()),
            "count_seconds": count_s,
            "training_seconds": train_s,
            "validation": val,
            "test": te,
            "generation": generation,
        }
        rows.append(row)
        log(
            f"完成 n={order}: types={row['model_ngram_types']:,}, training={train_s:.2f}s, test_ppl={te['perplexity']:.2f}"
        )
        del raw, model
        gc.collect()
    # Four-gram smoothing comparison; rebuild once.
    raw = count_ngrams(train, 4, bits)
    totals = build_context_totals(raw, 4, bits)
    jm_scores = []
    for lam in (0.2, 0.4, 0.6, 0.8):
        jm = JMModel(raw, totals, 4, bits, lam)
        jm_scores.append((evaluate(jm, valid, bits), lam))
    best_val, best_lam = min(jm_scores, key=lambda x: x[0]["perplexity"])
    jm = JMModel(raw, totals, 4, bits, best_lam)
    katz = build_katz(raw, 4, bits)
    mkn = build_mkn(raw, 4, bits)
    comparison = {
        "jm": {
            "lambda": best_lam,
            "validation": best_val,
            "test": evaluate(jm, test, bits),
        },
        "katz": {
            "validation": evaluate(katz, valid, bits),
            "test": evaluate(katz, test, bits),
        },
        "mkn": {
            "validation": evaluate(mkn, valid, bits),
            "test": evaluate(mkn, test, bits),
        },
    }
    vocab["corrupt_placeholder_occurrences_mapped_to_unk"] = sum(
        cleaning["monthly"][m]["corrupt_tokens_replaced_with_unk"]
        for m in ("01", "02", "03", "04")
    )
    result = {
        "experiment": "人民日报 1998 年上半年 n-gram 语言模型实验",
        "created_at_local": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cleaning": cleaning,
        "vocabulary": vocab,
        "split": {
            "train_months": ["1998-01", "1998-02", "1998-03", "1998-04"],
            "validation_months": ["1998-05"],
            "test_months": ["1998-06"],
        },
        "validation_oov": oov_rate(monthly[4], word_to_id),
        "test_oov": oov_rate(monthly[5], word_to_id),
        "prompt": PROMPT,
        "prompt_tokens": prompt_tokens,
        "order_sweep": rows,
        "fourgram_comparison": comparison,
        "environment": {
            "python_requirement": ">=3.10",
            "matplotlib_requirement": "3.8.4",
            "orders": list(range(1, 9)),
            "implementation": "Python packed-integer n-gram counters with modified Kneser–Ney, Jelinek–Mercer and Katz probability calculations",
        },
    }
    (RESULT_DIR / "experiment_results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    make_plots(result)
    log("实验完成")
    for label, path in (
        ("实验结果", RESULT_DIR / "experiment_results.json"),
        ("训练日志", RESULT_DIR / "training.log"),
        ("图表目录", FIG_DIR),
        ("清理后语料目录", CLEAN_DIR),
    ):
        log(f"{label}: {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
