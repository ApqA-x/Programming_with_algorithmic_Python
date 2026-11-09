"""ПР03: аналіз українських фільмів, варіант PY-C2Q / PR03-B2P."""

from __future__ import annotations

import argparse
import csv
import io
import json
import traceback
import warnings
from importlib.metadata import version
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import seaborn as sns

BASE = Path(__file__).resolve().parent


def load_data(path: Path) -> pd.DataFrame:
    raw = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path, dtype="str", encoding="utf-8-sig")
    if "source" not in raw:
        raw = raw.assign(source="films")
    if not raw.source.eq("films").any():
        fallback = pd.read_csv(BASE / "data/top_100_ukrainian_films.csv", dtype="str")
        raw = pd.concat([raw, fallback.assign(source="films")], ignore_index=True)
    return raw


def clean_films(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    films = raw.loc[raw.source.eq("films"), ["title", "year", "director", "duration_min", "link"]].reset_index(drop=True)
    stats = {"before": len(films)}
    # Пробіли не є частиною назви; порожній текст означає відсутність значення.
    for name in ("title", "director"):
        films[name] = films[name].str.strip().replace("", pd.NA)
    # «н/д» стає пропуском, Int64 зберігає цілі числа разом із пропусками.
    for name in ("year", "duration_min"):
        films[name] = pd.to_numeric(films[name], errors="coerce").astype("Int64")
    stats["missing_duration"] = int(films.duration_min.isna().sum())
    # Без назви й року фільм не ідентифікується. Невідому тривалість не вигадуємо.
    films = films.dropna(subset=["title", "year"])
    stats["after_dropna"] = len(films)
    stats["full_duplicates"] = int(films.duplicated().sum())
    stats["semantic_duplicates"] = int(films.duplicated(["title", "year"]).sum())
    # Змістовний ключ — назва та рік; перший запис зберігає порядок джерела.
    films = films.drop_duplicates(["title", "year"], keep="first")
    stats["after"] = len(films)
    # Короткометражки є справжніми спостереженнями, тому їх не видаляємо.
    suspicious = films.loc[(films.duration_min < 40) | (films.duration_min > 300)]
    stats["suspicious"] = json.loads(suspicious.to_json(orient="records", force_ascii=False))
    q1, q3 = films.duration_min.quantile([.25, .75])
    low, high = q1 - 1.5 * (q3-q1), q3 + 1.5 * (q3-q1)
    stats["iqr"] = {"lower": float(low), "upper": float(high), "count": int(((films.duration_min < low) | (films.duration_min > high)).sum())}
    if not films.year.between(1890, 2026).all():
        raise ValueError("Неможливий рік у наборі даних")
    return films.assign(decade=pd.col("year") // 10 * 10, hours=(pd.col("duration_min") / 60).round(2)), stats


def enrich(films: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    eras = pd.DataFrame({"decade": list(range(1920, 2000, 10)), "era": ["німе кіно", "довоєнне", "воєнне й повоєнне", "п'ятдесяті", "шістдесяті", "сімдесяті", "вісімдесяті", "дев'яності"]})
    before = len(films)
    # Ліве з'єднання зберігає нові фільми, яких ще немає в довіднику.
    films = films.merge(eras, on="decade", how="left", validate="m:1", indicator=True)
    stats = {"before": before, "after": len(films), "left_only": int(films._merge.eq("left_only").sum()), "missing_era": int(films.era.isna().sum())}
    films.loc[films.era.isna(), "era"] = "двотисячні й пізніше"
    films = films.drop(columns="_merge")
    words = films.title.str.split().str.len()
    films["title_words"] = pd.cut(words, [0, 1, 2, float("inf")], labels=["одне слово", "два слова", "три і більше"])
    # Права межа включена: 80 належить першій категорії, 100 — другій.
    films["duration_class"] = pd.cut(films.duration_min, [0, 80, 100, 400], labels=["до 80 хв", "80–100 хв", "понад 100 хв"])
    return films, stats


def chained_assignment_demo(films: pd.DataFrame) -> tuple[str, int]:
    demo = films.assign(mitka="звичайний")
    with warnings.catch_warnings():
        warnings.simplefilter("error", pd.errors.ChainedAssignmentError)
        try:
            demo[demo.duration_min > 100]["mitka"] = "довгий"
        except pd.errors.ChainedAssignmentError:
            trace = traceback.format_exc()
        else:
            raise RuntimeError("Очікуваний ChainedAssignmentError не виник")
    demo.loc[demo.duration_min > 100, "mitka"] = "довгий"
    return trace, int(demo.mitka.eq("довгий").sum())


def make_charts(films: pd.DataFrame, grouped: pd.DataFrame, out: Path) -> None:
    sns.set_theme(style="whitegrid", palette="colorblind")
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "axes.unicode_minus": False, "savefig.dpi": 200, "pdf.fonttype": 42, "svg.fonttype": "path"})
    def save(fig, name):
        with warnings.catch_warnings():
            warnings.filterwarnings("error", message="Glyph.*", category=UserWarning)
            fig.savefig(out / name, bbox_inches="tight")
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 1.4), layout="constrained")
    ax.set_title("Тест: Їжак, ґудзик, п'ять, знання — і мінус −5")
    ax.set_xticks([]); ax.set_yticks([])
    save(fig, "cyrillic.png")
    decade = grouped.sort_index()
    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    ax.plot(decade.index, decade.median_duration, marker="o")
    for x, row in decade.iterrows():
        ax.annotate(f"N={int(row.films_count)}", (x, row.median_duration), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=8)
    ax.set(xlabel="Десятиліття, роки", ylabel="Медіанна тривалість, хв", title=f"Медіана 2010-х — {decade.loc[2010, 'median_duration']:g} хв, а 1920-х — {decade.loc[1920, 'median_duration']:g} хв")
    save(fig, "classroom.png")
    top = grouped.head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 6), layout="constrained")
    bars = ax.barh(top.index.astype(str), top.films_count)
    ax.bar_label(bars, padding=3); ax.set_xlim(0, top.films_count.max()*1.15)
    ax.set(xlabel="Кількість фільмів, од.", ylabel="Десятиліття, роки", title=f"2010-ті представлені найбільше: {int(grouped.loc[2010, 'films_count'])} фільми зі {len(films)}")
    save(fig, "01-decades.png")
    durations = films.duration_min.dropna().astype(float)
    median = durations.median()
    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    ax.hist(durations, bins=list(range(0, (int(durations.max()) // 20 + 2) * 20, 20)), edgecolor="white")
    ax.axvline(median, color="#bd4b2b", label=f"Медіана: {median:g} хв")
    ax.legend(); ax.set(xlabel="Тривалість, хв", ylabel="Кількість фільмів, од.", title=f"Медіана тривалості — {median:g} хв серед {len(durations)} фільмів із відомим часом")
    save(fig, "02-duration.png")
    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    sns.countplot(data=films, y="title_words", hue="title_words", dodge=False, legend=False, ax=ax)
    for container in ax.containers:
        ax.bar_label(container, padding=3)
    counts = films.title_words.value_counts()
    ax.set(xlabel="Кількість фільмів, од.", ylabel="Кількість слів у назві, категорія", title=f"Назви категорії «{counts.index[0]}» мають {counts.iloc[0]} зі {len(films)} фільмів")
    ax.set_xlim(0, counts.max()*1.15)
    save(fig, "03-title-words.png")
    plot = px.scatter(films.dropna(subset=["duration_min"]), x="year", y="duration_min", color="title_words", hover_name="title", hover_data=["director", "year"], labels={"year": "Рік виходу, рік", "duration_min": "Тривалість, хв", "title_words": "Слів у назві", "director": "Режисер"}, title=f"{len(durations)} фільми з відомою тривалістю: {int((durations < 40).sum())} коротші за 40 хв")
    plot.update_layout(font_family="Arial", template="plotly_white")
    plot.write_html(out / "04-interactive.html", include_plotlyjs=True)


def run(path: Path, out: Path) -> dict:
    if int(pd.__version__.split('.')[0]) < 3:
        raise RuntimeError("Потрібна версія pandas 3 або новіша")
    out.mkdir(parents=True, exist_ok=True)
    charts = out / "grafiky"; charts.mkdir(exist_ok=True)
    raw = load_data(path)
    info = io.StringIO(); raw.info(buf=info)
    first = {"shape": list(raw.shape), "info": info.getvalue(), "dtypes": raw.dtypes.astype(str).to_dict(), "string_columns": raw.select_dtypes(include="str").columns.tolist(), "object_columns": raw.columns[raw.dtypes.eq("object")].tolist(), "sample": json.loads(raw.sample(min(5, len(raw)), random_state=3).to_json(orient="records", force_ascii=False)), "describe": raw.describe(include="all").to_string(), "missing_fraction": raw.isna().mean().to_dict(), "sources": raw.source.value_counts().to_dict()}
    films, cleaning = clean_films(raw)
    films, merge = enrich(films)
    trace, marked = chained_assignment_demo(films)
    (out / "chained_assignment.txt").write_text(trace, encoding="utf-8")
    sliced = films.loc[films.year.between(1930, 1969)]
    selected = sliced.loc[sliced.duration_min < 85].sort_values(["director", "title"])
    metric = round(float(selected.duration_min.mean()), 1)
    # Незалежний розрахунок стандартною бібліотекою повторює алгоритм ПР01.
    with (BASE / "data/top_100_ukrainian_films.csv").open(encoding="utf-8-sig", newline="") as f:
        reference = [r for r in csv.DictReader(f) if r["duration_min"].isdigit() and 1930 <= int(r["year"]) <= 1969 and int(r["duration_min"]) < 85]
    reference.sort(key=lambda r: (r["director"], r["title"]))
    baseline = round(sum(int(r["duration_min"]) for r in reference)/len(reference), 1)
    grouped = films.groupby("decade", observed=True).agg(films_count=("title", "count"), median_duration=("duration_min", "median"), max_duration=("duration_min", "max")).sort_values("films_count", ascending=False, kind="stable")
    periods = []
    for label, mask in [("до 1970", films.year < 1970), ("від 1970", films.year >= 1970)]:
        subset = films.loc[mask, "duration_min"]
        periods.append({"period": label, "all": len(subset), "known": int(subset.count()), "missing": int(subset.isna().sum()), "median": float(subset.median())})
    summary = {"variant": "PY-C2Q", "labVariant": "PR03-B2P", "source": path.name, "versions": {m: version(m) for m in ["pandas", "pyarrow", "matplotlib", "seaborn", "plotly"]}, "first_look": first, "cleaning": cleaning, "merge": merge, "marked_long": marked, "slice_count": len(sliced), "filter_count": len(selected), "metric": metric, "pr01_metric": baseline, "pr01_match": metric == baseline, "first_five_match": selected.title.head().tolist() == [r['title'] for r in reference[:5]], "categories": {c: {str(k): int(v) for k, v in films[c].value_counts(dropna=False, sort=False).items()} for c in ["title_words", "duration_class"]}, "periods": periods, "median_difference": periods[1]["median"]-periods[0]["median"], "overall_median": float(films.duration_min.median())}
    films.to_csv(out / "cleaned_films.csv", index=False, encoding="utf-8-sig")
    films.to_parquet(out / "cleaned_films.parquet", index=False)
    selected.to_csv(out / "selected_films.csv", index=False, encoding="utf-8-sig")
    grouped.to_csv(out / "groupby.csv", encoding="utf-8-sig")
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    make_charts(films, grouped, charts)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Аналіз фільмів: PY-C2Q / PR03-B2P")
    parser.add_argument("--input", type=Path, default=BASE / "data/top_100_ukrainian_films.csv", help="Вхідний CSV або Parquet")
    parser.add_argument("--output", type=Path, default=BASE / "results", help="Тека результатів")
    args = parser.parse_args()
    result = run(args.input, args.output)
    print(json.dumps({k: result[k] for k in ["slice_count", "filter_count", "metric", "pr01_match", "median_difference"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
