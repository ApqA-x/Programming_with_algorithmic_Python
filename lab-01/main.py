"""Практична робота 01, варіант PY-C2Q.

Аналізує набір даних про 100 українських фільмів стандартними засобами Python.
"""

from __future__ import annotations

import csv
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any


WORK_DIR = Path(__file__).resolve().parent
DATA_PATH = WORK_DIR / "top_100_ukrainian_films - top_100_ukrainian_films.csv"
SELECTED_CSV_PATH = WORK_DIR / "selected_films.csv"
SELECTED_JSON_PATH = WORK_DIR / "selected_films.json"


def read_films(path: Path) -> list[dict[str, str]]:
    """Зчитує CSV-файл у список словників із явним кодуванням UTF-8."""
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def filter_films(
    rows: list[dict[str, str]],
    *,
    year_from: int | None = None,
    year_to: int | None = None,
    min_duration: int | None = None,
    max_duration: int | None = None,
) -> tuple[list[dict[str, str]], int]:
    """Відбирає фільми за роком і тривалістю.

    Рядок, у якому рік або тривалість не є числом, не потрапляє до
    результату та рахується як пропущений. Вихідний список не змінюється.
    Повертає пару: відібрані рядки та кількість пропущених рядків.
    """
    selected: list[dict[str, str]] = []
    skipped = 0

    for row in rows:
        try:
            year = int(row["year"].strip())
            duration = int(row["duration_min"].strip())
        except (KeyError, ValueError):
            skipped += 1
            continue

        if year_from is not None and year < year_from:
            continue
        if year_to is not None and year > year_to:
            continue
        if min_duration is not None and duration < min_duration:
            continue
        if max_duration is not None and duration > max_duration:
            continue
        selected.append(row)

    return selected, skipped


def decade_distribution(rows: list[dict[str, str]]) -> Counter[int]:
    """Повертає кількість фільмів за десятиліттями."""
    return Counter(int(row["year"]) // 10 * 10 for row in rows)


def top_directors(rows: list[dict[str, str]], limit: int = 10) -> list[tuple[str, int]]:
    """Повертає режисерів, відсортованих за кількістю фільмів і прізвищем."""
    directors = Counter(row["director"] for row in rows)
    return sorted(directors.items(), key=lambda pair: (-pair[1], pair[0]))[:limit]


def benchmark_membership() -> dict[str, float | int]:
    """Порівнює час 100 пошуків елементів у списку та множині."""
    haystack = list(range(1_000_000))
    needles = [999_900 + index for index in range(100)]

    start = time.perf_counter()
    found_list = sum(1 for number in needles if number in haystack)
    list_time = time.perf_counter() - start

    start = time.perf_counter()
    haystack_set = set(haystack)
    build_time = time.perf_counter() - start

    start = time.perf_counter()
    found_set = sum(1 for number in needles if number in haystack_set)
    set_time = time.perf_counter() - start

    return {
        "found_list": found_list,
        "found_set": found_set,
        "build_time_ms": build_time * 1_000,
        "list_time_ms": list_time * 1_000,
        "set_time_ms": set_time * 1_000,
        "speedup": list_time / set_time if set_time else float("inf"),
    }


def write_and_check_roundtrip(rows: list[dict[str, str]]) -> bool:
    """Зберігає записи у CSV і JSON та перевіряє їх після повторного читання."""
    if not rows:
        raise ValueError("Неможливо записати порожній набір даних.")

    fieldnames = list(rows[0])
    with SELECTED_CSV_PATH.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with SELECTED_JSON_PATH.open("w", encoding="utf-8") as file:
        json.dump(rows, file, ensure_ascii=False, indent=2)

    with SELECTED_CSV_PATH.open(encoding="utf-8", newline="") as file:
        back_csv = list(csv.DictReader(file))
    with SELECTED_JSON_PATH.open(encoding="utf-8") as file:
        back_json = json.load(file)

    return rows == back_csv == back_json


def make_summary(rows: list[dict[str, str]], selected: list[dict[str, str]], skipped: int) -> dict[str, Any]:
    """Формує результати всіх розрахунків для звіту та виводу в консоль."""
    selected.sort(key=lambda row: (row["director"], row["title"]))
    mean_duration = sum(int(row["duration_min"]) for row in selected) / len(selected)
    return {
        "rows_total": len(rows),
        "decades": sorted(decade_distribution(rows).items()),
        "top_directors": top_directors(rows),
        "selected_count": len(selected),
        "skipped_count": skipped,
        "first_five": selected[:5],
        "mean_duration": round(mean_duration, 1),
        "roundtrip_ok": write_and_check_roundtrip(selected),
        "benchmark": benchmark_membership(),
    }


def main() -> None:
    """Запускає обчислення для варіанта PY-C2Q і друкує підсумок."""
    rows = read_films(DATA_PATH)
    # Умова duration_min < 85 для цілих чисел реалізована включною межею 84.
    selected, skipped = filter_films(
        rows,
        year_from=1930,
        year_to=1969,
        max_duration=84,
    )
    summary = make_summary(rows, selected, skipped)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
