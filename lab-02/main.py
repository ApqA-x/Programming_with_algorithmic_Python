"""Практична робота 02, варіант PY-C2Q / PR02-B1U.

Збирає датасет із CSV, Open Library та Books to Scrape, кешує сирі відповіді
й зберігає результат у CSV і Parquet.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import time
from collections import Counter
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path
from typing import Any
from urllib.parse import urljoin
from urllib.robotparser import RobotFileParser

import pandas as pd
import requests
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field, ValidationError

WORK_DIR = Path(__file__).resolve().parent
CACHE_DIR = WORK_DIR / "cache"
BASE_FILMS_PATH = WORK_DIR / "top_100_ukrainian_films.csv"
OUTPUT_CSV_PATH = WORK_DIR / "combined_dataset.csv"
OUTPUT_PARQUET_PATH = WORK_DIR / "combined_dataset.parquet"
LOG_PATH = WORK_DIR / "collection_log.txt"
SUMMARY_PATH = WORK_DIR / "summary.json"

NBU_URL = "https://bank.gov.ua/NBUStatService/v1/statdirectory/exchange"
OPEN_LIBRARY_URL = "https://openlibrary.org/search.json"
TRAVEL_URL = "https://books.toscrape.com/catalogue/category/books/travel_2/index.html"
AGENT = "istu-course-bot/1.0 (+educational-use)"
TITLE_PATTERN = re.compile(r"^([А-ЯІЇЄҐа-яіїєґ]+)\s+([А-ЯІЇЄҐа-яіїєґ]+)$")
RATING_MAP = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}


@dataclass
class FetchStats:
    """Лічильники звернень до мережі та влучань у файловий кеш."""

    network_requests: int = 0
    cache_hits: int = 0
    failed_urls: list[str] = field(default_factory=list)


class Rate(BaseModel):
    """Модель відповіді НБУ для одного валютного курсу."""

    cc: str = Field(min_length=3, max_length=3)
    txt: str
    rate: float = Field(gt=0)
    exchangedate: str


class OpenLibraryItem(BaseModel):
    """Три поля, визначені карткою завдання для запису Open Library."""

    title: str = Field(min_length=1)
    first_publish_year: int | None = None
    author_name: list[str] = Field(min_length=1)


class TravelBook(BaseModel):
    """Чотири поля картки товару в категорії Travel."""

    title: str = Field(min_length=1)
    price_gbp: float = Field(gt=0)
    rating: int = Field(ge=1, le=5)
    availability: str = Field(min_length=1)


def fetch_json(url: str, params: dict[str, Any] | None = None) -> list[Any] | dict[str, Any] | None:
    """Виконує GET-запит, повертає JSON або друкує повідомлення."""
    try:
        response = requests.get(url, params=params, timeout=(3.05, 20))
        response.raise_for_status()
        return response.json()
    except requests.Timeout:
        print(f"Сервер не відповів за 20 с: {url}")
    except requests.ConnectionError:
        print(f"Не вдалося під'єднатися до {url}. Перевірте адресу й мережу.")
    except requests.HTTPError as error:
        print(f"Сервер відповів кодом {error.response.status_code}: {url}")
    except ValueError:
        print(f"Відповідь не є коректним JSON: {url}")
    return None


def fetch_cached(url: str, params: dict[str, Any] | None, stats: FetchStats) -> str:
    """Повертає сире тіло відповіді з кешу або мережі та зберігає нову відповідь."""
    params = params or {}
    cache_identity = f"utf8-v1::{url}?{sorted(params.items())}"
    key = hashlib.sha256(cache_identity.encode()).hexdigest()[:16]
    path = CACHE_DIR / f"{key}.txt"
    if path.exists():
        stats.cache_hits += 1
        return path.read_text(encoding="utf-8")

    try:
        response = requests.get(
            url,
            params=params,
            headers={"User-Agent": AGENT},
            timeout=(3.05, 20),
        )
        response.raise_for_status()
    except requests.RequestException:
        stats.failed_urls.append(response.url if "response" in locals() else url)
        raise

    stats.network_requests += 1
    # Сервер навчального сайту не вказує кодування, тому фіксуємо UTF-8 до .text.
    response.encoding = "utf-8"
    path.write_text(response.text, encoding="utf-8")
    time.sleep(1)
    return response.text


def collect_nbu_rates(stats: FetchStats) -> list[Rate]:
    """Збирає три курси EUR із НБУ через кешовані сирі JSON-відповіді."""
    rates: list[Rate] = []
    for date in ("20260831", "20260901", "20260902"):
        raw = fetch_cached(NBU_URL, {"valcode": "EUR", "date": date, "json": ""}, stats)
        payload = json.loads(raw)
        rates.extend(Rate.model_validate(item) for item in payload)
    return rates


def collect_openlibrary_pages(stats: FetchStats, *, pages: int = 3) -> tuple[list[OpenLibraryItem], int, int]:
    """Збирає до трьох сторінок Open Library і рахує відхилені Pydantic-записи."""
    items: list[OpenLibraryItem] = []
    validation_errors = 0
    pages_read = 0
    for page in range(1, pages + 1):
        raw = fetch_cached(OPEN_LIBRARY_URL, {"q": "ukraine", "limit": 25, "page": page}, stats)
        payload = json.loads(raw)
        documents = payload.get("docs", [])
        if not documents:
            break
        pages_read += 1
        for document in documents[:25]:
            try:
                items.append(OpenLibraryItem.model_validate(document))
            except ValidationError:
                validation_errors += 1
    return items, pages_read, validation_errors


def check_robots(target_url: str) -> tuple[int, bool, int | None]:
    """Перевіряє robots.txt сайту та можливість звернутися до цільової сторінки."""
    robots_url = urljoin(target_url, "/robots.txt")
    response = requests.get(robots_url, headers={"User-Agent": AGENT}, timeout=(3.05, 20))
    parser = RobotFileParser()
    parser.set_url(robots_url)
    parser.read()
    return response.status_code, parser.can_fetch(AGENT, target_url), parser.crawl_delay(AGENT)


def collect_travel_books(stats: FetchStats) -> tuple[list[TravelBook], int]:
    """Збирає чотири поля з карток книг категорії Travel."""
    raw = fetch_cached(TRAVEL_URL, None, stats)
    soup = BeautifulSoup(raw, "lxml")
    cards = soup.select("article.product_pod")
    if not cards:
        raise RuntimeError("Не знайдено карток книг. Перевірте CSS-селектор.")

    books: list[TravelBook] = []
    missing = 0
    for card in cards:
        title_element = card.select_one("h3 a")
        price_element = card.select_one("p.price_color")
        rating_element = card.select_one("p.star-rating")
        availability_element = card.select_one("p.availability")
        if None in (title_element, price_element, rating_element, availability_element):
            missing += 1
            continue
        rating_class = next((name for name in rating_element.get("class", []) if name in RATING_MAP), None)
        if rating_class is None:
            missing += 1
            continue
        books.append(
            TravelBook(
                title=title_element.get("title", "").strip(),
                price_gbp=float(price_element.get_text(strip=True).replace("£", "")),
                rating=RATING_MAP[rating_class],
                availability=availability_element.get_text(" ", strip=True),
            )
        )
    return books, missing


def read_films() -> list[dict[str, str]]:
    """Читає базовий набір даних про українські фільми."""
    with BASE_FILMS_PATH.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def combine_records(
    films: list[dict[str, str]],
    api_items: list[OpenLibraryItem],
    travel_books: list[TravelBook],
) -> list[dict[str, Any]]:
    """Об'єднує три джерела, додаючи source, title_norm і year_norm."""
    records: list[dict[str, Any]] = [
        {"source": "films", "title_norm": row["title"], "year_norm": row["year"], **row}
        for row in films
    ]
    records.extend(
        {
            "source": "api",
            "title_norm": item.title,
            "year_norm": str(item.first_publish_year) if item.first_publish_year is not None else "",
            "api_title": item.title,
            "api_first_publish_year": item.first_publish_year,
            "api_author": item.author_name[0],
        }
        for item in api_items
    )
    records.extend(
        {
            "source": "scrape",
            "title_norm": book.title,
            "year_norm": "",
            "scrape_title": book.title,
            "scrape_price_gbp": book.price_gbp,
            "scrape_rating": book.rating,
            "scrape_availability": book.availability,
        }
        for book in travel_books
    )
    return records


def save_dataset(records: list[dict[str, Any]]) -> tuple[int, int]:
    """Зберігає об'єднаний датасет у CSV UTF-8-SIG і Parquet зі стисненням Zstandard."""
    dataframe = pd.DataFrame(records)
    dataframe["year_norm"] = dataframe["year_norm"].astype("string")
    dataframe.to_csv(OUTPUT_CSV_PATH, index=False, encoding="utf-8-sig")
    dataframe.to_parquet(OUTPUT_PARQUET_PATH, compression="zstd", index=False)
    return OUTPUT_CSV_PATH.stat().st_size, OUTPUT_PARQUET_PATH.stat().st_size


def regex_examples(films: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Повертає три приклади перевірки виразу для назв рівно з двох слів."""
    samples = ["Камінний хрест", "За двома зайцями", "Білий птах з чорною ознакою"]
    return [{"input": sample, "matches": TITLE_PATTERN.findall(sample)} for sample in samples]


def broken_url_messages() -> list[str]:
    """Повертає дружні повідомлення для недоступного домену та шляху 404."""
    messages: list[str] = []
    for url in (
        "https://openlibrary-nonexistent.example/search.json",
        "https://openlibrary.org/nonexistent",
    ):
        output = StringIO()
        with redirect_stdout(output):
            fetch_json(url)
        messages.append(output.getvalue().strip())
    return messages


def validate_rate_model() -> list[str]:
    """Повертає повідомлення про помилки моделі Rate на навмисно зламаному записі."""
    broken = {"cc": "US", "txt": "Долар США", "rate": -1, "exchangedate": "02.09.2026"}
    try:
        Rate.model_validate(broken)
    except ValidationError as error:
        return [f"{item['loc']}: {item['msg']}" for item in error.errors()]
    raise AssertionError("Некоректний запис не пройшов валідацію.")


def run_collection() -> dict[str, Any]:
    """Запускає два проходи збору для перевірки кешу та формує вихідні файли."""
    started_at = time.strftime("%Y-%m-%d %H:%M:%S")
    example_usd = fetch_json(NBU_URL, {"valcode": "USD", "date": "20260902", "json": ""})
    if not example_usd:
        raise RuntimeError("НБУ не повернув приклад відповіді для USD.")

    robots_status, robots_allowed, crawl_delay = check_robots(TRAVEL_URL)

    first_stats = FetchStats()
    started = time.perf_counter()
    rates, _, _ = collect_nbu_rates(first_stats), None, None
    api_items, pages_read, validation_errors = collect_openlibrary_pages(first_stats)
    travel_books, scrape_missing = collect_travel_books(first_stats)
    first_duration = time.perf_counter() - started

    second_stats = FetchStats()
    started = time.perf_counter()
    second_rates = collect_nbu_rates(second_stats)
    second_api_items, second_pages, second_validation_errors = collect_openlibrary_pages(second_stats)
    second_travel_books, second_scrape_missing = collect_travel_books(second_stats)
    second_duration = time.perf_counter() - started

    if (len(rates), len(api_items), len(travel_books)) != (
        len(second_rates),
        len(second_api_items),
        len(second_travel_books),
    ):
        raise RuntimeError("Повторний прохід через кеш повернув іншу кількість записів.")
    if (pages_read, validation_errors, scrape_missing) != (
        second_pages,
        second_validation_errors,
        second_scrape_missing,
    ):
        raise RuntimeError("Повторний прохід через кеш повернув інші службові показники.")

    films = read_films()
    records = combine_records(films, api_items, travel_books)
    csv_size, parquet_size = save_dataset(records)
    source_counts = Counter(record["source"] for record in records)

    summary = {
        "started_at": started_at,
        "variant": "PY-C2Q",
        "lab_variant": "PR02-B1U",
        "nby_usd_example": example_usd[0],
        "rate_validation_errors": validate_rate_model(),
        "demo_key_present": bool(os.getenv("PR02_DEMO_KEY")),
        "robots": {
            "status": robots_status,
            "can_fetch": robots_allowed,
            "crawl_delay": crawl_delay,
        },
        "cache": {
            "first_pass_seconds": first_duration,
            "second_pass_seconds": second_duration,
            "first_pass_network_requests": first_stats.network_requests,
            "first_pass_cache_hits": first_stats.cache_hits,
            "second_pass_network_requests": second_stats.network_requests,
            "second_pass_cache_hits": second_stats.cache_hits,
        },
        "api": {
            "pages_read": pages_read,
            "items_collected": len(api_items),
            "validation_errors": validation_errors,
        },
        "scrape": {"items_collected": len(travel_books), "missing_fields": scrape_missing},
        "dataset": {
            "total_rows": len(records),
            "source_counts": dict(source_counts),
            "csv_bytes": csv_size,
            "parquet_bytes": parquet_size,
        },
        "regex": {"pattern": TITLE_PATTERN.pattern, "examples": regex_examples(films)},
        "broken_url_messages": broken_url_messages(),
        "failed_urls": sorted(set(first_stats.failed_urls + second_stats.failed_urls)),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    LOG_PATH.write_text(make_log(summary), encoding="utf-8")
    return summary


def make_log(summary: dict[str, Any]) -> str:
    """Формує текстовий журнал збору з необхідними показниками."""
    cache = summary["cache"]
    api = summary["api"]
    scrape = summary["scrape"]
    dataset = summary["dataset"]
    failed = summary["failed_urls"] or ["немає"]
    return "\n".join(
        [
            "Журнал збору ПР02",
            f"Час початку: {summary['started_at']}",
            f"Тривалість першого проходу: {cache['first_pass_seconds']:.3f} с",
            f"Тривалість другого проходу: {cache['second_pass_seconds']:.3f} с",
            f"Запити мережею у першому проході: {cache['first_pass_network_requests']}",
            f"Влучання в кеш у першому проході: {cache['first_pass_cache_hits']}",
            f"Запити мережею у другому проході: {cache['second_pass_network_requests']}",
            f"Влучання в кеш у другому проході: {cache['second_pass_cache_hits']}",
            f"Сторінок Open Library: {api['pages_read']}",
            f"Записів Open Library: {api['items_collected']}",
            f"Помилок валідації Open Library: {api['validation_errors']}",
            f"Записів Books to Scrape: {scrape['items_collected']}",
            f"Карток Books to Scrape з пропущеними полями: {scrape['missing_fields']}",
            "Кількість записів за джерелами: "
            + " / ".join(f"{key} {value}" for key, value in dataset["source_counts"].items()),
            "Невдалі URL: " + ", ".join(failed),
        ]
    ) + "\n"


def main() -> None:
    """Виконує збір і друкує короткий підсумок у JSON."""
    summary = run_collection()
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
