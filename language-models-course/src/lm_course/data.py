# src/lm_course/data.py
"""Datasets used by the labs: downloads (TinyStories, GPT-2 files) and small generated corpora.

Nothing here is stored in the repository. Downloads go to ``data/`` (ignored by git) on first
use; the generated corpora are rebuilt from a seed every time. Licences are listed in NOTICE.md.
"""

import random
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from lm_course.utils import DATA_DIR

TINYSTORIES_URL = (
    "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/{name}"
)
TINYSTORIES_FILE = "TinyStoriesV2-GPT4-valid.txt"
TINYSTORIES_SEPARATOR = "<|endoftext|>"
GPT2_URL = "https://huggingface.co/openai-community/gpt2/resolve/main/{name}"
USER_AGENT = "lm-course/0.1"
DOWNLOAD_ATTEMPTS = 5


def download(url: str, path: Path) -> Path:
    """Download ``url`` to ``path`` once; later calls return the cached file.

    Large transfers can be cut off without an error, so the size is checked against
    Content-Length and an interrupted download resumes with an HTTP Range request.
    """
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"downloading {url}")
    partial = path.with_suffix(path.suffix + ".part")
    expected = None
    for _ in range(DOWNLOAD_ATTEMPTS):
        done = partial.stat().st_size if partial.exists() else 0
        headers = {"User-Agent": USER_AGENT}
        if done:
            headers["Range"] = f"bytes={done}-"
        request = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(request, timeout=120) as response:
            if expected is None:
                length = response.headers.get("Content-Length")
                expected = int(length) + done if length is not None else None
            if done and response.status != 206:  # server ignored the Range header
                done = 0
            with partial.open("ab" if done else "wb") as f:
                while chunk := response.read(1 << 20):
                    f.write(chunk)
        if expected is None or partial.stat().st_size >= expected:
            break
        logger.warning(
            f"transfer cut off at {partial.stat().st_size} of {expected} bytes"
        )
    else:
        raise OSError(f"could not download {url} after {DOWNLOAD_ATTEMPTS} attempts")
    partial.rename(path)
    logger.info(f"saved {path.name} ({path.stat().st_size / 1e6:.1f} MB)")
    return path


def tinystories_path() -> Path:
    """The GPT-4 half of TinyStories' validation split (22.5 MB, CDLA-Sharing-1.0)."""
    return download(
        TINYSTORIES_URL.format(name=TINYSTORIES_FILE), DATA_DIR / TINYSTORIES_FILE
    )


def load_stories(path: Path | None = None) -> list[str]:
    """One string per story. The file starts in the middle of a story, so that one is dropped."""
    text = (path or tinystories_path()).read_text(encoding="utf-8")
    chunks = [chunk.strip() for chunk in text.split(TINYSTORIES_SEPARATOR)]
    return [chunk for chunk in chunks[1:] if chunk]


def split_stories(
    stories: list[str], val_fraction: float = 0.1, seed: int = 0
) -> tuple[list[str], list[str]]:
    """Deterministic train / validation split by story (never by character offset)."""
    order = list(range(len(stories)))
    random.Random(seed).shuffle(order)
    num_val = int(len(stories) * val_fraction)
    val = [stories[i] for i in sorted(order[:num_val])]
    train = [stories[i] for i in sorted(order[num_val:])]
    return train, val


def gpt2_file(name: str) -> Path:
    """A file of the original 124M GPT-2 release, mirrored on Hugging Face (Modified MIT)."""
    return download(GPT2_URL.format(name=name), DATA_DIR / "gpt2" / name)


# ---------------------------------------------------------------------------------------------
# A generated corpus with known structure, for word2vec (lesson 02)
# ---------------------------------------------------------------------------------------------

GENDER_PAIRS = [
    ("king", "queen"),
    ("prince", "princess"),
    ("man", "woman"),
    ("boy", "girl"),
    ("father", "mother"),
    ("son", "daughter"),
    ("brother", "sister"),
    ("uncle", "aunt"),
    ("husband", "wife"),
    ("nephew", "niece"),
]
ROYAL = {"king", "queen", "prince", "princess"}
YOUNG = {"prince", "princess", "boy", "girl", "son", "daughter", "nephew", "niece"}
COUNTRIES = [
    # (country, capital, language)
    ("france", "paris", "french"),
    ("japan", "tokyo", "japanese"),
    ("italy", "rome", "italian"),
    ("germany", "berlin", "german"),
    ("spain", "madrid", "spanish"),
    ("russia", "moscow", "russian"),
    ("china", "beijing", "chinese"),
    ("greece", "athens", "greek"),
    ("poland", "warsaw", "polish"),
    ("portugal", "lisbon", "portuguese"),
    ("sweden", "stockholm", "swedish"),
    ("turkey", "ankara", "turkish"),
]
FILLER = [
    "today",
    "again",
    "yesterday",
    "quietly",
    "at night",
    "in the morning",
    "every day",
]


@dataclass(frozen=True)
class Analogy:
    """a : b :: c : d, e.g. king : queen :: man : woman."""

    a: str
    b: str
    c: str
    d: str
    relation: str


def _person_sentence(rng: random.Random, word: str, female: bool) -> str:
    pronoun, possessive = ("she", "her") if female else ("he", "his")
    templates = [
        f"the {word} said that {pronoun} was tired",
        f"{pronoun} is a {word} and {possessive} home is here",
        f"the {word} smiled and {pronoun} waved",
        f"everyone knows that {pronoun} is the {word}",
    ]
    if word in ROYAL:
        templates += [
            f"the {word} lives in the palace",
            f"the {word} wears a golden crown",
            f"the {word} rules the kingdom",
            f"the royal {word} sat on the throne",
        ]
    else:
        templates += [
            f"the {word} walks to the market",
            f"the {word} works in the village",
            f"the {word} cooks dinner at home",
        ]
    templates += (
        [f"the young {word} plays in the garden", f"the {word} goes to school"]
        if word in YOUNG
        else [f"the old {word} reads the newspaper", f"the {word} pays the bills"]
    )
    sentence = rng.choice(templates)
    return f"{sentence} {rng.choice(FILLER)}" if rng.random() < 0.3 else sentence


def _country_sentence(
    rng: random.Random, country: str, capital: str, language: str
) -> str:
    other = rng.choice([c for c in COUNTRIES if c[0] != country])
    templates = [
        f"{capital} is the capital of {country}",
        f"{capital} is a big city in {country}",
        f"people in {capital} speak {language}",
        f"people in {country} speak {language}",
        f"the city of {capital} is famous",
        f"the city of {capital} has a busy airport",
        f"{country} is a large country",
        f"the country of {country} has many mountains",
        f"we flew from {capital} to {other[1]}",
        f"{country} is a neighbor of {other[0]}",
        f"she learned {language} before moving to {country}",
        f"the {language} language is spoken in {capital}",
    ]
    return rng.choice(templates)


def toy_world_corpus(num_sentences: int = 200_000, seed: int = 0) -> list[list[str]]:
    """Sentences about people and countries whose co-occurrence statistics encode known relations.

    Gender is carried by the pronouns next to a word, royalty by palace/crown words, a capital by
    "city" words plus its country's language: exactly the kind of structure that shows up as
    vector offsets in word2vec (lesson 02 §2.5).
    """
    rng = random.Random(seed)
    sentences = []
    for _ in range(num_sentences):
        if rng.random() < 0.5:
            male, female = rng.choice(GENDER_PAIRS)
            is_female = rng.random() < 0.5
            text = _person_sentence(rng, female if is_female else male, is_female)
        else:
            text = _country_sentence(rng, *rng.choice(COUNTRIES))
        sentences.append(text.split())
    return sentences


def toy_world_analogies() -> list[Analogy]:
    """Every analogy the toy world encodes: gender pairs, capital-of and language-of."""
    analogies = []
    for i, (m1, f1) in enumerate(GENDER_PAIRS):
        for m2, f2 in GENDER_PAIRS[i + 1 :]:
            analogies.append(Analogy(m1, f1, m2, f2, "gender"))
    for i, (country1, capital1, language1) in enumerate(COUNTRIES):
        for country2, capital2, language2 in COUNTRIES[i + 1 :]:
            analogies.append(Analogy(country1, capital1, country2, capital2, "capital"))
            analogies.append(
                Analogy(country1, language1, country2, language2, "language")
            )
    return analogies
