import pickle
import difflib
from pathlib import Path

PICKLE_PATH = Path(__file__).parent / "ocr_model.pkl"


class OCRCorrectionModel:
    def __init__(self):
        self.corrections: dict[str, str] = {}
        self.vocabulary: list[str] = []

    def fit(self, pairs: list[tuple[str, str]]) -> None:
        for ocr_word, correct_word in pairs:
            key = ocr_word.strip().lower()
            val = correct_word.strip()
            self.corrections[key] = val
            if val and val not in self.vocabulary:
                self.vocabulary.append(val)

    def predict(self, word: str) -> str:
        key = word.strip().lower()
        if key in self.corrections:
            return self.corrections[key]
        if self.vocabulary:
            lower_vocab = [v.lower() for v in self.vocabulary]
            matches = difflib.get_close_matches(key, lower_vocab, n=1, cutoff=0.75)
            if matches:
                return self.vocabulary[lower_vocab.index(matches[0])]
        return word

    def correct_text(self, text: str) -> str:
        return " ".join(self.predict(w) for w in text.split())

    @property
    def correction_count(self) -> int:
        return len(self.corrections)


def align_words(ocr_text: str, correct_text: str) -> list[tuple[str, str]]:
    ocr_words = ocr_text.split()
    correct_words = correct_text.split()
    matcher = difflib.SequenceMatcher(
        None,
        [w.lower() for w in ocr_words],
        [w.lower() for w in correct_words],
    )
    pairs = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "replace" and (i2 - i1) == (j2 - j1):
            for ocr_w, cor_w in zip(ocr_words[i1:i2], correct_words[j1:j2]):
                if ocr_w.lower() != cor_w.lower():
                    pairs.append((ocr_w, cor_w))
    return pairs


def save_model(model: OCRCorrectionModel, path: Path = PICKLE_PATH) -> None:
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(path: Path = PICKLE_PATH) -> OCRCorrectionModel | None:
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None
