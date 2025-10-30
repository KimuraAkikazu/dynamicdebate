# src/sbd.py
from __future__ import annotations
import os
from functools import lru_cache

@lru_cache(maxsize=1)
def get_nlp():
    import spacy
    model = os.environ.get("SPACY_MODEL", "en_core_web_sm")
    try:
        nlp = spacy.load(model, disable=["ner", "tagger", "lemmatizer"])  # 軽量
    except OSError:
        # モデル未DL時のフォールバック：ルールベース sentencizer
        nlp = spacy.blank("en")
        nlp.add_pipe("sentencizer")
    return nlp

def split_into_sentences(text: str) -> list[str]:
    if not text:
        return []
    nlp = get_nlp()
    doc = nlp(text)
    return [s.text.strip() for s in doc.sents if s.text.strip()]
