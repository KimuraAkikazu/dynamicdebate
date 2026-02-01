# -*- coding: utf-8 -*-
"""
Streaming-friendly sentence boundary detector.

StreamSBD collects token chunks and emits a sentence string as soon as it
detects a terminator. The detector is thread-safe so it can be shared between
producer (Speaker) and consumer (Manager) threads.
"""
from __future__ import annotations

import os
import threading
from functools import lru_cache
from typing import List


class StreamSBD:
    def __init__(self, terminators: list[str] | None = None) -> None:
        self.buffer = ""
        self.terminators = terminators or ["。", "？", "！", ".", "?", "!", "\n"]
        self._lock = threading.Lock()

    def push(self, token: str) -> str | None:
        """
        Add a token fragment. When the buffer ends with any terminator,
        return the sentence (buffer is cleared). Otherwise return None.
        """
        if not token:
            return None
        with self._lock:
            self.buffer += token
            if any(self.buffer.endswith(t) for t in self.terminators):
                sentence = self.buffer
                self.buffer = ""
                return sentence.strip()
        return None

    def flush(self) -> str | None:
        """
        Force emit the remaining buffer (if any) and clear it.
        """
        with self._lock:
            if self.buffer.strip():
                sentence = self.buffer
                self.buffer = ""
                return sentence.strip()
        return None


@lru_cache(maxsize=1)
def get_nlp():
    import spacy

    model = os.environ.get("SPACY_MODEL", "en_core_web_sm")
    try:
        nlp = spacy.load(model, disable=["ner", "tagger", "lemmatizer"])
    except OSError:
        # Fallback: lightweight rule-based sentencizer
        nlp = spacy.blank("en")
        nlp.add_pipe("sentencizer")
    return nlp


def split_into_sentences(text: str) -> List[str]:
    """
    Non-streaming sentence split (kept for backward compatibility).
    """
    if not text:
        return []
    nlp = get_nlp()
    doc = nlp(text)
    return [s.text.strip() for s in doc.sents if s.text.strip()]

