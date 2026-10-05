"""Match source pronunciations to the shared, filename-labelled Voice library.

English labels use CMU pronunciations. Vietnamese phonetic filenames are
curated sound spellings, not English vocabulary. Stress/duration variants may
share a sound; consonants and vowel identity must match. This is a dictionary
comparison of ASR output, not acoustic verification of a recording.
"""
from functools import lru_cache
import math
from pathlib import PureWindowsPath
import re

from autoedit.audio.taxonomy import lookup, normalize, spoken_words, transcript_token

# Explicit English labels: short Vietnamese spellings such as a/ai/em/ba must
# not become English words merely because a dictionary contains that spelling.
ENGLISH_LABELS = {
    "my": "my", "kay": "kay", "let": "let", "one": "one", "two": "two",
    "three": "three", "five": "five", "ten": "ten", "ok": "okay", "hi": "hi",
    "you": "you", "love": "love", "bye": "bye", "go": "go", "go2": "go",
    "yeah": "yeah", "yeahh": "yeah", "yeah trầm": "yeah", "yay": "yay",
    "yes": "yes", "yess": "yes", "no": "no", "no2": "no", "what": "what",
    "whatt": "what", "why": "why", "who": "who", "woww": "wow", "wowww": "wow",
    "hey to": "hey", "heyy": "hey", "heyyy": "hey", "wait": "wait",
    "eww": "ew", "shit": "shit", "fight": "fight", "oh": "oh", "ohh": "oh",
    "ohhh": "oh", "oi": "oi", "hmm": "hmm",
}

# ARPAbet approximations of the user's Vietnamese transcriptions. Vowel accents
# remain significant; tone and elongation do not change the matching key.
# No English lexical aliases are invented here.
PHONETIC_LABELS = {
    "a": "AA", "aa": "AA", "ayy": "EY", "âyy": "EY",
    "oa": "W AA", "oaa": "W AA", "oo": "UW", "u": "UW", "wuu": "W UW",
    "huu": "HH UW", "hự": "HH AH", "po": "P AO",
    "âu": "AW", "âuu": "AW", "âuu2": "AW", "ấu": "AW", "ầu": "AW", "âu nam": "AW",
    "da": "D AA", "daa": "D AA", "daaa": "D AA", "gi": "JH IY", "giii": "JH IY",
    "dii": "D IY", "diuu": "D IY UW", "gơn": "G ER N", "trâu": "CH AW",
    "rát": "R AA T", "mái": "M AY", "mai": "M AY", "zai": "Z AY", "ái": "AY",
    "nhăm": "N AH M", "hap": "HH AA P", "dam": "D AE M", "damm": "D AE M",
    "ghim": "G IH M", "sa": "S AA", "ca": "K AA", "ga": "G AA", "gaa": "G AA",
    "ba": "B AA", "ai": "AY", "aii": "AY", "em": "EH M", "iu": "Y UW",
    "quoạc": "K W AA K", "rô": "R OW", "tok": "T AO K", "đi": "D IY",
    "đơm": "D ER M", "ắc": "AA K", "ết": "EH T", "ựa": "AH",
    "ê": "EY", "hơ": "HH ER", "hớ": "HH ER", "ó": "AO", "ôi": "OY",
    "pii": "P IY", "ye": "Y EH", "yaayy": "Y EY", "whu": "HH UW",
    "woa": "W OW", "woa-nữ": "W OW", "huah": "HH W AA",
}
NUMBER_WORDS = {"1": "one", "2": "two", "3": "three", "5": "five", "10": "ten"}
INTERJECTIONS = {"ew": (("IY", "UW"),), "eww": (("IY", "UW"),)}
ELONGATED_WORDS = {
    "oh": r"oh+", "ah": r"ah+", "hey": r"hey+", "yeah": r"yeah+",
    "yes": r"yes+", "wow": r"wow+", "ew": r"ew+", "no": r"no+",
    "what": r"wha+t+", "why": r"why+", "who": r"who+",
}


def source_token(word):
    token = normalize(word).replace("’", "'").strip(".,!?;:\"()[]'“”‘")
    token = NUMBER_WORDS.get(token, token)
    token = {"woah": "whoa"}.get(token, token)
    for canonical, pattern in ELONGATED_WORDS.items():
        if re.fullmatch(pattern, token):
            return canonical
    return token


@lru_cache(maxsize=1)
def _dictionary():
    import cmudict
    return cmudict.dict()


@lru_cache(maxsize=8192)
def word_pronunciations(word):
    token = source_token(word)
    # Retain apostrophes: I'm must not be mistaken for the complete word I.
    if token in INTERJECTIONS:
        return INTERJECTIONS[token]
    return tuple(sorted({pronunciation_key(phones) for phones in _dictionary().get(token, [])}))


def pronunciation_key(phones):
    phones = tuple(re.sub(r"\d", "", p) for p in phones)
    # CMU lists both /w/ and aspirated /hw/ readings for the same 'wh' word.
    # Treat this dialect variation consistently, without collapsing vowels.
    return phones[1:] if phones[:2] == ("HH", "W") else phones


def item_stem(item):
    return normalize(item.get("stem") or PureWindowsPath(item.get("path") or item.get("name") or "").stem)


def item_pronunciations(item):
    stem = item_stem(item)
    if stem in PHONETIC_LABELS:
        return (pronunciation_key(PHONETIC_LABELS[stem].split()),)
    if stem in ENGLISH_LABELS:
        return word_pronunciations(ENGLISH_LABELS[stem])
    return ()


def active_bounds(item):
    lo = float(item.get("active_start_sec", item.get("segment_start_sec", 0)) or 0)
    hi = float(item.get("active_end_sec", item.get("segment_end_sec", item.get("duration_sec", 0))) or 0)
    return lo, hi


def valid_sound(sound):
    try:
        start, end = float(sound["start"]), float(sound["end"])
        prob = float(sound.get("prob", 0))
        return all(math.isfinite(v) for v in (start, end, prob)) and 0 <= start < end and 0 <= prob <= 1
    except (KeyError, TypeError, ValueError):
        return False


class VoiceIndex:
    """Build once per run; expanded segments do not create duplicate votes."""
    def __init__(self, items):
        self.words, self.phones, self.actions = {}, {}, {}
        seen = set()
        for item in items:
            if item["path"] in seen:
                continue
            seen.add(item["path"])
            lo, hi = active_bounds(item)
            if not all(math.isfinite(v) for v in (lo, hi)) or not 0 <= lo < hi:
                continue
            row = lookup(item_stem(item))
            if not row:
                continue
            sound = row[1]
            if sound.action == "NONVOCAL":
                continue
            if sound.action != "SPEECH":
                self.actions.setdefault(sound.action, []).append(item)
            # Old taxonomy aliases conflate some Vietnamese sounds with words.
            # Only actual English labels use lexical matching. For instance,
            # I -> ai must pass the pronunciation comparison, not an alias.
            if sound.action == 'SPEECH' and item_stem(item) in ENGLISH_LABELS:
                for word in spoken_words(item):
                    self.words.setdefault(transcript_token(word), []).append(item)
            if sound.action == "SPEECH":
                for phones in item_pronunciations(item):
                    self.phones.setdefault(phones, []).append(item)

    def match(self, sound):
        if sound.get("action"):
            return [dict(path=item["path"], kind="source-audio-event", phones=[])
                    for item in self.actions.get(sound["action"], [])]
        word = sound.get("word") or ""
        exact = {item["path"] for item in self.words.get(transcript_token(source_token(word)), [])}
        found = {path: {"path": path, "kind": "source-audio-whisper-exact-word", "phones": []}
                 for path in sorted(exact)}
        pronunciations = word_pronunciations(word)
        # ASR text does not tell us which pronunciation was actually spoken.
        # A phonetic candidate must support every listed reading; otherwise a
        # reduced 'to' /tə/ would incorrectly use 'two' /tu:/. Exact English
        # word labels above remain valid without guessing a different word.
        compatible = None
        for phones in pronunciations:
            paths = {item["path"] for item in self.phones.get(phones, [])}
            compatible = paths if compatible is None else compatible & paths
        for phones in pronunciations:
            for item in self.phones.get(phones, []):
                if item["path"] in (compatible or set()):
                    found.setdefault(item["path"], {"path": item["path"],
                        "kind": "source-audio-pronunciation", "phones": list(phones)})
        return list(found.values())


def source_sounds(speech):
    return sorted((s for s in speech.get("words", []) + speech.get("events", []) if valid_sound(s)),
                  key=lambda s: (s["start"], s["end"]))


def build_sound_anchors(items, speech):
    """Strong onsets rank first; supported medium-confidence speech needs review."""
    index = VoiceIndex(items)
    anchors = []
    for sound in source_sounds(speech):
        threshold = .75 if sound.get("action") else .6
        if sound["prob"] < threshold:
            continue
        matches = index.match(sound)
        if matches:
            anchors.append({**sound, "matches": matches,
                            "confidence_tier": 1 if sound["prob"] >= .8 and not sound.get("action") else 2})
    return anchors
