"""Reviewable inventory of all 127 recordings in D:\\Editor\\Voice.

One row per actual filename stem; accents are significant (oi != ôi).
Columns: label, visemes, transcript word(s), action, needs_review.
NEUTRAL is a voice group, not a replacement for the recording's word label.
Ambiguous stems ai, aii, hi, bye, gru, rô, ó, ôi, ê, ựa were confirmed
NEUTRAL by the user. Review the full inventory before the first production run.
Visemes: A=open, E=wide, O=rounded, U=narrow, M=closed, F=lip/teeth.
"""
from dataclasses import dataclass
from pathlib import PureWindowsPath
import re
import unicodedata


@dataclass(frozen=True)
class Sound:
    label: str
    visemes: str
    words: tuple[str, ...] = ()
    action: str = "SPEECH"
    review: bool = False


VOWEL = {
    "a": Sound("A", "A"),
    "aa": Sound("A", "A"),
    "ayy": Sound("E", "E"),
    "âyy": Sound("E", "E"),
    "oa": Sound("O", "OA"),
    "oaa": Sound("O", "OA"),
    "oo": Sound("U", "U"),
    "u": Sound("U", "U"),
    "wuu": Sound("U", "U"),
    "huu": Sound("U", "U"),
    "hự": Sound("U", "U"),
    "po": Sound("O", "MO"),
    "âu": Sound("A", "AU"),
    "âuu": Sound("A", "AU"),
    "âuu2": Sound("A", "AU"),
    "ấu": Sound("A", "AU"),
    "ầu": Sound("A", "AU"),
}

NEUTRAL = {
    "da": Sound("DA", "A", ("da",)),
    "daa": Sound("DA", "A", ("da",)),
    "daaa": Sound("DA", "A", ("da",)),
    "gi": Sound("GI", "E", ("gi",)),
    "giii": Sound("GI", "E", ("gi",)),
    "dii": Sound("DI", "E", ("đi",)),
    "diuu": Sound("DIUU", "EU", ("diuu",)),
    "gơn": Sound("GƠN", "E", ("gơn",)),
    "trâu": Sound("TRÂU", "AU", ("trâu",)),
    "rát": Sound("RÁT", "A", ("rát",)),
    "mái": Sound("MÁI", "MAE", ("mái",)),
    "mai": Sound("MAI", "MAE", ("mai",)),
    "my": Sound("MY", "MAE", ("my",)),
    "zai": Sound("ZAI", "AE", ("zai",)),
    "ái": Sound("ÁI", "AE", ("ái",)),
    "âu nam": Sound("ÂU", "AU", ("âu",)),
    "nhăm": Sound("NHĂM", "AM", ("nhăm",)),
    "hap": Sound("HAP", "AM", ("hap",)),
    "dam": Sound("DAM", "AM", ("dam",)),
    "damm": Sound("DAM", "AM", ("dam",)),
    "ghim": Sound("GHIM", "EM", ("ghim",)),
    "sa": Sound("SA", "A", ("sa",)),
    "ca": Sound("CA", "A", ("ca",)),
    "ga": Sound("GA", "A", ("ga",)),
    "gaa": Sound("GA", "A", ("ga",)),
    "ba": Sound("BA", "MA", ("ba",)),
    "ai": Sound("AI", "AE", ("ai", "i", "eye")),
    "aii": Sound("AI", "AE", ("ai", "i", "eye")),
    "em": Sound("EM", "EM", ("em",)),
    "iu": Sound("IU", "EU", ("iu",)),
    "quoạc": Sound("QUOẠC", "UA", ("quoạc",)),
    "rô": Sound("RÔ", "O", ("rô",)),
    "tok": Sound("TOK", "O", ("tok",)),
    "đi": Sound("DI", "E", ("đi",)),
    "đơm": Sound("ĐƠM", "EM", ("đơm",)),
    "ắc": Sound("ẮC", "A", ("ắc",)),
    "ết": Sound("ẾT", "E", ("ết",)),
    "ợ": Sound("Ợ", "E", ("ợ",)),
    "ựa": Sound("ỰA", "EA", ("ựa",)),
    "ê": Sound("Ê", "E", ("ê",)),
    "hơ": Sound("HƠ", "E", ("hơ",)),
    "hớ": Sound("HỚ", "E", ("hớ",)),
    "ó": Sound("Ó", "O", ("ó",)),
    "ôi": Sound("ÔI", "OE", ("ôi",)),
    "pii": Sound("PII", "ME", ("pii",)),
    "kay": Sound("KAY", "E", ("kay",)),
    "let": Sound("LET", "E", ("let",)),
    "one": Sound("ONE", "UA", ("one", "1")),
    "two": Sound("TWO", "U", ("two", "2")),
    "three": Sound("THREE", "E", ("three", "3")),
    "five": Sound("FIVE", "FAEF", ("five", "5")),
    "ten": Sound("TEN", "E", ("ten", "10")),
    "ok": Sound("OK", "OE", ("ok", "okay")),
    "hi": Sound("HI", "AE", ("hi",)),
    "you": Sound("YOU", "EU", ("you",)),
    "love": Sound("LOVE", "AF", ("love",)),
    "bye": Sound("BYE", "MAE", ("bye",)),
    "hmm": Sound("HMM", "M", ("hmm",)),
    "gru": Sound("GROAN", "U", action="GROAN"),
    "grunt": Sound("GRUNT", "A", action="GROAN"),
    "go": Sound("GO", "OU", ("go",)),
    "go2": Sound("GO", "OU", ("go",)),
}

EMOTION = {
    "cười": Sound("LAUGH", "A", action="LAUGH"),
    "cười trẻ con": Sound("LAUGH_KID", "A", action="LAUGH"),
    "cười đàn ông": Sound("LAUGH_MAN", "A", action="LAUGH"),
    "ha": Sound("LAUGH", "A", ("ha",), "LAUGH"),
    "hah": Sound("LAUGH", "A", ("hah",), "LAUGH"),
    "heha": Sound("LAUGH", "A", action="LAUGH"),
    "heheha": Sound("LAUGH", "A", action="LAUGH"),
    "yeah": Sound("YEAH", "EA", ("yeah",)),
    "yeahh": Sound("YEAH", "EA", ("yeah",)),
    "yeah trầm": Sound("YEAH", "EA", ("yeah",)),
    "ye": Sound("YEAH", "E", ("yeah",)),
    "yay": Sound("YAY", "E", ("yay",)),
    "yaayy": Sound("YAY", "E", ("yay",)),
    "yes": Sound("YES", "E", ("yes",)),
    "yess": Sound("YES", "E", ("yes",)),
    "no": Sound("NO", "OU", ("no",)),
    "no2": Sound("NO", "OU", ("no",)),
    "what": Sound("WHAT", "UA", ("what",)),
    "whatt": Sound("WHAT", "UA", ("what",)),
    "why": Sound("WHY", "UAE", ("why",)),
    "who": Sound("WHO", "U", ("who",)),
    "whu": Sound("WHO", "U", ("who",)),
    "woa": Sound("WOA", "UA", ("wow", "woah", "whoa")),
    "woa-nữ": Sound("WOA", "UA", ("wow", "woah", "whoa")),
    "woww": Sound("WOW", "UAU", ("wow",)),
    "wowww": Sound("WOW", "UAU", ("wow",)),
    "hey to": Sound("HEY", "E", ("hey",)),
    "heyy": Sound("HEY", "E", ("hey",)),
    "heyyy": Sound("HEY", "E", ("hey",)),
    "wait": Sound("WAIT", "UE", ("wait",)),
    "eww": Sound("EWW", "EU", ("eww", "ew")),
    "shit": Sound("SHIT", "E", ("shit",)),
    "fight": Sound("FIGHT", "FAE", ("fight",)),
    "oh": Sound("OH", "O", ("oh",)),
    "ohh": Sound("OH", "O", ("oh",)),
    "ohhh": Sound("OH", "O", ("oh",)),
    "oi": Sound("OI", "OE", ("oi",)),
    "huah": Sound("HUAH", "UA", ("huah",)),
    "thở dài": Sound("SIGH", "A", action="SIGH"),
    "thở dài 2": Sound("SIGH", "A", action="SIGH"),
    "ho khụ khụ": Sound("COUGH", "A", action="COUGH"),
    "uuugh": Sound("GROAN", "A", ("ugh",), "GROAN"),
}

NONVOCAL = {
    "còi oto": Sound("HORN", "", action="NONVOCAL"),
    "ting": Sound("CHIME", "", action="NONVOCAL"),
    "tinggg": Sound("BELL", "", action="NONVOCAL"),
    "beepp": Sound("BEEP", "", action="NONVOCAL"),
    "xì hơi": Sound("RASPBERRY", "", action="NONVOCAL"),
    "xsxsxsx": Sound("RASPBERRY", "", action="NONVOCAL"),
}

GROUPS = {"VOWEL": VOWEL, "NEUTRAL": NEUTRAL, "EMOTION": EMOTION, "NONVOCAL": NONVOCAL}
TAXONOMY = {stem: (group, sound) for group, rows in GROUPS.items() for stem, sound in rows.items()}
assert len(TAXONOMY) == sum(map(len, GROUPS.values())), "A filename belongs to exactly one group"
ALIASES = {"haha": "heha", "hey": "heyy", "wow": "woww", "beep": "beepp", "bell": "tinggg"}


def normalize(text):
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


def lookup(stem):
    key = normalize(stem)
    return TAXONOMY.get(ALIASES.get(key, key))


def filename_label(stem):
    row = lookup(stem)
    return row[1].label if row else None


def item_group(item):
    stem = item.get("stem") or PureWindowsPath(item.get("path") or item.get("name") or "").stem
    row = lookup(stem)
    return row[0] if row else item.get("group")


def transcript_token(text):
    return re.sub(r"[^\w]+", "", normalize(text).replace("'", ""))


def spoken_words(item):
    stem = item.get("stem") or PureWindowsPath(item.get("path") or item.get("name") or "").stem
    row = lookup(stem)
    return row[1].words if row else ()


# Transcript vocabulary derives from the inventory; only out-of-library tokens
# are added here. No acoustic classification may infer age or gender of laughs.
TOKEN_CATEGORY = {word: sound.label for group, sound in TAXONOMY.values() for word in sound.words}
TOKEN_CATEGORY.update({"wow": EMOTION["woww"].label, "woa": EMOTION["woa"].label,
                       "horn": NONVOCAL["còi oto"].label, "chime": NONVOCAL["ting"].label,
                       "beep": NONVOCAL["beepp"].label, "bell": NONVOCAL["tinggg"].label,
                       "cry": "CRY", "four": "FOUR", "eight": "EIGHT", "woo": "WOO",
                       "huh": "HUH", "ah": "AH", "uh": "UH", "ho": "LAUGH",
                       "boo": "BOO", "bruh": "BRUH", "duh": "DUH", "come": "COME",
                       "on": "ON", "lets": "LET", "i": "I"})
