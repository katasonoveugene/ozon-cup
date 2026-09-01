from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence

import numpy as np
import pandas as pd

from quality_core.common import clean_text


SUPPLEMENT_CATEGORY = "БАД"
FLAMMABLE_CATEGORY = "Легковоспламеняющиеся"


_SPACE_RE = re.compile(r"\s+")


RULE_PATTERNS: tuple[tuple[str, str], ...] = (
    (
        "explicit_bad",
        r"(?:\bбад(?:ы|ами|ах|ов)?\b|б\s*[.\-/]?\s*а\s*[.\-/]?\s*д\b|"
        r"биологически\w*\s+активн\w*\s+добавк\w*|dietary\s+supplement)",
    ),
    (
        "not_bad",
        r"(?:не\s+(?:явля\w*|отно\w*|счита\w*)[^.!?\n]{0,100}"
        r"(?:\bбад\b|биологически\w*\s+активн\w*)|"
        r"(?:\bбад\b|биологически\w*\s+активн\w*)[^.!?\n]{0,50}"
        r"не\s+(?:явля\w*|отно\w*|счита\w*))",
    ),
    (
        "sport_direct",
        r"спортивн\w*\s+(?:питан\w*|добавк\w*)|питан\w*\s+спортсмен\w*",
    ),
    ("bcaa", r"(?<!\w)(?:bcaa|бцаа|всаа)(?!\w)"),
    ("lcarn", r"(?<!\w)(?:l|л)[\s-]*карнитин\w*"),
    ("protein", r"\bпротеин\w*|\bprotein\b"),
    ("amino", r"аминокислот\w*|amino\s*acid"),
    ("creatine", r"\bкреатин\w*|\bcreatine\w*"),
    ("gainer", r"\bгейнер\w*|\bgainer\w*"),
    ("prework", r"предтрен\w*|предтрениров\w*|pre[- ]?workout"),
    ("fatburn", r"жиросжига\w*|fat\s*burn"),
    ("isotonic", r"\bизотоник\w*|\bisotonic\w*"),
    ("foodadd", r"пищев\w*\s+добавк\w*|food\s+supplement"),
    ("not_drug", r"не\s+явля\w*\s+лекарств\w*"),
    ("caps", r"\bкапсул\w*|\bтаблет\w*|\bпорош\w*"),
    ("matches", r"\bспич(?:к\w*|ечн\w*)"),
    ("lighter", r"\bзажигал\w*"),
    ("firestarter", r"\b(?:огнив\w*|растопк\w*|розжиг\w*)"),
    ("dryfuel", r"сух\w*\s+(?:горюч\w*|спирт\w*)"),
    (
        "ignitionfuel",
        r"(?:жидкост\w*|гель\w*|средств\w*)\s+для\s+розжиг\w*",
    ),
    ("roll", r"\b(?:ролл\w*|кубик\w*|палочк\w*|брусочк\w*)"),
    ("charcoal", r"\b(?:угол\w*|брик\w*|топлив\w*)"),
    (
        "gas",
        r"\b(?:газ\w*|бутан\w*|пропан\w*|мапп\w*|бензин\w*|нефрас\w*)",
    ),
    ("can", r"\b(?:баллон\w*|балон\w*|картридж\w*|аэрозол\w*)"),
    (
        "firework",
        r"\b(?:салют\w*|фейерверк\w*|петард\w*|пиротех\w*|бенгальск\w*|"
        r"хлопушк\w*|цветн\w*\s+дым\w*|дым\w*\s+шашк\w*)",
    ),
    ("candle", r"\bсвеч\w*"),
    ("burner", r"\bгорелк\w*"),
    ("device", r"\b(?:мангал\w*|грил\w*|барбекю|печ\w*|камин\w*|плит\w*)"),
    ("kit", r"\b(?:комплект\w*|набор\w*|входит\w*|поставля\w*|включа\w*)"),
    (
        "accessory",
        r"\b(?:чехл\w*|футляр\w*|держател\w*|адаптер\w*|переходник\w*|"
        r"насадк\w*|ремкомплект\w*|коробк\w*)",
    ),
    (
        "empty",
        r"(?:без\s+(?:газ\w*|топлив\w*|горюч\w*|жидкост\w*|содержим\w*|"
        r"спич\w*)|пуст\w*|не\s+входит\w*\s+в\s+комплект)",
    ),
    (
        "built_in",
        r"(?:пьезо\w*|встроенн\w*\s+(?:поджиг\w*|зажиг\w*))",
    ),
    (
        "neg_flame",
        r"(?:не\s+(?:явля\w*\s+)?(?:пожароопас\w*|воспламен\w*|горюч\w*)|"
        r"непожароопас\w*|негорюч\w*)",
    ),
    ("pneumatic", r"пневмо\w*|сжат\w*\s+воздух\w*"),
    ("electric", r"электр\w*"),
    (
        "drawing",
        r"(?:угол\w*\s+для\s+рисован\w*|активированн\w*\s+угол\w*|"
        r"угольн\w*\s+фильтр\w*)",
    ),
)


_COMPILED_RULE_PATTERNS = tuple(
    (name, re.compile(pattern, flags=re.IGNORECASE)) for name, pattern in RULE_PATTERNS
)


def normalize_rule_text(value: object) -> str:
    text = unicodedata.normalize("NFKC", clean_text(value)).casefold().replace("ё", "е")
    return _SPACE_RE.sub(" ", text).strip()


def normalized_fields(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    names = np.asarray([normalize_rule_text(value) for value in frame["name"]], dtype=object)
    descriptions = np.asarray(
        [normalize_rule_text(value) for value in frame["description"]], dtype=object
    )
    return names, descriptions


def rule_feature_names() -> list[str]:
    result = ["name_len", "description_len"]
    for pattern_name, _ in RULE_PATTERNS:
        result.extend([f"{pattern_name}__name", f"{pattern_name}__description"])
    return result


def build_rule_features(
    frame: pd.DataFrame,
    fields: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, list[str]]:
    names, descriptions = fields if fields is not None else normalized_fields(frame)
    columns: list[np.ndarray] = [
        np.log1p(np.fromiter((len(value) for value in names), dtype=np.float32)),
        np.log1p(np.fromiter((len(value) for value in descriptions), dtype=np.float32)),
    ]
    for _, pattern in _COMPILED_RULE_PATTERNS:
        for values in (names, descriptions):
            counts = np.fromiter(
                (sum(1 for _ in pattern.finditer(value)) for value in values),
                dtype=np.float32,
            )
            columns.append(np.log1p(counts))
    return np.column_stack(columns).astype(np.float32), rule_feature_names()


def _contains(values: Sequence[str], pattern: str) -> np.ndarray:
    regex = re.compile(pattern, flags=re.IGNORECASE)
    return np.fromiter((regex.search(value) is not None for value in values), dtype=bool)


def conservative_override_masks(
    frame: pd.DataFrame,
    fields: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    names, descriptions = fields if fields is not None else normalized_fields(frame)
    text = np.asarray([f"{name} {description}" for name, description in zip(names, descriptions)])
    categories = frame["category"].astype(str).to_numpy()
    supplement = categories == SUPPLEMENT_CATEGORY
    flammable = categories == FLAMMABLE_CATEGORY

    explicit_any = _contains(
        text,
        r"(?:\bбад(?:ы|ами|ах|ов)?\b|биологически\w*\s+активн\w*\s+добавк\w*|"
        r"dietary\s+supplement)",
    )
    explicit_title = _contains(
        names,
        r"(?:\bбад(?:ы|ами|ах|ов)?\b|биологически\w*\s+активн\w*\s+добавк\w*|"
        r"dietary\s+supplement)",
    )
    negative_statement = _contains(
        text,
        r"(?:продукт\s+не\s+относится\s+к\s+категории\s+бад|"
        r"не\s+явля\w*\s+(?:биологически\w*\s+активн\w*\s+добавк\w*|бад\w*)|"
        r"(?<!\w)не\s+бад(?!\w))",
    )
    sports_statement = _contains(
        text,
        r"спортивн\w*\s+(?:питан\w*|добавк\w*)",
    )
    supplement_accessory = _contains(
        names,
        r"\b(?:таблетниц\w*|органайзер\w*|контейнер\w*|коробочк\w*|"
        r"футляр\w*|чехл\w*)",
    )
    supplement_positive = (
        supplement
        & explicit_title
        & ~negative_statement
        & ~sports_statement
        & ~supplement_accessory
    )
    supplement_negative = supplement & (
        (negative_statement & ~sports_statement) | (sports_statement & ~explicit_any)
    )

    flammable_positive = flammable & (
        _contains(
            names,
            r"(?:maclay\s+)?мангал\w*\s+однораз\w*[^\n]{0,100}"
            r"(?:с\s+угл\w*|в\s+комплект\w*\s+с\s+угл\w*)",
        )
        | _contains(names, r"брик\w*\s+для\s+грил\w*\s+weber")
        | _contains(
            names,
            r"сух\w*\s+горюч\w*[^\n]{0,80}(?:с\s+поджиг\w*|спич\w*)",
        )
    )
    negative_patterns = (
        r"(?:адаптер\w*|переходник\w*)[^\n]{0,80}(?:газ\w*|баллон\w*)",
        r"хлопушк\w*\s+пневмат\w*|пневмо\w*хлопушк\w*",
        r"(?:спичечниц\w*|футляр\w*|коробк\w*)[^\n]{0,80}"
        r"(?:без\s+спич\w*|для\s+спич\w*)|без\s+спич\w*",
        r"(?:чехл\w*|футляр\w*|ремкомплект\w*|кремн\w*|фитил\w*)"
        r"[^\n]{0,80}зажигал\w*",
        r"баллон\w*[^\n]{0,80}(?:пуст\w*|без\s+газ\w*)",
        r"горелк\w*[^\n]{0,160}(?:баллон\w*\s+"
        r"(?:не\s+входит|без\s+газ\w*)|без\s+баллон\w*)",
    )
    flammable_negative = np.zeros(len(frame), dtype=bool)
    for pattern in negative_patterns:
        flammable_negative |= flammable & _contains(names, pattern)

    return {
        "supplement_positive": supplement_positive,
        "supplement_negative": supplement_negative,
        "flammable_positive": flammable_positive,
        "flammable_negative": flammable_negative,
    }


def apply_conservative_overrides(
    frame: pd.DataFrame,
    predictions: Sequence[int],
    fields: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    output = np.asarray(predictions, dtype=np.int8).copy()
    masks = conservative_override_masks(frame, fields=fields)
    output[masks["supplement_positive"]] = 1
    output[masks["supplement_negative"]] = 0
    output[masks["flammable_positive"]] = 1
    output[masks["flammable_negative"]] = 0
    return output
