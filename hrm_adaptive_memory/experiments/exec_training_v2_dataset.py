"""ANSWER_PROBE_GATE_V2 training-suite generator: the ANSWER_NOW_viable family.

Per configs/gate_answer_probe_v2_design.json PHASE_1: five new hand-verified
GK categories, additive to (never modifying) the frozen, consumed V1 tables
in hrm_adaptive_memory/experiments/exec_training_dataset.py. Reuses that
module's ExecTrainingTask dataclass and verify_native_parsing() unchanged --
both are already generic over any (subject, relation, question, answer)
table, not V1-specific.

Every fact below was chosen for the same reason V1's capitals/element-symbol
tables were: individually true, stable, and unambiguous, not merely
internally consistent. Currency and continent facts were cross-checked
against a live 2026 currency-code reference (geocountries.com/country/
currencies, fetched 2026-08-10) rather than relying on recall alone, since
those two categories are the ones most likely to have changed recently.
Countries/currencies flagged as unstable, redenominated, or ambiguous in
that source (Zimbabwe, Venezuela, and any entry contradicted by
independently-known recent history such as Croatia's 2023 Euro adoption,
which the fetched source had NOT updated) are excluded outright rather than
corrected by hand -- if a source is already known to be stale on one fact,
it is not trusted for adjacent facts in the same category either.
"""
from __future__ import annotations

from hrm_adaptive_memory.experiments.exec_training_dataset import (  # noqa: F401
    ExecTrainingTask, ParserVerificationError, verify_native_parsing)

_RELATION_BY_DOMAIN_V2 = {
    "atomic_number": "atomic number",
    "currency": "currency",
    "continent": "continent",
    "state_capital": "state capital",
    "planet_order": "position from sun",
}

# --- atomic_number: same 36 elements as V1's element_symbols, new relation -
_ATOMIC_NUMBERS: tuple[tuple[str, str], ...] = (
    ("Hydrogen", "1"), ("Helium", "2"), ("Lithium", "3"), ("Carbon", "6"),
    ("Nitrogen", "7"), ("Oxygen", "8"), ("Fluorine", "9"), ("Neon", "10"),
    ("Sodium", "11"), ("Magnesium", "12"), ("Aluminum", "13"), ("Silicon", "14"),
    ("Phosphorus", "15"), ("Sulfur", "16"), ("Chlorine", "17"), ("Potassium", "19"),
    ("Calcium", "20"), ("Iron", "26"), ("Copper", "29"), ("Zinc", "30"),
    ("Silver", "47"), ("Tin", "50"), ("Iodine", "53"), ("Gold", "79"),
    ("Mercury", "80"), ("Lead", "82"), ("Nickel", "28"), ("Titanium", "22"),
    ("Chromium", "24"), ("Manganese", "25"), ("Cobalt", "27"), ("Platinum", "78"),
    ("Uranium", "92"), ("Barium", "56"), ("Krypton", "36"), ("Xenon", "54"),
)

# --- country_currency: 80 sovereign countries, single stable currency name -
# Excludes Zimbabwe (multi-currency/ZiG instability), Venezuela (repeated
# Bolivar redenomination), North Korea and Cuba (unverifiable/convoluted
# recent currency history). 20 Eurozone + 60 distinct-currency countries.
_CURRENCIES: tuple[tuple[str, str], ...] = (
    # Eurozone (20) -- Croatia included per independently-known 2023 adoption,
    # not per the fetched source (which was stale on this one entry).
    ("Austria", "Euro"), ("Belgium", "Euro"), ("Croatia", "Euro"),
    ("Cyprus", "Euro"), ("Estonia", "Euro"), ("Finland", "Euro"),
    ("France", "Euro"), ("Germany", "Euro"), ("Greece", "Euro"),
    ("Ireland", "Euro"), ("Italy", "Euro"), ("Latvia", "Euro"),
    ("Lithuania", "Euro"), ("Luxembourg", "Euro"), ("Malta", "Euro"),
    ("Netherlands", "Euro"), ("Portugal", "Euro"), ("Slovakia", "Euro"),
    ("Slovenia", "Euro"), ("Spain", "Euro"),
    # Distinct currencies (60)
    ("Canada", "Canadian Dollar"), ("Mexico", "Mexican Peso"),
    ("Brazil", "Brazilian Real"), ("Argentina", "Argentine Peso"),
    ("Chile", "Chilean Peso"), ("Colombia", "Colombian Peso"),
    ("Peru", "Peruvian Sol"), ("Switzerland", "Swiss Franc"),
    ("Norway", "Norwegian Krone"), ("Sweden", "Swedish Krona"),
    ("Denmark", "Danish Krone"), ("Iceland", "Icelandic Krona"),
    ("Poland", "Polish Zloty"), ("Romania", "Romanian Leu"),
    ("Russia", "Russian Ruble"), ("Ukraine", "Ukrainian Hryvnia"),
    ("Japan", "Japanese Yen"), ("China", "Chinese Yuan"),
    ("India", "Indian Rupee"), ("Indonesia", "Indonesian Rupiah"),
    ("Malaysia", "Malaysian Ringgit"), ("Singapore", "Singapore Dollar"),
    ("Thailand", "Thai Baht"), ("Vietnam", "Vietnamese Dong"),
    ("Pakistan", "Pakistani Rupee"), ("Bangladesh", "Bangladeshi Taka"),
    ("Nepal", "Nepalese Rupee"), ("Australia", "Australian Dollar"),
    ("Fiji", "Fijian Dollar"), ("Nigeria", "Nigerian Naira"),
    ("Kenya", "Kenyan Shilling"), ("Egypt", "Egyptian Pound"),
    ("Morocco", "Moroccan Dirham"), ("Ghana", "Ghanaian Cedi"),
    ("Tanzania", "Tanzanian Shilling"), ("Uganda", "Ugandan Shilling"),
    ("Ethiopia", "Ethiopian Birr"), ("Zambia", "Zambian Kwacha"),
    ("Malawi", "Malawian Kwacha"), ("Botswana", "Botswana Pula"),
    ("Israel", "Israeli Shekel"), ("Jordan", "Jordanian Dinar"),
    ("Kuwait", "Kuwaiti Dinar"), ("Bahrain", "Bahraini Dinar"),
    ("Oman", "Omani Rial"), ("Iraq", "Iraqi Dinar"),
    ("Iran", "Iranian Rial"), ("Afghanistan", "Afghan Afghani"),
    ("Kazakhstan", "Kazakhstani Tenge"), ("Mongolia", "Mongolian Togrog"),
    ("Cambodia", "Cambodian Riel"), ("Laos", "Lao Kip"),
    ("Guatemala", "Guatemalan Quetzal"), ("Honduras", "Honduran Lempira"),
    ("Nicaragua", "Nicaraguan Cordoba"), ("Paraguay", "Paraguayan Guarani"),
    ("Uruguay", "Uruguayan Peso"), ("Bolivia", "Bolivian Boliviano"),
    ("Namibia", "Namibian Dollar"), ("Mozambique", "Mozambican Metical"),
    ("Algeria", "Algerian Dinar"), ("Tunisia", "Tunisian Dinar"),
)

# --- country_continent: 80 sovereign countries, single unambiguous continent.
# Deliberately excludes transcontinental countries (Russia, Turkey, Egypt,
# Kazakhstan, Georgia, Azerbaijan, Armenia, Cyprus) to avoid ambiguity.
# Multi-word country names are kept naturally spaced -- empirically verified
# against the real extract_subject (its capture group is permissive text
# between "for" and "?", not restricted to single tokens).
_CONTINENTS: tuple[tuple[str, str], ...] = (
    ("France", "Europe"), ("Germany", "Europe"), ("Italy", "Europe"),
    ("Spain", "Europe"), ("Portugal", "Europe"), ("Poland", "Europe"),
    ("Sweden", "Europe"), ("Norway", "Europe"), ("Denmark", "Europe"),
    ("Finland", "Europe"), ("Iceland", "Europe"), ("Ireland", "Europe"),
    ("Austria", "Europe"), ("Switzerland", "Europe"), ("Belgium", "Europe"),
    ("Netherlands", "Europe"), ("Greece", "Europe"), ("Hungary", "Europe"),
    ("Romania", "Europe"), ("Ukraine", "Europe"), ("Croatia", "Europe"),
    ("Slovakia", "Europe"), ("Slovenia", "Europe"), ("Estonia", "Europe"),
    ("Latvia", "Europe"), ("Lithuania", "Europe"),
    ("Japan", "Asia"), ("China", "Asia"), ("India", "Asia"),
    ("Thailand", "Asia"), ("Vietnam", "Asia"), ("Indonesia", "Asia"),
    ("Malaysia", "Asia"), ("Philippines", "Asia"), ("Pakistan", "Asia"),
    ("Bangladesh", "Asia"), ("Nepal", "Asia"), ("Mongolia", "Asia"),
    ("Cambodia", "Asia"), ("Laos", "Asia"), ("Myanmar", "Asia"),
    ("Sri Lanka", "Asia"), ("Saudi Arabia", "Asia"), ("Israel", "Asia"),
    ("Jordan", "Asia"), ("Kuwait", "Asia"), ("Qatar", "Asia"),
    ("Iraq", "Asia"), ("Iran", "Asia"), ("Afghanistan", "Asia"),
    ("South Korea", "Asia"), ("Singapore", "Asia"),
    ("Nigeria", "Africa"), ("Kenya", "Africa"), ("Ethiopia", "Africa"),
    ("Ghana", "Africa"), ("Morocco", "Africa"), ("Algeria", "Africa"),
    ("Tunisia", "Africa"), ("Tanzania", "Africa"), ("Uganda", "Africa"),
    ("Zambia", "Africa"), ("Malawi", "Africa"), ("Botswana", "Africa"),
    ("Namibia", "Africa"), ("Senegal", "Africa"), ("Mozambique", "Africa"),
    ("Canada", "North America"), ("Mexico", "North America"),
    ("Guatemala", "North America"), ("Honduras", "North America"),
    ("Nicaragua", "North America"), ("Panama", "North America"),
    ("Costa Rica", "North America"), ("Cuba", "North America"),
    ("Jamaica", "North America"),
    ("Brazil", "South America"), ("Argentina", "South America"),
    ("Chile", "South America"), ("Peru", "South America"),
    ("Colombia", "South America"), ("Bolivia", "South America"),
    ("Paraguay", "South America"), ("Uruguay", "South America"),
    ("Ecuador", "South America"), ("Venezuela", "South America"),
    ("Australia", "Oceania"), ("New Zealand", "Oceania"), ("Fiji", "Oceania"),
)

# --- us_state_capital: all 50 US states, stable/unambiguous, naturally
# spaced multi-word names -- same empirical verification as continents above.
_US_STATE_CAPITALS: tuple[tuple[str, str], ...] = (
    ("Alabama", "Montgomery"), ("Alaska", "Juneau"), ("Arizona", "Phoenix"),
    ("Arkansas", "Little Rock"), ("California", "Sacramento"), ("Colorado", "Denver"),
    ("Connecticut", "Hartford"), ("Delaware", "Dover"), ("Florida", "Tallahassee"),
    ("Georgia", "Atlanta"), ("Hawaii", "Honolulu"), ("Idaho", "Boise"),
    ("Illinois", "Springfield"), ("Indiana", "Indianapolis"), ("Iowa", "Des Moines"),
    ("Kansas", "Topeka"), ("Kentucky", "Frankfort"), ("Louisiana", "Baton Rouge"),
    ("Maine", "Augusta"), ("Maryland", "Annapolis"), ("Massachusetts", "Boston"),
    ("Michigan", "Lansing"), ("Minnesota", "Saint Paul"), ("Mississippi", "Jackson"),
    ("Missouri", "Jefferson City"), ("Montana", "Helena"), ("Nebraska", "Lincoln"),
    ("Nevada", "Carson City"), ("New Hampshire", "Concord"), ("New Jersey", "Trenton"),
    ("New Mexico", "Santa Fe"), ("New York", "Albany"), ("North Carolina", "Raleigh"),
    ("North Dakota", "Bismarck"), ("Ohio", "Columbus"), ("Oklahoma", "Oklahoma City"),
    ("Oregon", "Salem"), ("Pennsylvania", "Harrisburg"), ("Rhode Island", "Providence"),
    ("South Carolina", "Columbia"), ("South Dakota", "Pierre"), ("Tennessee", "Nashville"),
    ("Texas", "Austin"), ("Utah", "Salt Lake City"), ("Vermont", "Montpelier"),
    ("Virginia", "Richmond"), ("Washington", "Olympia"), ("West Virginia", "Charleston"),
    ("Wisconsin", "Madison"), ("Wyoming", "Cheyenne"),
)

# --- planet_order: trivial, 8 planets, position counted from the sun.
_PLANET_ORDER: tuple[tuple[str, str], ...] = (
    ("Mercury", "1"), ("Venus", "2"), ("Earth", "3"), ("Mars", "4"),
    ("Jupiter", "5"), ("Saturn", "6"), ("Uranus", "7"), ("Neptune", "8"),
)


def _build_domain(table: tuple[tuple[str, str], ...], domain: str,
                  id_prefix: str) -> list[ExecTrainingTask]:
    relation = _RELATION_BY_DOMAIN_V2[domain]
    tasks = []
    for i, (subject, answer) in enumerate(table, 1):
        question = f"What is the {relation} for {subject}?"
        tasks.append(ExecTrainingTask(
            task_id=f"exec-v2-gk-{id_prefix}-{i:04d}", domain=domain,
            question=question, answer=answer, subject=subject,
            metadata={"relation": relation}))
    return tasks


def build_answer_now_tasks_v2() -> list[ExecTrainingTask]:
    """Build the full V2 ANSWER_NOW-viable family: the 5 new categories only
    (326 tasks). Does NOT include V1's 72 capitals/element_symbols tasks --
    those remain in the consumed exec_training_v1 split and are not
    resampled into V2 per configs/gate_answer_probe_v2_design.json."""
    tasks: list[ExecTrainingTask] = []
    tasks += _build_domain(_ATOMIC_NUMBERS, "atomic_number", "atomic")
    tasks += _build_domain(_CURRENCIES, "currency", "currency")
    tasks += _build_domain(_CONTINENTS, "continent", "continent")
    tasks += _build_domain(_US_STATE_CAPITALS, "state_capital", "statecap")
    tasks += _build_domain(_PLANET_ORDER, "planet_order", "planet")
    return tasks
