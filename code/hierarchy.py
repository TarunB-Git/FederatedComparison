#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

KNOWN_RACES = ("Prot", "Terr", "Zerg")
SHARED_RULE_KEY = "_shared"

LEGACY_FAMILIES = [
    "Build",
    "Train",
    "Tech",
    "Expand",
    "Attack",
    "Scout",
    "Defend",
    "Warp-in",
    "Other",
]

LEGACY_PROTOSS_LINK_FAMILY_HINTS = {
    41: "Build",
    43: "Build",
    46: "Build",
    122: "Expand",
    172: "Train",
    173: "Train",
    174: "Tech",
    175: "Tech",
    176: "Tech",
    177: "Train",
    216: "Attack",
    220: "Attack",
    235: "Scout",
    279: "Defend",
    421: "Defend",
    528: "Tech",
    540: "Defend",
    610: "Scout",
    715: "Warp-in",
    722: "Warp-in",
}

# Expanded Protoss ability-to-family mapping.  Reduces the massive "Other"
# bucket (~275 fine actions, 14 % of data) by classifying the most frequent
# unmapped SC2 Protoss ability IDs into the existing legacy8 families.
LEGACY_PROTOSS_EXPANDED_LINK_FAMILY_HINTS = {
    # ── Economy / Build ──────────────────────────────────────────────────
    40: "Build",
    42: "Build",
    45: "Build",       # very high frequency (~28 % of Other)
    61: "Build",
    62: "Build",
    64: "Build",
    120: "Build",
    123: "Build",
    125: "Build",
    127: "Build",
    129: "Build",
    133: "Build",
    140: "Build",
    # ── Production / Train ───────────────────────────────────────────────
    170: "Train",      # high frequency (~10 % of Other)
    171: "Train",
    178: "Train",
    179: "Train",
    180: "Train",
    181: "Train",
    183: "Train",
    184: "Train",
    # ── Technology / Research ────────────────────────────────────────────
    69: "Tech",
    71: "Tech",
    74: "Tech",
    75: "Tech",
    76: "Tech",
    77: "Tech",
    78: "Tech",
    79: "Tech",
    80: "Tech",
    81: "Tech",
    82: "Tech",
    83: "Tech",
    84: "Tech",
    85: "Tech",
    86: "Tech",
    87: "Tech",
    88: "Tech",
    89: "Tech",
    91: "Tech",
    93: "Tech",
    102: "Tech",
    104: "Tech",
    105: "Tech",
    107: "Tech",
    # ── Attack / Tactical Combat ─────────────────────────────────────────
    182: "Attack",
    214: "Attack",     # high frequency (~5.8 % of Other)
    218: "Attack",
    228: "Attack",
    233: "Attack",
    234: "Attack",
    236: "Attack",
    238: "Attack",
    239: "Attack",
    269: "Attack",
    # ── Scout / Information ──────────────────────────────────────────────
    605: "Scout",
    608: "Scout",
    635: "Scout",
    636: "Scout",
    698: "Scout",
    706: "Scout",
    714: "Scout",
    717: "Scout",
    718: "Scout",
    723: "Scout",
}

BROAD_FAMILIES = [
    "Economy",
    "Production",
    "Technology",
    "Tactical Combat",
    "Information",
    "Other",
]

BROAD_FAMILIES_WITH_WARP_IN = [
    "Economy",
    "Production",
    "Technology",
    "Tactical Combat",
    "Information",
    "Warp-in",
    "Other",
]

BROAD_SHARED_LINK_FAMILY_HINTS = {
    41: "Economy",
    43: "Economy",
    46: "Economy",
    122: "Economy",
}

BROAD_PROTOSS_LINK_FAMILY_HINTS = {
    172: "Production",
    173: "Production",
    174: "Technology",
    175: "Technology",
    176: "Technology",
    177: "Production",
    216: "Tactical Combat",
    220: "Tactical Combat",
    235: "Information",
    279: "Tactical Combat",
    421: "Tactical Combat",
    528: "Technology",
    540: "Tactical Combat",
    610: "Information",
    715: "Production",
    722: "Production",
}

BROAD_PROTOSS_LINK_FAMILY_HINTS_WITH_WARP_IN = {
    **BROAD_PROTOSS_LINK_FAMILY_HINTS,
    715: "Warp-in",
    722: "Warp-in",
}

# ── Balanced-6 taxonomy ──────────────────────────────────────────────────────
# Deterministic merge of legacy families into 6 balanced groups.
# No catch-all "Other" family — every action must map to an explicit group.

BALANCED6_FAMILIES = [
    "EconomyExpand",
    "ProductionUnits",
    "ProductionWarpSpell",
    "TechnologyUpgrades",
    "CombatDefense",
    "InformationUtility",
]

BALANCED6_PROTOSS_LINK_FAMILY_HINTS = {
    # Build + Expand → EconomyExpand
    41: "EconomyExpand",
    43: "EconomyExpand",
    46: "EconomyExpand",
    122: "EconomyExpand",
    # Train → ProductionUnits
    172: "ProductionUnits",
    173: "ProductionUnits",
    177: "ProductionUnits",
    # Warp-in → ProductionWarpSpell
    715: "ProductionWarpSpell",
    722: "ProductionWarpSpell",
    # Tech → TechnologyUpgrades
    174: "TechnologyUpgrades",
    175: "TechnologyUpgrades",
    176: "TechnologyUpgrades",
    528: "TechnologyUpgrades",
    # Attack + Defend → CombatDefense
    216: "CombatDefense",
    220: "CombatDefense",
    279: "CombatDefense",
    421: "CombatDefense",
    540: "CombatDefense",
    # Scout → InformationUtility
    235: "InformationUtility",
    610: "InformationUtility",
}

# Default fallback family for balanced6 (no "Other" catch-all).
BALANCED6_FALLBACK_FAMILY = "InformationUtility"

# ── Macro-Tactical-3 taxonomy ────────────────────────────────────────────────
# 3-class simplification to eliminate structural noise between overlapping macro actions.
# Categories: Macro (Build/Expand/Train/Warp/Tech), Tactical (Attack/Defend), Information (Scout/Other)

MACRO_TACTICAL3_FAMILIES = [
    "Macro",
    "Tactical",
    "Information",
]

MACRO_TACTICAL3_PROTOSS_LINK_FAMILY_HINTS = {
    # Macro
    41: "Macro",
    43: "Macro",
    46: "Macro",
    122: "Macro",
    172: "Macro",
    173: "Macro",
    174: "Macro",
    175: "Macro",
    176: "Macro",
    177: "Macro",
    528: "Macro",
    715: "Macro",
    722: "Macro",
    # Tactical
    216: "Tactical",
    220: "Tactical",
    279: "Tactical",
    421: "Tactical",
    540: "Tactical",
    # Information
    235: "Information",
    610: "Information",
}

MACRO_TACTICAL3_FALLBACK_FAMILY = "Information"

# ── ProdEcon-4 taxonomy ──────────────────────────────────────────────────────
# 4-class merge: Production+Economy are merged into "ProdEcon" because they are
# frequently confused (Production=42% accuracy vs Economy=82% in broad5).
# Classes: ProdEcon, Technology, TacticalCombat, Information

PRODECON4_FAMILIES = [
    "ProdEcon",
    "Technology",
    "TacticalCombat",
    "Information",
]

PRODECON4_PROTOSS_LINK_FAMILY_HINTS = {
    # ProdEcon: Economy + Production actions
    41: "ProdEcon",   # Economy
    43: "ProdEcon",   # Economy
    46: "ProdEcon",   # Economy
    122: "ProdEcon",  # Economy
    172: "ProdEcon",  # Production
    173: "ProdEcon",  # Production
    177: "ProdEcon",  # Production
    715: "ProdEcon",  # Production
    722: "ProdEcon",  # Production
    # Technology
    174: "Technology",
    175: "Technology",
    176: "Technology",
    528: "Technology",
    # TacticalCombat
    216: "TacticalCombat",
    220: "TacticalCombat",
    279: "TacticalCombat",
    421: "TacticalCombat",
    540: "TacticalCombat",
    # Information
    235: "Information",
    610: "Information",
}

PRODECON4_FALLBACK_FAMILY = "Information"

# These defaults focus on broad coverage of the checked-in Terran action space so
# the hierarchy no longer collapses almost everything to Other. Users can extend
# or override them via a race-keyed JSON rules file.
BROAD_TERRAN_LINK_FAMILY_HINTS = {
    92: "Information",
    121: "Technology",
    125: "Technology",
    126: "Technology",
    128: "Technology",
    129: "Technology",
    130: "Technology",
    131: "Technology",
    133: "Economy",
    134: "Economy",
    137: "Economy",
    138: "Economy",
    139: "Economy",
    140: "Economy",
    141: "Economy",
    142: "Economy",
    143: "Economy",
    144: "Economy",
    145: "Tactical Combat",
    146: "Tactical Combat",
    147: "Economy",
    148: "Economy",
    149: "Economy",
    150: "Economy",
    151: "Technology",
    152: "Technology",
    153: "Technology",
    154: "Technology",
    155: "Technology",
    156: "Technology",
    157: "Production",
    158: "Production",
    159: "Production",
    160: "Technology",
    161: "Production",
    162: "Production",
    163: "Production",
    164: "Production",
    167: "Production",
    168: "Tactical Combat",
    169: "Tactical Combat",
    170: "Tactical Combat",
    171: "Tactical Combat",
    224: "Tactical Combat",
    229: "Tactical Combat",
    232: "Technology",
    233: "Technology",
    243: "Information",
    257: "Tactical Combat",
    399: "Technology",
    400: "Technology",
    406: "Information",
    530: "Information",
    532: "Information",
    534: "Technology",
    536: "Technology",
    617: "Information",
    618: "Information",
    693: "Tactical Combat",
    695: "Information",
    697: "Information",
    709: "Tactical Combat",
    718: "Information",
    721: "Information",
}

BROAD_ZERG_LINK_FAMILY_HINTS = {
    70: "Information",
    72: "Information",
    97: "Information",
    108: "Economy",
    109: "Economy",
    112: "Technology",
    113: "Production",
    119: "Tactical Combat",
    124: "Economy",
    125: "Economy",
    126: "Economy",
    185: "Technology",
    186: "Technology",
    187: "Production",
    188: "Production",
    189: "Production",
    190: "Production",
    191: "Production",
    192: "Tactical Combat",
    193: "Tactical Combat",
    194: "Tactical Combat",
    195: "Economy",
    196: "Tactical Combat",
    197: "Tactical Combat",
    198: "Tactical Combat",
    199: "Technology",
    200: "Technology",
    201: "Tactical Combat",
    203: "Economy",
    204: "Economy",
    205: "Economy",
    206: "Economy",
    213: "Tactical Combat",
    217: "Tactical Combat",
    218: "Tactical Combat",
    219: "Tactical Combat",
    221: "Tactical Combat",
    222: "Tactical Combat",
    223: "Tactical Combat",
    225: "Tactical Combat",
    226: "Tactical Combat",
    245: "Production",
    247: "Production",
    261: "Production",
    262: "Production",
    263: "Production",
    264: "Technology",
    265: "Technology",
    266: "Production",
    267: "Production",
    270: "Production",
    276: "Economy",
    283: "Technology",
    305: "Tactical Combat",
    306: "Tactical Combat",
    309: "Technology",
    310: "Technology",
    311: "Technology",
    312: "Technology",
    383: "Information",
    385: "Technology",
    388: "Technology",
    521: "Information",
    524: "Information",
    539: "Tactical Combat",
    546: "Technology",
    609: "Information",
    613: "Information",
    614: "Information",
    700: "Information",
    702: "Information",
    713: "Information",
    716: "Information",
    717: "Information",
    729: "Information",
}


_BROAD_TO_LEGACY_FAMILY = {
    "Economy": "Build",
    "Production": "Train",
    "Technology": "Tech",
    "Tactical Combat": "Attack",
    "Information": "Scout",
    "Warp-in": "Warp-in",
    "Other": "Other",
}


def _broad_rules_as_legacy(rules: dict[int, str]) -> dict[int, str]:
    return {
        int(link): _BROAD_TO_LEGACY_FAMILY.get(str(family), "Other")
        for link, family in rules.items()
    }


def action_link_from_key(action_key: str) -> int:
    txt = str(action_key)
    if ":" in txt:
        txt = txt.split(":", 1)[0]
    try:
        return int(txt)
    except Exception:
        return -1


def normalize_race_key(value: str | None) -> str:
    txt = str(value or "").strip()
    if txt in KNOWN_RACES:
        return txt
    if txt.lower() in {"protoss", "prot"}:
        return "Prot"
    if txt.lower() in {"terran", "terr"}:
        return "Terr"
    if txt.lower() == "zerg":
        return "Zerg"
    if txt.lower() == "all":
        return "all"
    if txt == SHARED_RULE_KEY:
        return SHARED_RULE_KEY
    return txt


def taxonomy_families(taxonomy: str, separate_warp_in: bool = False) -> list[str]:
    mode = str(taxonomy).strip().lower()
    if mode in {"legacy", "legacy8"}:
        return list(LEGACY_FAMILIES)
    if mode in {"broad", "broad5"}:
        return list(BROAD_FAMILIES)
    if mode == "broad6_warp":
        return list(BROAD_FAMILIES_WITH_WARP_IN)
    if mode == "balanced6":
        return list(BALANCED6_FAMILIES)
    if mode == "macro_tactical3":
        return list(MACRO_TACTICAL3_FAMILIES)
    if mode == "prodecon4":
        return list(PRODECON4_FAMILIES)
    raise ValueError(f"Unsupported hierarchy taxonomy: {taxonomy}")


def default_link_family_rules(taxonomy: str, separate_warp_in: bool = False) -> dict[str, dict[int, str]]:
    mode = str(taxonomy).strip().lower()
    if mode in {"legacy", "legacy8"}:
        shared_rules = {
            41: "Build",
            43: "Build",
            46: "Build",
            122: "Expand",
        }
        # Merge the base legacy hints with the expanded Protoss coverage.
        prot_rules = dict(LEGACY_PROTOSS_LINK_FAMILY_HINTS)
        prot_rules.update(LEGACY_PROTOSS_EXPANDED_LINK_FAMILY_HINTS)
        return {
            SHARED_RULE_KEY: shared_rules,
            "Prot": prot_rules,
            "Terr": _broad_rules_as_legacy(BROAD_TERRAN_LINK_FAMILY_HINTS),
            "Zerg": _broad_rules_as_legacy(BROAD_ZERG_LINK_FAMILY_HINTS),
        }

    if mode == "balanced6":
        return {"Prot": dict(BALANCED6_PROTOSS_LINK_FAMILY_HINTS)}

    if mode == "macro_tactical3":
        return {"Prot": dict(MACRO_TACTICAL3_PROTOSS_LINK_FAMILY_HINTS)}

    if mode == "prodecon4":
        return {"Prot": dict(PRODECON4_PROTOSS_LINK_FAMILY_HINTS)}

    if mode not in {"broad", "broad5", "broad6_warp"}:
        raise ValueError(f"Unsupported hierarchy taxonomy: {taxonomy}")

    use_warp = separate_warp_in or mode == "broad6_warp"
    return {
        SHARED_RULE_KEY: dict(BROAD_SHARED_LINK_FAMILY_HINTS),
        "Prot": dict(BROAD_PROTOSS_LINK_FAMILY_HINTS_WITH_WARP_IN if use_warp else BROAD_PROTOSS_LINK_FAMILY_HINTS),
        "Terr": dict(BROAD_TERRAN_LINK_FAMILY_HINTS),
        "Zerg": dict(BROAD_ZERG_LINK_FAMILY_HINTS),
    }


def _copy_rules(rules: dict[str, dict[int, str]]) -> dict[str, dict[int, str]]:
    return {str(race): {int(k): str(v) for k, v in race_rules.items()} for race, race_rules in rules.items()}


def _merge_rule_map(
    base: dict[str, dict[int, str]],
    incoming: dict,
    *,
    allowed: set[str],
) -> None:
    for raw_race, race_rules in incoming.items():
        race_key = normalize_race_key(raw_race)
        if race_key == "all":
            race_key = SHARED_RULE_KEY
        if race_key not in KNOWN_RACES and race_key != SHARED_RULE_KEY:
            continue
        if not isinstance(race_rules, dict):
            continue
        base.setdefault(race_key, {})
        for k, v in race_rules.items():
            try:
                link = int(k)
            except Exception:
                continue
            fam = str(v)
            if fam not in allowed:
                continue
            base[race_key][link] = fam


def load_hierarchy_rules(
    path: str,
    *,
    taxonomy: str,
    separate_warp_in: bool,
    family_order: list[str],
) -> dict[str, dict[int, str]]:
    rules = _copy_rules(default_link_family_rules(taxonomy, separate_warp_in))
    if not path:
        return rules

    p = Path(path)
    if not p.exists():
        print(f"[warn] hierarchy rules not found at {p}; using defaults", flush=True)
        return rules

    try:
        obj = json.loads(p.read_text())
    except Exception as exc:
        print(f"[warn] failed to parse hierarchy rules {p}: {exc}; using defaults", flush=True)
        return rules

    if not isinstance(obj, dict):
        print(f"[warn] hierarchy rules in {p} are not a JSON object; using defaults", flush=True)
        return rules

    allowed = set(family_order)
    if obj and all(str(k).lstrip("-").isdigit() for k in obj.keys()):
        obj = {SHARED_RULE_KEY: obj}

    _merge_rule_map(rules, obj, allowed=allowed)
    return rules


def assign_coarse_family(
    action_key: str,
    *,
    player_race: str | None,
    link_family_rules: dict[str, dict[int, str]],
    family_order: list[str],
    fallback_family: str = "Other",
) -> str:
    link = action_link_from_key(action_key)
    race_key = normalize_race_key(player_race)

    for key in [race_key, SHARED_RULE_KEY]:
        fam = link_family_rules.get(key, {}).get(link)
        if fam in family_order:
            return str(fam)
    return fallback_family


def _taxonomy_fallback_family(taxonomy: str) -> str:
    """Return the default unmapped-action family for a taxonomy."""
    mode = str(taxonomy).strip().lower()
    if mode == "balanced6":
        return BALANCED6_FALLBACK_FAMILY
    if mode == "macro_tactical3":
        return MACRO_TACTICAL3_FALLBACK_FAMILY
    if mode == "prodecon4":
        return PRODECON4_FALLBACK_FAMILY
    return "Other"


def _family_key_order(
    family_df: pd.DataFrame,
    *,
    train_counts: dict[str, int],
) -> list[str]:
    keys = [str(x) for x in family_df["action_key"].astype(str).unique().tolist()]

    def sort_key(k: str) -> tuple[int, str]:
        return (-int(train_counts.get(k, 0)), k)

    return sorted(keys, key=sort_key)


def _build_flat_action_family_map(action_family_df: pd.DataFrame) -> dict[str, str]:
    out: dict[str, str] = {}
    for action_key, g in action_family_df.groupby("action_key"):
        families = sorted(set(g["coarse_family"].astype(str).tolist()))
        if len(families) == 1:
            out[str(action_key)] = families[0]
    return out


def _build_flat_link_rules(link_family_rules: dict[str, dict[int, str]]) -> dict[str, str]:
    per_link: dict[int, set[str]] = {}
    for rules in link_family_rules.values():
        for link, fam in rules.items():
            per_link.setdefault(int(link), set()).add(str(fam))

    flat = {}
    for link, fams in per_link.items():
        if len(fams) == 1:
            flat[str(link)] = next(iter(fams))
    return flat


def _build_action_family_by_race(action_family_df: pd.DataFrame) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for race, g in action_family_df.groupby("player_race"):
        out[str(race)] = {
            str(action_key): str(coarse_family)
            for action_key, coarse_family in g[["action_key", "coarse_family"]].itertuples(index=False, name=None)
        }
    return out


def build_hierarchy_labels(
    df_events: pd.DataFrame,
    *,
    family_top_k: int,
    family_other_token: str,
    link_family_rules: dict[str, dict[int, str]],
    family_order: list[str],
    keep_all_fine_actions: bool,
    taxonomy: str,
    separate_warp_in: bool,
) -> tuple[pd.DataFrame, dict]:
    out = df_events.copy()

    if "player_race" in out.columns:
        race_values = out["player_race"].astype(str).map(normalize_race_key)
    else:
        race_values = pd.Series(["all"] * len(out), index=out.index, dtype="object")

    fallback = _taxonomy_fallback_family(taxonomy)
    coarse_vals = [
        assign_coarse_family(
            action_key,
            player_race=player_race,
            link_family_rules=link_family_rules,
            family_order=family_order,
            fallback_family=fallback,
        )
        for action_key, player_race in zip(out["action_key"].astype(str).tolist(), race_values.tolist())
    ]
    out["coarse_family"] = coarse_vals

    family_to_id = {fam: idx for idx, fam in enumerate(family_order)}
    out["coarse_family_id"] = out["coarse_family"].map(family_to_id).astype("int64")

    train_mask = out["split"] == "train"
    train_counts_df = out.loc[train_mask, ["coarse_family", "action_key"]].value_counts().rename("count").reset_index()
    train_action_counts = out.loc[train_mask, "action_key"].astype(str).value_counts().to_dict()

    family_top_actions: dict[str, list[str]] = {}
    family_fine_to_id: dict[str, dict[str, int]] = {}
    family_other_exact_action_id: dict[str, int] = {}

    for fam in family_order:
        fam_rows_all = out[out["coarse_family"] == fam]
        ordered_keys = _family_key_order(fam_rows_all, train_counts=train_action_counts)

        if keep_all_fine_actions or family_top_k <= 0:
            top_actions = list(ordered_keys)
            fine_keys = list(ordered_keys)
        else:
            fam_rows_train = train_counts_df[train_counts_df["coarse_family"] == fam]
            top_actions = fam_rows_train.head(max(1, int(family_top_k)))["action_key"].astype(str).tolist()
            fine_keys = list(top_actions)
            if family_other_token not in fine_keys:
                fine_keys.append(family_other_token)

        family_top_actions[fam] = list(top_actions)
        family_fine_to_id[fam] = {k: i for i, k in enumerate(fine_keys)}

        if family_other_token in family_fine_to_id[fam] and len(top_actions) > 0:
            top_exact = out.loc[(train_mask) & (out["action_key"].astype(str) == top_actions[0]), "action_id"]
            family_other_exact_action_id[fam] = int(top_exact.iloc[0]) if len(top_exact) else -1
        else:
            family_other_exact_action_id[fam] = -1

    top_set_by_family = {fam: set(v) for fam, v in family_top_actions.items()}

    def _fine_key(row: pd.Series) -> str:
        fam = str(row["coarse_family"])
        ak = str(row["action_key"])
        if keep_all_fine_actions or family_top_k <= 0:
            return ak
        if ak in top_set_by_family.get(fam, set()):
            return ak
        return family_other_token

    out["fine_action_key"] = out.apply(_fine_key, axis=1)
    out["fine_action_id"] = out.apply(
        lambda r: family_fine_to_id[str(r["coarse_family"])][str(r["fine_action_key"])],
        axis=1,
    ).astype("int64")
    out["fine_is_other"] = out["fine_action_key"].astype(str).eq(family_other_token)
    out["exact_action_id"] = out["action_id"].astype("int64")

    family_counts = out["coarse_family"].value_counts().to_dict()
    fine_support = out.groupby(["coarse_family", "fine_action_key"]).size().rename("count").reset_index().to_dict("records")

    action_family_df = out[["action_key", "coarse_family"]].drop_duplicates().copy()
    if "player_race" in out.columns:
        action_family_by_race_df = out[["player_race", "action_key", "coarse_family"]].drop_duplicates().copy()
        action_family_by_race_df["player_race"] = action_family_by_race_df["player_race"].astype(str).map(normalize_race_key)
    else:
        action_family_by_race_df = pd.DataFrame(
            {
                "player_race": ["all"] * len(action_family_df),
                "action_key": action_family_df["action_key"].tolist(),
                "coarse_family": action_family_df["coarse_family"].tolist(),
            }
        )

    flat_action_to_family = _build_flat_action_family_map(action_family_df)
    action_to_family_by_race = _build_action_family_by_race(action_family_by_race_df)

    family_counts_by_race = {}
    if "player_race" in out.columns:
        for race, g in out.groupby(out["player_race"].astype(str).map(normalize_race_key)):
            family_counts_by_race[str(race)] = {str(k): int(v) for k, v in g["coarse_family"].value_counts().to_dict().items()}

    unmapped_links_by_race = {}
    if "player_race" in out.columns:
        out_tmp = out.copy()
        out_tmp["player_race_norm"] = out_tmp["player_race"].astype(str).map(normalize_race_key)
        out_tmp["action_link_resolved"] = out_tmp["action_key"].astype(str).map(action_link_from_key)
        for race, g in out_tmp.groupby("player_race_norm"):
            explicit = set(link_family_rules.get(str(race), {}).keys()).union(link_family_rules.get(SHARED_RULE_KEY, {}).keys())
            candidates = g[g["coarse_family"].astype(str) == "Other"]["action_link_resolved"].dropna().astype(int).unique().tolist()
            unmapped_links = sorted([int(x) for x in candidates if int(x) not in explicit])
            unmapped_links_by_race[str(race)] = unmapped_links

    mapping = {
        "hierarchy_taxonomy": str(taxonomy),
        "separate_warp_in": bool(separate_warp_in),
        "keep_all_fine_actions": bool(keep_all_fine_actions or family_top_k <= 0),
        "family_order": family_order,
        "family_to_id": family_to_id,
        "id_to_family": {str(v): k for k, v in family_to_id.items()},
        "action_to_family": flat_action_to_family,
        "action_to_family_by_race": action_to_family_by_race,
        "family_top_actions": family_top_actions,
        "family_fine_to_id": family_fine_to_id,
        "family_id_to_fine": {fam: {str(v): k for k, v in fine_map.items()} for fam, fine_map in family_fine_to_id.items()},
        "family_other_token": family_other_token,
        "family_other_exact_action_id": family_other_exact_action_id,
        "family_counts": {k: int(v) for k, v in family_counts.items()},
        "family_counts_by_race": family_counts_by_race,
        "fine_support": fine_support,
        "link_family_rules": _build_flat_link_rules(link_family_rules),
        "link_family_rules_by_race": {
            str(race): {str(k): v for k, v in race_rules.items()}
            for race, race_rules in link_family_rules.items()
        },
        "unmapped_links_by_race": {str(race): [int(x) for x in links] for race, links in unmapped_links_by_race.items()},
    }
    return out, mapping


def _safe_read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _safe_read_events(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return None


def _resolve_events_artifact(path: Path) -> Path | None:
    candidates = [path / "processed_events.parquet", path / "processed_events.csv"]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _resolve_hierarchy_mapping(action_vocab: dict, hierarchy_mapping_path: Path | None) -> dict:
    if hierarchy_mapping_path is not None and hierarchy_mapping_path.exists():
        return _safe_read_json(hierarchy_mapping_path)
    hierarchy = action_vocab.get("hierarchy")
    if isinstance(hierarchy, dict):
        return hierarchy
    return {}


def audit_hierarchy_mapping(
    *,
    action_vocab_path: str | Path,
    hierarchy_mapping_path: str | Path | None = None,
    outdir: str | Path | None = None,
    report_prefix: str = "sc2egset_hierarchy",
    events_path: str | Path | None = None,
) -> dict:
    action_vocab_path = Path(action_vocab_path)
    hierarchy_mapping_p = Path(hierarchy_mapping_path) if hierarchy_mapping_path else None

    vocab = _safe_read_json(action_vocab_path)
    mapping = _resolve_hierarchy_mapping(vocab, hierarchy_mapping_p)

    action_to_id = {str(k): int(v) for k, v in (vocab.get("action_to_id") or {}).items()}
    action_counts = {str(k): int(v) for k, v in (vocab.get("action_counts") or {}).items()}
    family_order = list(mapping.get("family_order") or [])
    family_top_actions = {str(k): [str(x) for x in v] for k, v in (mapping.get("family_top_actions") or {}).items()}
    other_token = str(mapping.get("family_other_token", "__OTHER__"))

    action_to_family_flat = {str(k): str(v) for k, v in (mapping.get("action_to_family") or {}).items()}
    action_to_family_by_race = {
        normalize_race_key(race): {str(k): str(v) for k, v in race_map.items()}
        for race, race_map in (mapping.get("action_to_family_by_race") or {}).items()
        if isinstance(race_map, dict)
    }
    link_rules_by_race = {
        normalize_race_key(race): {str(k): str(v) for k, v in race_map.items()}
        for race, race_map in (mapping.get("link_family_rules_by_race") or {}).items()
        if isinstance(race_map, dict)
    }
    link_rules_flat = {str(k): str(v) for k, v in (mapping.get("link_family_rules") or {}).items()}

    resolved_events_path = None
    if events_path is not None:
        resolved_events_path = Path(events_path)
    else:
        resolved_events_path = _resolve_events_artifact(action_vocab_path.parent)
    events_df = _safe_read_events(resolved_events_path) if resolved_events_path is not None else None

    rows = []
    if events_df is not None and {"action_key", "split"}.issubset(events_df.columns):
        race_series = (
            events_df["player_race"].astype(str).map(normalize_race_key)
            if "player_race" in events_df.columns
            else pd.Series(["all"] * len(events_df), index=events_df.index, dtype="object")
        )
        work = pd.DataFrame(
            {
                "player_race": race_series,
                "action_key": events_df["action_key"].astype(str),
                "split": events_df["split"].astype(str),
            }
        )
        train_counts_by_race_action = (
            work[work["split"] == "train"]
            .groupby(["player_race", "action_key"])
            .size()
            .rename("train_count")
            .reset_index()
        )

        for race, action_key, train_count in train_counts_by_race_action.itertuples(index=False, name=None):
            coarse = action_to_family_by_race.get(str(race), {}).get(
                str(action_key),
                action_to_family_flat.get(str(action_key), "Other"),
            )
            top_actions = family_top_actions.get(coarse, [])

            if str(action_key) in top_actions:
                fine_label = str(action_key)
                collapsed = False
                fine_rule = f"kept_in_family_top_k_rank_{top_actions.index(str(action_key)) + 1}_of_{len(top_actions)}"
            else:
                fine_label = other_token if top_actions else str(action_key)
                collapsed = fine_label == other_token
                fine_rule = "collapsed_to_family_OTHER" if collapsed else "family_without_topk_exact_kept"

            link = str(action_link_from_key(action_key))
            race_rules = link_rules_by_race.get(str(race), {})
            coarse_rule = "fallback_to_other"
            if link in race_rules or link in link_rules_by_race.get(SHARED_RULE_KEY, {}):
                coarse_rule = "explicit_link_rule"
            elif link in link_rules_flat:
                coarse_rule = "explicit_link_rule"

            rows.append(
                {
                    "player_race": str(race),
                    "original_action": str(action_key),
                    "coarse_family": str(coarse),
                    "fine_label": str(fine_label),
                    "mapping_rule": f"coarse={coarse_rule}; fine={fine_rule}",
                    "collapsed_to_other": bool(collapsed),
                    "train_count": int(train_count),
                }
            )
    else:
        for action_key, _action_id in sorted(action_to_id.items(), key=lambda kv: kv[1]):
            coarse = action_to_family_flat.get(str(action_key), "Other")
            top_actions = family_top_actions.get(coarse, [])

            if str(action_key) in top_actions:
                fine_label = str(action_key)
                collapsed = False
                fine_rule = f"kept_in_family_top_k_rank_{top_actions.index(str(action_key)) + 1}_of_{len(top_actions)}"
            else:
                fine_label = other_token if top_actions else str(action_key)
                collapsed = fine_label == other_token
                fine_rule = "collapsed_to_family_OTHER" if collapsed else "family_without_topk_exact_kept"

            coarse_rule = "fallback_to_other"
            link = str(action_link_from_key(action_key))
            if link in link_rules_flat:
                coarse_rule = "explicit_link_rule"

            rows.append(
                {
                    "player_race": "all",
                    "original_action": str(action_key),
                    "coarse_family": str(coarse),
                    "fine_label": str(fine_label),
                    "mapping_rule": f"coarse={coarse_rule}; fine={fine_rule}",
                    "collapsed_to_other": bool(collapsed),
                    "train_count": int(action_counts.get(str(action_key), 0)),
                }
            )

    table_df = pd.DataFrame(rows)
    if table_df.empty:
        table_df = pd.DataFrame(
            columns=[
                "player_race",
                "original_action",
                "coarse_family",
                "fine_label",
                "mapping_rule",
                "collapsed_to_other",
                "train_count",
            ]
        )

    count_df = (
        table_df.groupby(["player_race", "coarse_family"])
        .agg(
            n_original_actions=("original_action", "nunique"),
            train_event_count=("train_count", "sum"),
        )
        .reset_index()
    )

    unmapped_rows = []
    if "player_race" in table_df.columns and not table_df.empty:
        for race, g in table_df.groupby("player_race"):
            links = sorted(
                {
                    int(action_link_from_key(action_key))
                    for action_key, coarse in g[["original_action", "coarse_family"]].itertuples(index=False, name=None)
                    if str(coarse) == "Other"
                }
            )
            explicit = set()
            for key in [str(race), SHARED_RULE_KEY]:
                explicit.update(int(k) for k in link_rules_by_race.get(key, {}).keys() if str(k).lstrip("-").isdigit())
            missing = [int(x) for x in links if int(x) not in explicit]
            for link in missing:
                unmapped_rows.append({"player_race": str(race), "action_link": int(link)})
    unmapped_df = pd.DataFrame(unmapped_rows)

    collapsed_to_fine_other = int(table_df["collapsed_to_other"].sum()) if not table_df.empty else 0
    mapped_to_coarse_other = int((table_df["coarse_family"] == "Other").sum()) if not table_df.empty else 0

    one_to_one = {
        "unique_action_count_equals_rows": bool(
            table_df[["player_race", "original_action"]].drop_duplicates().shape[0] == len(table_df)
        ),
        "one_coarse_each": bool(
            (table_df.groupby(["player_race", "original_action"])["coarse_family"].nunique() == 1).all()
        )
        if not table_df.empty
        else True,
        "one_fine_each": bool(
            (table_df.groupby(["player_race", "original_action"])["fine_label"].nunique() == 1).all()
        )
        if not table_df.empty
        else True,
    }

    verdict = "mostly hierarchical but partially heuristic"
    if collapsed_to_fine_other > 0 and mapped_to_coarse_other > 0:
        verdict = "mostly loose grouping"
    if not unmapped_df.empty and mapped_to_coarse_other < max(1, len(table_df) // 4):
        verdict = "usable broad race-aware grouping with explicit gaps"

    per_race_family_counts = {}
    if not count_df.empty:
        for race, g in count_df.groupby("player_race"):
            per_race_family_counts[str(race)] = {
                str(row["coarse_family"]): int(row["train_event_count"])
                for _, row in g.iterrows()
            }

    summary = {
        "n_original_actions": int(len(table_df)),
        "family_order": family_order,
        "coarse_action_counts": {
            str(r["coarse_family"]): int(r["n_original_actions"])
            for _, r in count_df.groupby("coarse_family", as_index=False)["n_original_actions"].sum().iterrows()
        },
        "coarse_event_counts": {
            str(r["coarse_family"]): int(r["train_event_count"])
            for _, r in count_df.groupby("coarse_family", as_index=False)["train_event_count"].sum().iterrows()
        },
        "coarse_event_counts_by_race": per_race_family_counts,
        "collapsed_to_fine_other": collapsed_to_fine_other,
        "mapped_to_coarse_other": mapped_to_coarse_other,
        "explicit_link_rule_count": int(sum("explicit_link_rule" in str(x) for x in table_df["mapping_rule"].tolist())),
        "fallback_other_rule_count": int(sum("fallback_to_other" in str(x) for x in table_df["mapping_rule"].tolist())),
        "unmapped_links_by_race": {
            str(race): sorted(g["action_link"].astype(int).tolist())
            for race, g in unmapped_df.groupby("player_race")
        }
        if not unmapped_df.empty
        else {},
        "one_to_one": one_to_one,
        "verdict": verdict,
    }

    if outdir is not None:
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)

        table_path = outdir / f"{report_prefix}_mapping_table.csv"
        counts_path = outdir / "coarse_family_action_counts.csv"
        counts_by_race_path = outdir / "coarse_family_action_counts_by_race.csv"
        unmapped_path = outdir / "unmapped_links_by_race.csv"
        compact_path = outdir / "compact_counts_summary.json"
        summary_path = outdir / f"{report_prefix}_summary.md"

        table_df[
            [
                "player_race",
                "original_action",
                "coarse_family",
                "fine_label",
                "mapping_rule",
                "collapsed_to_other",
                "train_count",
            ]
        ].to_csv(table_path, index=False)

        count_df.groupby("coarse_family", as_index=False)[["n_original_actions", "train_event_count"]].sum().to_csv(
            counts_path,
            index=False,
        )
        count_df.to_csv(counts_by_race_path, index=False)
        unmapped_df.to_csv(unmapped_path, index=False)
        compact_path.write_text(json.dumps(summary, indent=2))

        md_lines = [
            "# SC2EGSet Hierarchy Audit",
            "",
            "## Per-action mapping",
            table_df.to_markdown(index=False),
            "",
            "## Family counts by race",
            count_df.to_markdown(index=False),
            "",
            "## Unmapped links by race",
            (unmapped_df.to_markdown(index=False) if not unmapped_df.empty else "None"),
            "",
            "## Summary",
            f"- n_original_actions: {summary['n_original_actions']}",
            f"- collapsed_to_fine_other: {summary['collapsed_to_fine_other']}",
            f"- mapped_to_coarse_other: {summary['mapped_to_coarse_other']}",
            f"- explicit_link_rule_count: {summary['explicit_link_rule_count']}",
            f"- fallback_other_rule_count: {summary['fallback_other_rule_count']}",
            f"- verdict: {summary['verdict']}",
            "",
        ]
        summary_path.write_text("\n".join(md_lines))

        summary["saved_files"] = {
            "mapping_table_csv": str(table_path),
            "coarse_counts_csv": str(counts_path),
            "coarse_counts_by_race_csv": str(counts_by_race_path),
            "unmapped_links_by_race_csv": str(unmapped_path),
            "compact_summary_json": str(compact_path),
            "markdown_summary": str(summary_path),
        }

    return summary
