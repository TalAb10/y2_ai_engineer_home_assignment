"""Node: complete — taxonomy-driven inference of implied fields.

Extraction finds what the query *states*; this stage fills in what those values
*imply*. It runs after extract (so all values are already canonical taxonomy
forms) and before validate (so the inferred fields are schema-checked too).

Current rule (second-hand): infer the brand. A second-hand item often names a
product, not a brand — "אייפון" rather than "אפל". The brand is obvious and lives
in the taxonomy, so we resolve it from:
  1. a colloquial product alias  (אייפון → אפל), or
  2. a brand the user typed directly (the taxonomy's brand lists).
…and, when a brand pins down a single subcategory, backfill the subcategory/sector.

Every inferred value is validated against the taxonomy: a brand is only set if it
is listed under the (known) subcategory, so this stage can never invent a value.
This generalises the vehicle model→manufacturer backfill that extract already does.
"""

from __future__ import annotations

import re

from graph.context import NodeContext
from graph.state import GraphState
from patterns.segment_types import CAT_SH
from taxonomy.loader import strip_prefixes

_WORD_RE = re.compile(r"\S+")


def run(state: GraphState, ctx: NodeContext) -> dict:
    if state.category != CAT_SH:
        return {}

    params = dict(state.params)
    tax = ctx.taxonomy
    if "מותג" in params:
        return {}

    subcat = params.get("תת_קטגוריה")

    def valid_for_subcat(brand: str) -> bool:
        # No subcategory yet → accept; otherwise the brand must be listed under it.
        return subcat is None or subcat in tax.sh_brand_to_subcats.get(brand, set())

    brand = _find_brand(state.clean_q, tax, valid_for_subcat)
    if not brand:
        return {}

    params["מותג"] = brand

    # If the brand pins down exactly one subcategory and we don't have one, backfill it.
    if subcat is None:
        subcats = tax.sh_brand_to_subcats.get(brand, set())
        if len(subcats) == 1:
            only = next(iter(subcats))
            params["תת_קטגוריה"] = only
            params.setdefault("סקטור", tax.sh_subcat_to_sector.get(only))

    return {"params": params}


def _find_brand(clean_q: str, tax, valid_for_subcat) -> str | None:
    """First product-alias or direct brand in the query that fits the subcategory.

    Aliases win over direct brands: "אייפון" should resolve to אפל even though אפל
    is also a directly-typeable brand.
    """
    words = [m.group() for m in _WORD_RE.finditer(clean_q)]
    for lookup in (tax.sh_product_aliases, {b: b for b in tax.sh_brands}):
        for word in words:
            for variant in strip_prefixes(word):
                brand = lookup.get(variant)
                if brand and valid_for_subcat(brand):
                    return brand
    return None
