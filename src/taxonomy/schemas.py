"""Per-vertical Pydantic output schemas.

These are the ONLY allowed shapes for `params` in the API response.
The validate node runs the user-facing response through these models to:
  - Drop any keys not declared here (injection / hallucination guard)
  - Coerce and validate types
  - Enforce enum membership
  - Enforce numeric range limits from the taxonomy

Range fields (price, rooms, etc.) use the shared RangeField model so that both
`{"min": 0, "max": 1000000}` and `{"max": 70000}` (partial ranges) are valid.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic.functional_validators import AfterValidator


class RangeField(BaseModel):
    """Numeric range — both bounds optional to allow open-ended ranges."""

    model_config = ConfigDict(extra="forbid")

    min: float | None = None
    max: float | None = None


def _bounded_range(lo: float, hi: float) -> type:
    """Return an Annotated RangeField that enforces min/max within [lo, hi]."""
    def _validate(v: RangeField) -> RangeField:
        if v.min is not None and not (lo <= v.min <= hi):
            raise ValueError(f"min must be between {lo} and {hi}, got {v.min}")
        if v.max is not None and not (lo <= v.max <= hi):
            raise ValueError(f"max must be between {lo} and {hi}, got {v.max}")
        if v.min is not None and v.max is not None and v.min > v.max:
            raise ValueError(f"min ({v.min}) must not exceed max ({v.max})")
        return v
    return Annotated[RangeField, AfterValidator(_validate)]


def _bounded_int(lo: int, hi: int) -> type:
    """Return an Annotated int that enforces value within [lo, hi]."""
    def _validate(v: int) -> int:
        if not (lo <= v <= hi):
            raise ValueError(f"must be between {lo} and {hi}, got {v}")
        return v
    return Annotated[int, AfterValidator(_validate)]


# ── Real Estate ────────────────────────────────────────────────────────────────

RealEstatePropertyType = Literal[
    "דירה", "דירת גן", "דופלקס", "פנטהאוז", "גג/טיפוס", "מיני פנטהאוז",
    "דירת סטודיו", "דירת נופש", "קוטג׳", "דו משפחתי", "בית פרטי/וילה",
    "יחידת דיור", "מרתף", "דירת יוקרה", "דירה בבניין לשימור",
]

RealEstateTransactionMode = Literal[
    "מכירה", "השכרה", "שותפים", "מסחרי", "מגרשים",
]

RealEstateCondition = Literal["חדש", "משופץ", "שמור", "דורש שיפוץ"]

RealEstateFurnishing = Literal["ללא", "חלקי", "מלא"]

RealEstateOwnership = Literal["פרטי", "קבלן", "תיווך", "חברה"]

RealEstateLandOwnership = Literal["טאבו", "חכירה", "חברה משכנת"]

RealEstateDirection = Literal["צפון", "דרום", "מזרח", "מערב"]

RealEstateProximity = Literal[
    "קרוב לרכבת", "קרוב לקו מטרו עתידי", "קרוב לבתי ספר", "קרוב לים", "קרוב לפארק",
]

# Bounded types — taxonomy limits
_RoomsRange = _bounded_range(1, 12)
_RoomsInt   = _bounded_int(1, 12)
_FloorRange = _bounded_range(0, 60)
_FloorInt   = _bounded_int(0, 60)
_PriceRangeRE = _bounded_range(0, 50_000_000)
_SqmBuiltRange = _bounded_range(10, 1000)
_SqmLandRange  = _bounded_range(0, 10_000)


class RealEstateParams(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    סוגי_נכס: list[RealEstatePropertyType] | None = None
    מצבי_עסקה: list[RealEstateTransactionMode] | None = None
    עיר: str | None = None
    שכונה: str | None = None
    רחוב: str | None = None
    מחיר: _PriceRangeRE | None = None
    מס_חדרים: _RoomsRange | _RoomsInt | None = Field(
        default=None,
        alias="מס׳_חדרים",
        description="Number of rooms; exact value or range.",
    )
    קומה: _FloorRange | _FloorInt | None = None
    סה_כ_קומות: _bounded_int(1, 100) | None = Field(default=None, alias="סה״כ_קומות")
    מ_ר_בנוי: _SqmBuiltRange | None = Field(default=None, alias="מ״ר_בנוי")
    מ_ר_מגרש: _SqmLandRange | None = Field(default=None, alias="מ״ר_מגרש")
    מרפסות: _bounded_int(0, 6) | None = None
    מרפסת_שמש: bool | None = None
    מעלית: bool | None = None
    חניה: _bounded_int(0, 4) | None = None
    מחסן: bool | None = None
    מיזוג: bool | None = None
    מממ_ד: bool | None = Field(default=None, alias="ממ״ד")
    גישה_לנכים: bool | None = None
    חיות_מחמד: bool | None = None
    מצב_נכס: RealEstateCondition | None = None
    ריהוט: RealEstateFurnishing | None = None
    בעלות: RealEstateOwnership | None = None
    בעלות_מקרקעין: RealEstateLandOwnership | None = None
    כיווני_אוויר: list[RealEstateDirection] | None = None
    קרבה: list[RealEstateProximity] | None = None
    תאריך_כניסה: str | None = None
    ארנונה_חודשית: RangeField | None = None


# ── Vehicles ───────────────────────────────────────────────────────────────────

VehicleType = Literal[
    "פרטי", "מסחרי", "אופנוע", "קטנוע", "משאית", "רכב שטח", "רכב היברידי/חשמלי",
]

VehicleFuelType = Literal[
    "בנזין", "דיזל", "היברידי", "היברידי נטען", "חשמלי", "גז",
]

VehicleGearbox = Literal["אוטומטית", "ידנית", "רובוטית", "CVT"]

VehicleColor = Literal["לבן", "שחור", "אפור", "כסוף", "כחול", "אדום", "ירוק", "צהוב"]

VehicleOwnership = Literal["פרטי", "ליסינג", "חברה", "השכרה", "יבוא אישי"]

VehicleSafetyFeature = Literal[
    "בלימה אוטונומית", "שמירת נתיב", "בקרת שיוט אדפטיבית", "תצוגת שטחים מתים",
]

VehicleAccessory = Literal[
    "גג שמש", "מושבי עור", "חיישני רוורס", "מצלמה אחורית", "מולטימדיה",
]

# Bounded types — taxonomy limits
_YearRange  = _bounded_range(1980, 2025)
_YearInt    = _bounded_int(1980, 2025)
_KmRange    = _bounded_range(0, 1_000_000)
_EngineRange = _bounded_range(600, 6000)
_HpRange    = _bounded_range(40, 1200)
_EvRangeKm  = _bounded_range(0, 700)
_PriceRangeV = _bounded_range(5000, 2_000_000)


class VehicleParams(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    סוג_רכב: VehicleType | None = None
    יצרן: str | None = None
    דגם: str | None = None
    תת_דגם: str | None = None
    שנה: _YearRange | _YearInt | None = None
    יד: _bounded_int(1, 20) | None = None
    ק_מ: _KmRange | None = Field(default=None, alias="ק״מ")
    נפח_מנוע_סמ_ק: _EngineRange | None = Field(default=None, alias="נפח_מנוע_סמ״ק")
    הספק_כ_ס: _HpRange | None = Field(default=None, alias="הספק_כ״ס")
    תיבת_הילוכים: VehicleGearbox | None = None
    סוג_דלק: VehicleFuelType | None = None
    טווח_חשמלי_ק_מ: _EvRangeKm | None = Field(default=None, alias="טווח_חשמלי_ק״מ")
    טעינה_מהירה: bool | None = None
    מחיר: _PriceRangeV | None = None
    צבע: VehicleColor | str | None = None
    בעלות: VehicleOwnership | None = None
    טסט_עד: str | None = None
    מערכות_בטיחות: list[VehicleSafetyFeature] | None = None
    אבזור: list[VehicleAccessory] | None = None
    מספר_בעלים_קודמים: _bounded_int(0, 20) | None = None


# ── Second-hand ────────────────────────────────────────────────────────────────

SecondHandCondition = Literal["חדש", "כמו חדש", "משומש", "לחלפים"]

SecondHandRegion = Literal["צפון", "חיפה", "מרכז", "שרון", "שפלה", "ירושלים", "דרום"]

# Bounded types — taxonomy limits
_SHYearRange = _bounded_range(1980, 2025)
_SHYearInt   = _bounded_int(1980, 2025)
_RamRange    = _bounded_range(4, 128)
_RamInt      = _bounded_int(4, 128)
_StorageRange = _bounded_range(128, 8192)
_StorageInt   = _bounded_int(128, 8192)
_TvSizeRange  = _bounded_range(24, 100)
_TvSizeInt    = _bounded_int(24, 100)


class SecondHandParams(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    סקטור: str | None = None
    תת_קטגוריה: str | None = None
    מותג: str | None = None
    דגם: str | None = None
    מצב: SecondHandCondition | None = None
    מחיר: RangeField | None = None
    אזור: SecondHandRegion | None = None
    עיר: str | None = None
    צבע: str | None = None
    שנת_ייצור: _SHYearRange | _SHYearInt | None = None
    # Electronics — phones
    נפח_אחסון: str | None = None
    # Electronics — laptops
    מעבד: str | None = None
    זיכרון_RAM: _RamRange | _RamInt | None = None
    אחסון_GB: _StorageRange | _StorageInt | None = None
    # TVs
    גודל_אינצ: _TvSizeRange | _TvSizeInt | None = Field(default=None, alias="גודל_אינצ׳")
    טכנולוגיה: str | None = None
    רזולוציה: str | None = None
    # Bikes
    גודל_גלגל: str | None = None
    סוג: str | None = None


# ── Union for runtime dispatch ─────────────────────────────────────────────────

CATEGORY_TO_SCHEMA: dict[str, type[RealEstateParams] | type[VehicleParams] | type[SecondHandParams]] = {
    "נדל״ן": RealEstateParams,
    "רכב": VehicleParams,
    "יד_שנייה": SecondHandParams,
}

VALID_CATEGORIES = frozenset(CATEGORY_TO_SCHEMA.keys())
