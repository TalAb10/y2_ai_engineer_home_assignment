import urllib.request, json, sys

queries = [
    "דירת 3 חדרים בירושלים עד מליון שח",
    "דירת סטודיו להשכרה בתל אביב עד 5000 שח",
    "דירה 4 חדרים עם מעלית וחניה בחיפה",
    "פנטהאוז עם נוף לים 5 חדרים בנתניה עד 4 מיליון",
    "דירה בין 1.5 מיליון ל-2 מיליון",
    "טויוטה קורולה 2018-2021 עד 70 אלף שח צבע לבן",
    "טסלה מודל 3 חשמלי עד 150000 שח",
    "יונדאי טוסון עד 80000 ק״מ",
    "יונדי טוסון 2020",
    "אייפון 13 פרו 256 ג׳יגה כמו חדש עד 2500",
    "מחשב נייד HP i7 16 גיגה RAM עד 3000 שח",
    "ספה פינתית עד 2000 שח",
]

for q in queries:
    data = json.dumps({"q": q}).encode("utf-8")
    req = urllib.request.Request(
        "http://localhost:8000/parse", data=data,
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as r:
        result = json.loads(r.read().decode("utf-8"))
    sys.stdout.buffer.write(f"Q: {q}\n".encode("utf-8"))
    sys.stdout.buffer.write(f"   category: {result['category']}\n".encode("utf-8"))
    sys.stdout.buffer.write(f"   params:   {json.dumps(result['params'], ensure_ascii=False)}\n".encode("utf-8"))
    if result["notes"]:
        sys.stdout.buffer.write(f"   notes:    {result['notes']}\n".encode("utf-8"))
    sys.stdout.buffer.write(b"\n")
    sys.stdout.buffer.flush()
