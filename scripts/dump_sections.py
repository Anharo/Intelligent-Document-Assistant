import json

with open("data/processed/handbook_parsed.json") as f:
    records = json.load(f)

sections = sorted(set(r["section"] for r in records))
for s in sections:
    print(repr(s))