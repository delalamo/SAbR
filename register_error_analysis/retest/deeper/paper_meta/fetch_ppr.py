#!/usr/bin/env python3
"""Resolve Europe PMC 'Preprint in' PPR ids (preprints on servers not linked through Crossref/bioRxiv, e.g.
Research Square, SSRN) to their DOI, server and first posting date. Cached under CACHE/ppr/<id>.json"""
import glob, json, os, time
import requests

DEEP = "/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad/deeper"
CACHE = f"{DEEP}/cache/paper_meta"
os.makedirs(f"{CACHE}/ppr", exist_ok=True)
S = requests.Session()
ids = set()
for f in glob.glob(f"{CACHE}/epmc/*.json"):
    for x in (json.load(open(f)).get("result") or [])[:1]:
        for cc in (x.get("commentCorrectionList") or {}).get("commentCorrection", []):
            if cc.get("type") == "Preprint in" and cc.get("source") == "PPR":
                ids.add(cc["id"])
print(len(ids), "PPR ids")
for i, pid in enumerate(sorted(ids)):
    p = f"{CACHE}/ppr/{pid}.json"
    if os.path.exists(p):
        continue
    for t in range(4):
        try:
            r = S.get("https://www.ebi.ac.uk/europepmc/webservices/rest/search",
                      params={"query": f"EXT_ID:{pid} AND SRC:PPR", "resultType": "core", "format": "json"}, timeout=40)
            r.raise_for_status()
            res = r.json().get("resultList", {}).get("result", [])
            out = [{"id": x.get("id"), "doi": x.get("doi"), "firstPublicationDate": x.get("firstPublicationDate"),
                    "publisher": (x.get("bookOrReportDetails") or {}).get("publisher"),
                    "versionNumber": x.get("versionNumber"), "title": x.get("title")} for x in res]
            json.dump(out, open(p, "w"))
            break
        except Exception:
            time.sleep(2 * (t + 1))
    time.sleep(0.12)
    if i % 100 == 0:
        print(i)
