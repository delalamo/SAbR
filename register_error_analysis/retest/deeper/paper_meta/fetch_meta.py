#!/usr/bin/env python3
"""Fetch publication metadata for every DOI cited by an antibody entry deposited 2018 onward.

Sources (official APIs only), cached per DOI as JSON under CACHE:
  * Europe PMC REST (core record: journal, pub types, dates, authors+affiliations, grants, OA, citations,
    preprint links via commentCorrectionList)
  * Crossref works (assertions with received/accepted/first-online for Springer Nature, has-preprint relation,
    funders, is-referenced-by-count, licence)
  * bioRxiv/medRxiv "pubs" endpoint (preprint DOI + posting date for a published DOI)
  * NCBI PubMed efetch XML (History received/revised/accepted/aheadofprint dates, publication types,
    affiliations, grant agencies) in batches of 200 PMIDs
  * For each preprint DOI found: bioRxiv/medRxiv "details" (all versions -> first posting date) or Crossref
    (posted/created date) for other servers (Research Square, SSRN, ...).
"""
import json, os, sys, time, hashlib, re, threading
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import requests

DEEP = "/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad/deeper"
CACHE = f"{DEEP}/cache/paper_meta"
for sub in ["epmc", "crossref", "biorxiv", "pubmed", "preprint"]:
    os.makedirs(f"{CACHE}/{sub}", exist_ok=True)

S = requests.Session()
S.headers["User-Agent"] = "SAbR-register-error-analysis/0.1 (research script)"
locks = {k: threading.Lock() for k in ["epmc", "crossref", "biorxiv", "pubmed"]}
last = {k: 0.0 for k in locks}
MIN_GAP = {"epmc": 0.12, "crossref": 0.12, "biorxiv": 0.15, "pubmed": 0.4}


def throttle(api):
    with locks[api]:
        wait = MIN_GAP[api] - (time.time() - last[api])
        if wait > 0:
            time.sleep(wait)
        last[api] = time.time()


def key(doi):
    return hashlib.md5(doi.lower().encode()).hexdigest()


def get(api, url, params=None, tries=4):
    for t in range(tries):
        throttle(api)
        try:
            r = S.get(url, params=params, timeout=40)
            if r.status_code == 404:
                return None
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(2 * (t + 1))
                continue
            r.raise_for_status()
            return r
        except requests.RequestException as ex:
            time.sleep(2 * (t + 1))
            err = ex
    return "ERR"


def cached(sub, doi, fn):
    p = f"{CACHE}/{sub}/{key(doi)}.json"
    if os.path.exists(p):
        return json.load(open(p))
    out = fn(doi)
    if out == "ERR":
        return None
    json.dump(out, open(p, "w"))
    return out


def f_epmc(doi):
    r = get("epmc", "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
            {"query": f'DOI:"{doi}"', "resultType": "core", "format": "json", "pageSize": 5})
    if r is None or r == "ERR":
        return {"doi": doi, "result": None} if r is None else "ERR"
    res = r.json().get("resultList", {}).get("result", [])
    res = [x for x in res if (x.get("doi") or "").lower() == doi.lower()] or res
    # drop very large fields we do not need
    for x in res:
        for k in ["abstractText", "fullTextUrlList", "chemicalList", "dbCrossReferenceList"]:
            x.pop(k, None)
    return {"doi": doi, "result": res}


def f_crossref(doi):
    r = get("crossref", f"https://api.crossref.org/works/{doi}")
    if r is None:
        return {"doi": doi, "message": None}
    if r == "ERR":
        return "ERR"
    m = r.json().get("message", {})
    m.pop("reference", None)
    return {"doi": doi, "message": m}


def f_biorxiv(doi):
    out = {"doi": doi}
    for server in ["biorxiv", "medrxiv"]:
        r = get("biorxiv", f"https://api.biorxiv.org/pubs/{server}/{doi}")
        if r == "ERR":
            return "ERR"
        out[server] = None if r is None else r.json().get("collection", [])
    return out


def f_preprint(pdoi):
    """first posting date of a preprint DOI"""
    out = {"doi": pdoi}
    if pdoi.startswith("10.1101/") or pdoi.startswith("10.64898/"):
        for server in ["biorxiv", "medrxiv"]:
            r = get("biorxiv", f"https://api.biorxiv.org/details/{server}/{pdoi}")
            if r == "ERR":
                return "ERR"
            col = [] if r is None else r.json().get("collection", [])
            if col:
                out["server"] = server
                out["versions"] = [{"version": c.get("version"), "date": c.get("date"),
                                    "published": c.get("published"), "category": c.get("category"),
                                    "type": c.get("type"), "title": c.get("title"),
                                    "author_corresponding_institution": c.get("author_corresponding_institution")}
                                   for c in col]
                return out
    r = get("crossref", f"https://api.crossref.org/works/{pdoi}")
    if r == "ERR":
        return "ERR"
    if r is not None:
        m = r.json().get("message", {})
        out["server"] = m.get("institution", [{}])[0].get("name") if m.get("institution") else m.get("publisher")
        out["crossref_type"] = m.get("type")
        out["posted"] = m.get("posted")
        out["created"] = m.get("created")
        out["group_title"] = m.get("group-title")
    return out


def pubmed_batch(pmids):
    todo = [p for p in pmids if not os.path.exists(f"{CACHE}/pubmed/{p}.xml")]
    for i in range(0, len(todo), 150):
        chunk = todo[i:i + 150]
        r = get("pubmed", "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
                {"db": "pubmed", "id": ",".join(chunk), "retmode": "xml"})
        if r in (None, "ERR"):
            print("pubmed batch failed", i, file=sys.stderr)
            continue
        txt = r.text
        arts = re.findall(r"<PubmedArticle>.*?</PubmedArticle>", txt, flags=re.S)
        for a in arts:
            m = re.search(r'<PMID Version="\d+">(\d+)</PMID>', a)
            if m:
                open(f"{CACHE}/pubmed/{m.group(1)}.xml", "w").write(a)
        print(f"pubmed {i + len(chunk)}/{len(todo)}", file=sys.stderr)


def main():
    e = pd.read_csv(f"{DEEP}/entries.csv")
    e = e[pd.to_datetime(e.deposit) >= "2018-01-01"]
    dois = sorted({d.strip() for d in e.doi.dropna() if d.strip()})
    print(len(dois), "DOIs", file=sys.stderr)

    def run(sub, fn, items, workers=4):
        n = [0]
        def w(d):
            cached(sub, d, fn)
            n[0] += 1
            if n[0] % 200 == 0:
                print(sub, n[0], file=sys.stderr)
        with ThreadPoolExecutor(workers) as ex:
            list(ex.map(w, items))

    with ThreadPoolExecutor(3) as ex:
        futs = [ex.submit(run, "epmc", f_epmc, dois), ex.submit(run, "crossref", f_crossref, dois),
                ex.submit(run, "biorxiv", f_biorxiv, dois, 2)]
        for f in futs:
            f.result()

    # PubMed
    pmids = set()
    for d in dois:
        p = f"{CACHE}/epmc/{key(d)}.json"
        if os.path.exists(p):
            for x in json.load(open(p)).get("result") or []:
                if x.get("pmid"):
                    pmids.add(x["pmid"])
    pubmed_batch(sorted(pmids))

    # preprint DOIs
    pre = set()
    for d in dois:
        b = cached("biorxiv", d, f_biorxiv) or {}
        for s in ["biorxiv", "medrxiv"]:
            for c in b.get(s) or []:
                if c.get("preprint_doi"):
                    pre.add(c["preprint_doi"].lower())
        cr = (cached("crossref", d, f_crossref) or {}).get("message") or {}
        for rel in (cr.get("relation") or {}).get("has-preprint", []):
            if rel.get("id-type") == "doi":
                pre.add(rel["id"].lower())
        if d.lower().startswith("10.1101/") or d.lower().startswith("10.64898/") or d.lower().startswith("10.21203/"):
            pre.add(d.lower())
    print(len(pre), "preprint DOIs", file=sys.stderr)
    run("preprint", f_preprint, sorted(pre), 3)
    print("done", file=sys.stderr)


if __name__ == "__main__":
    main()


def fetch_missing_preprints():
    """second pass: every preprint DOI listed by parse_meta.py (all sources) gets a posting-date record"""
    M = pd.read_csv("/home/user/SAbR/register_error_analysis/retest/deeper/paper_meta/paper_meta.csv")
    pre = sorted({d for x in M.all_preprint_dois.dropna() for d in x.split("|") if d})
    todo = [d for d in pre if not os.path.exists(f"{CACHE}/preprint/{key(d)}.json")]
    print(len(pre), "preprints;", len(todo), "missing", file=sys.stderr)
    with ThreadPoolExecutor(3) as ex:
        list(ex.map(lambda d: cached("preprint", d, f_preprint), todo))
