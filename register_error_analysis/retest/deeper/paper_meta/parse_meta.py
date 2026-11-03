#!/usr/bin/env python3
"""Parse cached Europe PMC / Crossref / PubMed / bioRxiv records into one row per DOI (paper_meta.csv),
plus an author table (paper_authors.csv) for co-authorship analyses."""
import glob, hashlib, json, os, re
import numpy as np
import pandas as pd

DEEP = "/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad/deeper"
CACHE = f"{DEEP}/cache/paper_meta"
OUT = "/home/user/SAbR/register_error_analysis/retest/deeper/paper_meta"


def key(doi):
    return hashlib.md5(doi.lower().encode()).hexdigest()


def load(sub, doi):
    p = f"{CACHE}/{sub}/{key(doi)}.json"
    return json.load(open(p)) if os.path.exists(p) else None


def dparts(x):
    try:
        p = x["date-parts"][0]
        return pd.Timestamp(year=p[0], month=p[1] if len(p) > 1 else 1, day=p[2] if len(p) > 2 else 1)
    except Exception:
        return pd.NaT


def todate(s):
    try:
        return pd.Timestamp(s)
    except Exception:
        return pd.NaT


US_STATES = ("Alabama|Alaska|Arizona|Arkansas|California|Colorado|Connecticut|Delaware|Florida|Georgia|Hawaii|Idaho|"
             "Illinois|Indiana|Iowa|Kansas|Kentucky|Louisiana|Maine|Maryland|Massachusetts|Michigan|Minnesota|"
             "Mississippi|Missouri|Montana|Nebraska|Nevada|New Hampshire|New Jersey|New Mexico|New York|"
             "North Carolina|North Dakota|Ohio|Oklahoma|Oregon|Pennsylvania|Rhode Island|South Carolina|"
             "South Dakota|Tennessee|Texas|Utah|Vermont|Virginia|Washington|West Virginia|Wisconsin|Wyoming")
COUNTRIES = [
    ("USA", r"\bUSA\b|U\.S\.A|United States|\bUS\b|\b(?:CA|MA|MD|NY|WA|TN|TX|NC|PA|GA|IL|MO|OR|WI|MI|OH|CT|NJ|MN|CO|IN|UT|AZ|FL|IA|VA|DC)\s+\d{5}|"
     + US_STATES + r"|Bethesda|Seattle|La Jolla|Boston|Nashville|Stanford|Pasadena|Cambridge, MA|Baltimore|Houston|Atlanta|Chicago|San Diego|San Francisco|New Haven|Durham"),
    ("China", r"China|Beijing|Shanghai|Shenzhen|Wuhan|Guangzhou|Hangzhou|Tianjin|Nanjing|Chengdu|Hefei|Xiamen|Changchun|Kunming|Harbin|Jinan|Suzhou"),
    ("Hong Kong", r"Hong Kong"), ("Taiwan", r"Taiwan|Taipei"),
    ("UK", r"\bUK\b|United Kingdom|England|Scotland|Wales|Oxford|Cambridge CB|London|Didcot|Dundee|Edinburgh|Glasgow"),
    ("Germany", r"Germany|Deutschland|Heidelberg|Munich|München|Berlin|Hamburg|Frankfurt|Martinsried|Göttingen|Dortmund"),
    ("France", r"France|Paris|Grenoble|Strasbourg|Lyon|Marseille|Montpellier"),
    ("Switzerland", r"Switzerland|Zurich|Zürich|Basel|Geneva|Lausanne|Bern"),
    ("Netherlands", r"Netherlands|Utrecht|Amsterdam|Leiden|Nijmegen|Groningen"),
    ("Belgium", r"Belgium|Ghent|Gent|Brussels|Leuven"), ("Sweden", r"Sweden|Stockholm|Uppsala|Lund"),
    ("Denmark", r"Denmark|Copenhagen|Aarhus"), ("Austria", r"Austria|Vienna|Wien|Klosterneuburg"),
    ("Italy", r"Italy|Milan|Rome|Siena|Padova"), ("Spain", r"Spain|Madrid|Barcelona"),
    ("Israel", r"Israel|Rehovot|Tel Aviv|Jerusalem"), ("Japan", r"Japan|Tokyo|Osaka|Kyoto|Yokohama"),
    ("Korea", r"Korea|Seoul|Daejeon"), ("Singapore", r"Singapore"), ("India", r"India|Bangalore|Bengaluru"),
    ("Australia", r"Australia|Melbourne|Sydney|Brisbane|Canberra|Victoria \d{4}"), ("Canada", r"Canada|Toronto|Vancouver|Montreal|Ontario|Quebec|Alberta|British Columbia"),
    ("Brazil", r"Brazil"), ("Russia", r"Russia"), ("South Africa", r"South Africa|Johannesburg|Cape Town"),
    ("Finland", r"Finland|Helsinki"), ("Norway", r"Norway|Oslo"), ("Ireland", r"Ireland|Dublin"),
    ("Czechia", r"Czech"), ("Poland", r"Poland"), ("Portugal", r"Portugal|Lisbon|Oeiras"),
    ("New Zealand", r"New Zealand|Auckland|Christchurch"), ("Mexico", r"Mexico\b|México"), ("Kenya", r"Kenya"),
    ("Thailand", r"Thailand|Bangkok"), ("Saudi Arabia", r"Saudi Arabia"), ("Argentina", r"Argentina"),
    ("Chile", r"Chile\b"), ("Vietnam", r"Vietnam|Viet Nam"), ("Turkey", r"Turkey|Türkiye"), ("Hungary", r"Hungary"),
    ("USA", r"\b(?:NY|Va|CA|MD|MA)\.$|\b(?:NY|Va|CA|MD|MA)\.\s"),
]


def country_of(aff):
    if not aff:
        return None
    aff = re.sub(r"\S+@\S+", "", aff)
    best, pos = None, -1
    for name, pat in COUNTRIES:
        for m in re.finditer(pat, aff):
            if m.start() > pos:
                best, pos = name, m.start()
    return best


FUNDER_PATTERNS = {
    "NIH": r"\bNIH\b|National Institutes? of Health|NIAID|NIGMS|National Institute of Allergy|National Cancer Institute|NCI\b|NHLBI|NIDA|NINDS|NIDDK|NIMH|NICHD|NIEHS|NEI\b",
    "NIAID": r"NIAID|National Institute of Allergy",
    "HHMI": r"Howard Hughes|HHMI",
    "Gates": r"Gates Foundation|Bill & Melinda|Bill and Melinda|Gates Found|CAVD",
    "Wellcome": r"Wellcome",
    "China_funders": r"National Natural Science Foundation of China|NSFC|Ministry of Science and Technology of (?:the People|China)|Chinese Academy|National Key R|Strategic Priority Research Program|China Postdoctoral|Beijing Municipal|Shanghai Municipal|Guangdong",
    "EU_ERC": r"European Research Council|\bERC\b|Horizon 2020|European Union|European Commission|H2020|Horizon Europe",
    "UKRI_MRC": r"Medical Research Council|\bMRC\b|UKRI|BBSRC|Biotechnology and Biological Sciences|UK Research",
    "DFG": r"Deutsche Forschungsgemeinschaft|\bDFG\b",
    "Japan_funders": r"AMED|Japan Agency for Medical|JSPS|Japan Society for the Promotion|MEXT",
    "CEPI": r"CEPI|Coalition for Epidemic",
    "DoD_DARPA": r"Department of Defense|DARPA|Defense Advanced|DTRA|Defense Threat|\bArmy\b|\bNavy\b|Joint Program Executive",
    "BARDA_HHS": r"BARDA|Biomedical Advanced Research|Department of Health and Human Services",
    "NSF": r"National Science Foundation(?! of China)",
    "SNSF": r"Swiss National Science",
    "CIHR_Canada": r"Canadian Institutes|CIHR|Canada Foundation|Genome Canada",
    "NHMRC": r"NHMRC|National Health and Medical Research Council",
    "Industry": r"\bInc\b|Inc\.|Ltd|GmbH|Pharmaceutic|Biotherapeutics|Regeneron|Vir Biotechnology|AstraZeneca|Merck|Pfizer|Genentech|Novartis|Sanofi|GSK|GlaxoSmithKline|Janssen|AbbVie|Amgen|Lilly|AbCellera|Adimab",
    "Fast_Grants_COVID": r"Fast Grants|COVID-19 Fast|Coronavirus|COVID|SARS-CoV-2|CoVIC",
}


def funders_text(epmc_rec, cr_msg, pm_xml):
    names = []
    if epmc_rec:
        for g in (epmc_rec.get("grantsList") or {}).get("grant", []):
            names.append(g.get("agency") or "")
    if cr_msg:
        for f in cr_msg.get("funder") or []:
            names.append(f.get("name") or "")
    if pm_xml:
        names += re.findall(r"<Agency>(.*?)</Agency>", pm_xml)
    return " | ".join(n for n in names if n)


def pubmed_xml(pmid):
    p = f"{CACHE}/pubmed/{pmid}.xml"
    return open(p).read() if pmid and os.path.exists(p) else None


def pm_history(xml):
    out = {}
    if not xml:
        return out
    for st, y, m, d in re.findall(r'<PubMedPubDate PubStatus="(\w+)">\s*<Year>(\d+)</Year>\s*<Month>(\d+)</Month>\s*<Day>(\d+)</Day>', xml):
        try:
            out.setdefault(st, pd.Timestamp(int(y), int(m), int(d)))
        except Exception:
            pass
    return out


KW = {
    "kw_covid": r"SARS-CoV-2|COVID|coronavirus|SARS-CoV|Omicron|sarbecovirus|spike",
    "kw_vaccine": r"vaccin|immunogen",
    "kw_neutraliz": r"neutrali[sz]",
    "kw_broad": r"broad(ly)?\b|pan-|cross-reactiv|cross-neutrali",
    "kw_structural_basis": r"structural basis|structural insight|molecular basis|structures? of|structural characteri[sz]",
    "kw_nanobody": r"nanobod|\bVHH|sybod|single-domain antibod|single domain antibod",
    "kw_fiducial": r"fiducial|chaperone|Fab-assisted|Fab-enabled|legobod|megabod|BRIL|Fab binding enables|enables (?:cryo|structure)|scaffold",
    "kw_antibody": r"antibod|\bFab\b|\bmAb|nanobod|immunoglobulin|\bIgG|\bIgA|\bIgM|\bIgE|B[- ]cell|scFv",
    "kw_hiv": r"\bHIV|Env\b|envelope glycoprotein",
    "kw_influenza": r"influenza|hemagglutinin|haemagglutinin|neuraminidase",
    "kw_receptor_mech": r"receptor|channel|transporter|GPCR|activation|inhibition|mechanism",
    "kw_therapeutic": r"therap|treatment|clinical|prophyla",
    "kw_escape": r"escape|variant|resist|evasion|mutation",
    "kw_human_infection": r"convalescent|infected individual|patient|donor|elite",
    "kw_cryoem_title": r"cryo-?EM|cryo-electron|electron cryo",
}


def main():
    P = pd.read_csv(f"{OUT}/paper_base.csv")
    dois = sorted(set(P.doi.dropna()))
    rows, authors = [], []
    for doi in dois:
        r = {"doi": doi}
        ep = load("epmc", doi)
        rec = (ep or {}).get("result") or []
        rec = rec[0] if rec else None
        cr = (load("crossref", doi) or {}).get("message")
        bx = load("biorxiv", doi) or {}
        r["epmc_found"] = rec is not None
        r["crossref_found"] = cr is not None
        pmid = rec.get("pmid") if rec else None
        xml = pubmed_xml(pmid)
        r["pmid"] = pmid
        r["pmcid"] = rec.get("pmcid") if rec else None
        # journal
        jt = None
        if rec:
            jt = ((rec.get("journalInfo") or {}).get("journal") or {}).get("title") or (rec.get("bookOrReportDetails") or {}).get("publisher")
        if not jt and cr:
            jt = (cr.get("container-title") or [None])[0] or (cr.get("institution") or [{}])[0].get("name")
        r["journal_epmc"] = jt
        r["publisher"] = cr.get("publisher") if cr else None
        r["cr_type"] = cr.get("type") if cr else None
        r["epmc_source"] = rec.get("source") if rec else None
        pts = (rec.get("pubTypeList") or {}).get("pubType", []) if rec else []
        if xml:
            pts += re.findall(r"<PublicationType UI=\"\w+\">(.*?)</PublicationType>", xml)
        pts_l = [p.lower() for p in pts]
        r["pubtypes"] = "|".join(sorted(set(pts)))
        r["pt_letter_brief"] = any(p in ("letter", "brief-report", "rapid-communication", "report") for p in pts_l)
        r["pt_preprint_only"] = (rec or {}).get("source") == "PPR" or (cr or {}).get("type") == "posted-content"
        r["pt_nih_extramural"] = any("n.i.h., extramural" in p for p in pts_l)
        r["pt_nih_intramural"] = any("n.i.h., intramural" in p for p in pts_l)
        r["pt_nonus_gov"] = any("non-u.s. gov" in p for p in pts_l)
        # dates
        r["epmc_first_pub"] = todate(rec.get("firstPublicationDate")) if rec else pd.NaT
        r["epmc_epub"] = todate(rec.get("electronicPublicationDate")) if rec else pd.NaT
        r["cr_pub_online"] = dparts(cr.get("published-online")) if cr and cr.get("published-online") else pd.NaT
        r["cr_pub_print"] = dparts(cr.get("published-print")) if cr and cr.get("published-print") else pd.NaT
        r["cr_created"] = dparts(cr.get("created")) if cr and cr.get("created") else pd.NaT
        asr = {a.get("name"): a.get("value") for a in (cr.get("assertion") or [])} if cr else {}
        hist = pm_history(xml)
        r["received"] = hist.get("received", todate(asr.get("received")))
        r["revised"] = hist.get("revised", pd.NaT)
        r["accepted"] = hist.get("accepted", todate(asr.get("accepted")))
        r["pm_aheadofprint"] = hist.get("aheadofprint", pd.NaT)
        cands = [r["epmc_first_pub"], r["cr_pub_online"], todate(asr.get("first_online")), r["pm_aheadofprint"]]
        cands = [x for x in cands if pd.notna(x)]
        r["pub_date"] = min(cands) if cands else (r["cr_pub_print"] if pd.notna(r["cr_pub_print"]) else r["cr_created"])
        # preprint links
        pre = set()
        for s in ["biorxiv", "medrxiv"]:
            for cc in bx.get(s) or []:
                if cc.get("preprint_doi"):
                    pre.add((cc["preprint_doi"].lower(), cc.get("preprint_date"), s))
        for rel in ((cr or {}).get("relation") or {}).get("has-preprint", []):
            if rel.get("id-type") == "doi":
                pre.add((rel["id"].lower(), None, "crossref"))
        ppr_ids = []
        for cc in ((rec or {}).get("commentCorrectionList") or {}).get("commentCorrection", []):
            if cc.get("type") == "Preprint in" and cc.get("source") == "PPR":
                ppr_ids.append(cc.get("id"))
            if cc.get("type") == "Update of":
                m = re.search(r"doi: (10\.\S+?)\.?$", cc.get("reference") or "")
                if m:
                    pre.add((m.group(1).lower(), None, "epmc_update_of"))
        r["ppr_ids"] = "|".join(ppr_ids)
        for pid in ppr_ids:
            pf = f"{CACHE}/ppr/{pid}.json"
            for x in (json.load(open(pf)) if os.path.exists(pf) else []):
                if x.get("doi"):
                    pre.add((x["doi"].lower(), x.get("firstPublicationDate"), x.get("publisher") or "epmc_ppr"))
        r["n_erratum"] = sum(1 for cc in ((rec or {}).get("commentCorrectionList") or {}).get("commentCorrection", []) if cc.get("type") == "Erratum in")
        r["n_comment_in"] = sum(1 for cc in ((rec or {}).get("commentCorrectionList") or {}).get("commentCorrection", []) if cc.get("type") == "Comment in")
        # first posting date of each preprint
        best = None
        for pdoi, pdate, src in pre:
            pp = load("preprint", pdoi) or {}
            d0 = pd.NaT
            if pp.get("versions"):
                ds = [todate(v.get("date")) for v in pp["versions"] if v.get("date")]
                d0 = min(ds) if ds else pd.NaT
                srv = pp.get("server")
                nver = len({v.get("version") for v in pp["versions"]})
            else:
                d0 = dparts(pp["posted"]) if pp.get("posted") else (dparts(pp["created"]) if pp.get("created") else todate(pdate))
                srv = pp.get("server") or src
                nver = np.nan
            if pd.isna(d0) and pdate:
                d0 = todate(pdate)
            if pd.notna(d0) and (best is None or d0 < best[1]):
                best = (pdoi, d0, srv, nver)
        r["all_preprint_dois"] = "|".join(sorted({x[0] for x in pre}))
        r["has_preprint"] = bool(pre) or bool(ppr_ids)
        r["preprint_doi"], r["preprint_date"], r["preprint_server"], r["preprint_nver"] = best if best else (None, pd.NaT, None, np.nan)
        # authors
        alist = ((rec or {}).get("authorList") or {}).get("author", [])
        if not alist and cr:
            alist = [{"fullName": f"{a.get('family','')} {a.get('given','')}", "lastName": a.get("family"),
                      "initials": "".join(w[0] for w in (a.get("given") or "").replace("-", " ").split() if w),
                      "authorId": {"type": "ORCID", "value": (a.get("ORCID") or "").split("/")[-1]} if a.get("ORCID") else None,
                      "authorAffiliationDetailsList": {"authorAffiliation": [{"affiliation": x.get("name")} for x in a.get("affiliation", [])]}}
                     for a in cr.get("author", []) if a.get("family")]
        alist = [a for a in alist if a.get("lastName") or a.get("collectiveName") is None and a.get("fullName")]
        r["n_authors"] = len(alist) if alist else np.nan
        def affs(a):
            return [x.get("affiliation") for x in ((a.get("authorAffiliationDetailsList") or {}).get("authorAffiliation") or []) if x.get("affiliation")]
        if alist:
            la, fa = alist[-1], alist[0]
            r["last_aff"] = " || ".join(affs(la))[:500]
            r["first_aff"] = " || ".join(affs(fa))[:500]
            r["last_country"] = country_of(affs(la)[0]) if affs(la) else None
            r["first_country"] = country_of(affs(fa)[0]) if affs(fa) else None
            cs = {country_of(x) for a in alist for x in affs(a)[:1]} - {None}
            r["n_countries"] = len(cs)
            r["countries"] = "|".join(sorted(cs))
            alla = " ".join(x for a in alist for x in affs(a))
            r["aff_industry"] = bool(re.search(FUNDER_PATTERNS["Industry"], alla))
            for i, a in enumerate(alist):
                oid = (a.get("authorId") or {}).get("value") if (a.get("authorId") or {}).get("type") == "ORCID" else None
                authors.append({"doi": doi, "pos": i, "n": len(alist), "last": a.get("lastName"),
                                "initials": a.get("initials"), "orcid": oid,
                                "aff": (affs(a)[0] if affs(a) else None)})
        # funders
        ft = funders_text(rec, cr, xml)
        r["funders_text"] = ft[:1500]
        r["n_funder_records"] = len([x for x in ft.split(" | ") if x])
        for fk, pat in FUNDER_PATTERNS.items():
            r["fund_" + fk] = bool(re.search(pat, ft))
        # OA / licence / citations
        lic = " ".join(l.get("URL", "") for l in (cr.get("license") or [])) if cr else ""
        r["oa_epmc"] = (rec or {}).get("isOpenAccess") == "Y"
        r["oa_cc"] = "creativecommons" in lic
        r["open_access"] = r["oa_epmc"] or r["oa_cc"]
        r["cites_epmc"] = (rec or {}).get("citedByCount", np.nan)
        r["cites_cr"] = cr.get("is-referenced-by-count", np.nan) if cr else np.nan
        r["has_review_relation"] = bool(((cr or {}).get("relation") or {}).get("has-review"))
        # title
        t = (rec or {}).get("title") or ((cr or {}).get("title") or [None])[0] or ""
        r["title"] = re.sub(r"<[^>]+>", "", t)
        for k, pat in KW.items():
            r[k] = bool(re.search(pat, r["title"], flags=re.I))
        r["mesh"] = "|".join(m.get("descriptorName", "") for m in ((rec or {}).get("meshHeadingList") or {}).get("meshHeading", []))
        rows.append(r)
    M = pd.DataFrame(rows)
    M.to_csv(f"{OUT}/paper_meta.csv", index=False)
    pd.DataFrame(authors).to_csv(f"{OUT}/paper_authors.csv.gz", index=False)
    print(M.shape, M.epmc_found.mean(), M.crossref_found.mean(), M.has_preprint.mean(),
          M.received.notna().mean(), M.accepted.notna().mean(), M.n_authors.notna().mean(), M.last_country.notna().mean())
    print(M.last_country.value_counts().head(15))


if __name__ == "__main__":
    main()
