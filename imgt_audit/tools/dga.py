"""Minimal client for IMGT/DomainGapAlign. Caches raw HTML responses."""
import hashlib, html, os, re, sys, time
import requests

URL = "https://www.imgt.org/3Dstructure-DB/cgi/DomainGapAlign.cgi"
CACHE = os.path.join(os.path.dirname(__file__), "..", "dga_cache")


def query(name, seq, domtype, displayed=1):
    os.makedirs(CACHE, exist_ok=True)
    key = hashlib.sha1(f"{domtype}|{seq}|{displayed}".encode()).hexdigest()[:16]
    path = os.path.join(CACHE, f"{name}_{domtype}_{key}.html")
    if os.path.exists(path):
        return open(path).read()
    data = {"Sequence": f">{name}\n{seq}", "domtype": domtype, "species": "",
            "SWScore": "0", "displayed": str(displayed), "evalue": "200",
            "gpenal": "-5", "gpenalref": "-20", "submit": "Show"}
    for attempt in range(4):
        try:
            r = requests.post(URL, files={k: (None, v) for k, v in data.items()}, timeout=180)
            r.raise_for_status()
            break
        except Exception as e:
            print("retry", e, file=sys.stderr); time.sleep(2 ** attempt * 2)
    open(path, "w").write(r.text)
    time.sleep(1.0)
    return r.text


def gapped(h):
    """Return list of (domain_label, gapped_seq, allele, species) from SeqFasta links."""
    out = []
    for m in re.finditer(r'href="SeqFasta\.cgi\?([^"]+)"', h):
        q = html.unescape(m.group(1))
        if "gaps=yes" not in q or q.endswith("&download"):
            continue
        d = dict(kv.split("=", 1) for kv in q.split("&") if "=" in kv)
        out.append(d)
    return out


def text(h):
    t = re.sub(r"<script.*?</script>", "", h, flags=re.S)
    t = re.sub(r"<style.*?</style>", "", t, flags=re.S)
    t = re.sub(r"<[^>]+>", " ", t)
    t = html.unescape(t)
    t = re.sub(r"[ \t]+", " ", t)
    return re.sub(r"\n\s*\n+", "\n", t)


def _header_tokens(raw_line):
    """Numbers in the header line with visible column spans; tags split tokens."""
    toks, col, cur, start, i = [], 0, "", None, 0
    while i < len(raw_line):
        c = raw_line[i]
        if c == "<":
            j = raw_line.index(">", i)
            if cur: toks.append((int(cur), start, col - 1)); cur = ""
            i = j + 1; continue
        if c == "&":
            j = raw_line.index(";", i)
            c = html.unescape(raw_line[i:j + 1]); i = j
        if c.isdigit():
            if not cur: start = col
            cur += c
        elif cur:
            toks.append((int(cur), start, col - 1)); cur = ""
        col += 1; i += 1
    if cur: toks.append((int(cur), start, col - 1))
    return toks


def _split_digit_run(run):
    """Return index where attachment switches from previous to next position."""
    vals = [10 if d == "0" else int(d) for d in run]
    n = len(vals)
    if n == 1 or all(vals[k] < vals[k + 1] for k in range(n - 1)):
        return n if vals[0] == 1 else 0
    if all(vals[k] > vals[k + 1] for k in range(n - 1)):
        return 0
    for k in range(n - 1):  # doubled peak, e.g. 1233 21
        if vals[k] == vals[k + 1]:
            return k + 1
    peak = vals.index(max(vals))
    return peak  # single peak goes to the following position (IMGT 112 rule)


def ruler_labels(hdr_raw, ruler):
    """IMGT label for every non-space ruler column, in order."""
    toks = _header_tokens(hdr_raw)
    anchors = {}
    for n, s, e in toks:
        for c in (s, e):
            if c < len(ruler) and ruler[c] == "|":
                anchors[c] = n; break
    cols = [i for i, c in enumerate(ruler) if c != " "]
    std = {}
    last = None
    for i in cols:
        ch = ruler[i]
        if ch in "|.":
            if i in anchors:
                last = anchors[i]
            elif last is not None:
                last += 1
            std[i] = last
    # back-fill standard positions before the first anchor
    first = min(anchors)
    prior = [i for i in cols if i < first and ruler[i] in "|."]
    for k, i in enumerate(reversed(prior)):
        std[i] = anchors[first] - 1 - k
    labels = {}
    k = 0
    while k < len(cols):
        i = cols[k]
        ch = ruler[i]
        if ch in "|.":
            labels[i] = str(std[i]); k += 1; continue
        m = k
        while m < len(cols) and ruler[cols[m]] not in "|.":
            m += 1
        run_cols = cols[k:m]
        prev = next((std[c] for c in reversed(cols[:k]) if c in std), None)
        nxt = next((std[c] for c in cols[m:] if c in std), None)
        run = "".join(ruler[c] for c in run_cols)
        if run.isdigit():
            split = _split_digit_run(run)
            for j, c in enumerate(run_cols):
                d = 10 if ruler[c] == "0" else int(ruler[c])
                base = prev if (j < split and prev is not None) or nxt is None else nxt
                labels[c] = f"{base}.{d}"
        else:
            for c in run_cols:
                base = prev if prev is not None else nxt
                labels[c] = f"{base}{ruler[c]}"
        k = m
    return labels


def parse(h):
    """List of domains: dict(label, gene, species, sw, ident, overlap, gapped, labels)."""
    rows = []
    t0 = h.find("Closest reference gene")
    t1 = h.find("Alignment with the closest")
    if t0 < 0 or t1 < 0:
        return []
    for tr in re.findall(r"<tr>(.*?)</tr>", h[t0:t1], re.S):
        cells = [html.unescape(re.sub(r"<[^>]+>", "", c)).strip()
                 for c in re.findall(r"<td[^>]*>(.*?)</td>", tr, re.S)]
        cells = [c for c in cells if c != ""]
        if len(cells) >= 7 and cells[2].isdigit():
            rows.append(dict(species=cells[0], gene=cells[1], dom=int(cells[2]), label=cells[3],
                             sw=int(cells[4]), ident=float(cells[5]), overlap=int(cells[6])))
    rows = [r for r in rows if r["label"] not in ("J-REGION", "J-LIKE")]
    seen, uniq = set(), []
    for r in rows:
        if r["dom"] not in seen:
            seen.add(r["dom"]); uniq.append(r)
    chunks = h.split("Alignment with the closest")[1:]
    out = []
    for r, ch in zip(uniq, chunks):
        pres = re.findall(r'<pre style="display:inline;">(.*?)</pre>', ch, re.S)
        lines = pres[0].split("\n")
        ruler = html.unescape(re.sub(r"<[^>]+>", "", lines[-1])).rstrip()
        hdr_raw = lines[-2]
        q = html.unescape(re.sub(r"<[^>]+>", "", pres[1].split("\n")[0])).rstrip()
        m = re.search(r'SeqFasta\.cgi\?domtype=\w&amp;seq=([^&]+)&amp;[^"]*gaps=yes[^"]*"', ch)
        ungapped = m.group(1).replace(".", "")
        if len(q) > len(ruler) or any(c not in " ." for c, rc in zip(q, ruler) if rc == " "):
            out.append(dict(r, error="query row inconsistent with DGA ruler")); continue
        lab = ruler_labels(hdr_raw, ruler)
        g = "".join(c if c != " " else "." for c, rc in zip(q.ljust(len(ruler), "."), ruler) if rc != " ")
        labs = [lab[i] for i, rc in enumerate(ruler) if rc != " "]
        if g.replace(".", "") != ungapped:
            out.append(dict(r, error="query row residues differ from DGA domain sequence")); continue
        r = dict(r, gapped=g, labels=labs, fasta_disagrees=(m.group(1) != g))
        out.append(r)
    return out
