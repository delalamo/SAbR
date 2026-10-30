"""RCSB search helpers: representative polymer entities per Pfam family."""
import json, sys, requests

URL = "https://search.rcsb.org/rcsbsearch/v2/query"


def pfam_reps(pfam, cutoff=95, maxres=2.8, rows=10000):
    q = {"query": {"type": "group", "logical_operator": "and", "nodes": [
        {"type": "terminal", "service": "text", "parameters": {
            "attribute": "rcsb_polymer_entity_annotation.annotation_id", "operator": "exact_match", "value": pfam}},
        {"type": "terminal", "service": "text", "parameters": {
            "attribute": "rcsb_entry_info.resolution_combined", "operator": "less_or_equal", "value": maxres}}]},
        "return_type": "polymer_entity",
        "request_options": {"group_by": {"aggregation_method": "sequence_identity", "similarity_cutoff": cutoff,
                                         "ranking_criteria_type": {"sort_by": "rcsb_entry_info.resolution_combined", "direction": "asc"}},
                            "group_by_return_type": "representatives", "paginate": {"start": 0, "rows": rows}}}
    r = requests.post(URL, json=q, timeout=120)
    if r.status_code == 204:
        return 0, []
    r.raise_for_status()
    j = r.json()
    return j.get("total_count"), [x["identifier"] for x in j["result_set"]]


if __name__ == "__main__":
    for pf in sys.argv[1:]:
        for c in (95, 70):
            tot, reps = pfam_reps(pf, c)
            print(pf, c, tot, len(reps))
