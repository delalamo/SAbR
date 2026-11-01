"""Inject compact data and per-class notes into the comparison page."""
import json

D = json.load(open("/tmp/pan/viz/data.json"))
for d in D["domains"]:
    d["residues"] = [dict(aa=r["aa"], auth=r["auth"], imgt=r["imgt"], pan=r["pan"], kind=r["kind"][0],
                          ss=("F" if r["sheet"] == "ABED" else "B") if r["ss"] in "EB" and r["sheet"] in ("ABED", "A'GFCC'C''") else "",
                          **{"is": r["istr"], "ps": r["pstr"]}) for r in d["residues"]]
    d["occ_imgt"] = {int(k): v for k, v in d["occ_imgt"].items()}
    d["occ_pan"] = {int(k): v for k, v in d["occ_pan"].items()}

C1 = [
    "<b>A strand has no A′.</b> The residues IMGT numbers 9–16 are not at the V-domain A′ sites, so they become <code>8.1–8.7</code>, then 10, then 17.",
    "<b>C→D crossover.</b> IMGT's <code>45.x</code> residues form a short back-sheet strand at the V-domain C′ position: they take anchors 50–54.",
    "<b>D–E turn.</b> IMGT's <code>84.x/85.x</code> insertions become <code>82.x/83.x</code>, because 84 and 85 are structural anchors at the D and E strand ends.",
    "<b>E–F turn and G strand.</b> The EF turn gets <code>95.x</code> or <code>98.x/99.x</code>. In most C1 domains IMGT's G strand is one residue off from the V-domain G, so G labels shift +1 and the C-terminus runs to 127, 128, 129…",
    "C1 has no C″ strand, so the C″ anchors (65–76) stay empty; IMGT likewise jumps from <code>45.x</code> to 77.",
]
NOTES = {
    "VH": dict(status="ok", points=[
        "The reference shape for the frame: 96% of labels are identical to IMGT, and all ten strands are present.",
        "Differences are positional. At the 8–10 kink the gap moves from 10 to 9 when a residue is not at the A′ anchor; the C″–D loop (68–73) and an occasional <code>49.1</code> bulge in C′ are renumbered.",
        "The CDRs use IMGT's own rule, so CDR3 numbering is unchanged."]),
    "V-KAPPA": dict(status="ok", points=[
        "<b>CDR2 start, systematic.</b> In 98% of Vκ the first CDR2 residue continues the C′ strand: IMGT 56 becomes <code>55.1</code> and IMGT 57 takes anchor 56.",
        "D–E turn residues shift into the 82/83 positional numbers (25% of D-region labels).",
        "Everything else matches IMGT; 96% of labels are identical overall."]),
    "V-LAMBDA": dict(status="ok", points=[
        "Same CDR2 start as Vκ in 80% of domains (<code>55.1</code>).",
        "The A-strand kink (9/10) differs in about 16% of domains.",
        "94% of labels are identical to IMGT."]),
    "V-ALPHA": dict(status="ok", points=[
        "<b>C″ switches sheets.</b> In the representative, C″ is H-bonded to D on the ABED sheet, not at VH's C″ sites. Residues IMGT puts at 64–68 become positional (59, 60, 70, 71, 73).",
        "Small CDR1 boundary shifts; 92% of labels are identical to IMGT."]),
    "V-BETA": dict(status="ok", points=[
        "<b>C″ is one residue out of register</b> relative to VH: 58% of C″ claims shift by −1.",
        "An extra β-bulge residue at the C/C′ junction becomes <code>46.1</code>; the longer D–E loop (HV4) gets <code>80.1</code>.",
        "87% of labels are identical to IMGT."]),
    "V-GAMMA": dict(status="ok", points=[
        "Close to IMGT (96% identical). The A-strand kink and the D strand differ in some domains; <code>78.1</code> marks an extra residue (a bulge) inside the D strand.",
        "Based on 6 domains only."]),
    "V-DELTA": dict(status="ok", points=[
        "The C′ to C″ region differs: in the representative C″ pairs with D on the ABED sheet, as in Vα. About one <code>49.1</code> insertion per domain marks a C′ bulge.",
        "86% identical; based on 6 domains only."]),
    "V-LIKE": dict(status="ok", points=[
        "IgSF V-like domains vary most in C′, C″ and D: 68–81% of labels there differ from IMGT.",
        "Insertions collect at <code>49.x</code> (about 2.5 per domain), <code>71.x</code> and 79–83; the G strand is ±1.",
        "Some of these shifts are two residues or more; those are not yet verified (see Caveats)."]),
    "C-LIKE:IgSF": dict(status="ok", points=[
        "<b>The strand IMGT calls D is C′.</b> In the representative it is H-bonded into the A′GFCC′ sheet, and pan-Ig numbers it at C′ sites (49–56) across the class, so 97% of IMGT D-region labels change and D stays empty. This is the textbook C2-set topology.",
        "The A strand splits into A and A′ variably (−1/−2 shifts), as in I-set domains.",
        "The G strand shifts +1 in about half of domains, as in C1."]),
    "F:FN/TNC": dict(status="ok", points=[
        "<b>D → C′, as in C2.</b> IMGT's 75–84 strand is on the A′GFCC′ sheet; pan-Ig gives it C′ numbers (about 49–56), with insertions <code>49.x</code>.",
        "The first strand pairs with B on the ABED sheet, so pan-Ig calls it A where IMGT's numbers (9–15) imply A′.",
        "G-strand shifts of about 3 residues are reported but unverified: they may be the same artefact as TRAC."]),
    "F:other": dict(status="ok", points=[
        "Same pattern as fibronectin/tenascin FN3: D → C′ by sheet membership, the first strand is A on the ABED sheet, and the A and G registers vary.",
        "Based on 11 domains; G-strand shifts of two or more residues are unverified."]),
    "C-ALPHA": dict(status="warn", points=[
        "<b>Pan-Ig is wrong for TRAC.</b> It places the disulfide cysteines (IMGT C23/C104) at 21 and an unclaimed position, a two-residue slide of strand B.",
        "Those cysteines are bonded (Cβ–Cβ 3.9 Å) and superpose onto Cβ's and CH1's own disulfide (Cys23 within 0.6–0.8 Å), so IMGT's numbering is the structurally consistent one.",
        "Cause: TRAC's outer sheet is barely formed (no C, F or G strand in the representative), so superposition on the consensus core slides along strand B.",
        "Use IMGT numbering for TRAC until the disulfide-constrained fix is applied."]),
}
for c in ("CH1", "C-KAPPA", "C-LAMBDA", "CH2", "CH3", "C-BETA", "C-LIKE:MHC-I-a3", "C-LIKE:MHC-II", "C-LIKE:B2M"):
    NOTES[c] = dict(status="ok", points=C1)
html = open("/tmp/pan/viz/template.html").read()
html = html.replace("__DATA__", json.dumps(D, separators=(",", ":"))).replace("__NOTES__", json.dumps(NOTES))
open("/home/user/SAbR/imgt_audit/panig/imgt_vs_panig.html", "w").write(html)
print(len(html))
