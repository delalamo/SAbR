import re
AB_TITLE=re.compile(r'antibod|\bfabs?\b|fab[- ]|\bfab\d|nanobod|\bvhh|\bscfv|sybod|\bnb[- ]?\d|\bigg|\biga\b|\bigm\b|\bige\b|\bmab|immunoglobulin|megabod|legobod|\bsab\b|\bfv\b|fragment antigen',re.I)
AUX=re.compile(r'nanobody[- ]?35|\bnb[- ]?35\b|scfv[- ]?16|\bfv[- ]?16|\bbril\b|bag2|nabfab|anti[- ]fab|fab[- ]specific|fab[- ]binding nanobod|fab[- ]nanobod|nanobody[- ]fab|trimbod|anti[- ]bril|synthetic (antibod|nanobod|fab|binder)|\bsab\b|\bsab[- ]?\d|sybod|megabod|legobod|fiducial|chaperone|nb-?bril|anti-?gfp|anti-?alfa|alfa[- ]?tag',re.I)
FOCUS=re.compile(r'neutrali[sz]|epitope|\bbnab|broadly|germline|paratope|immunogen|vaccin|autoantibod|therapeutic|elicit|protective|escape|cross[- ]?react|monoclonal|\bmab|antibody[- ]dependent|bispecific|biparatopic|(agonist|antagonist|inhibitory|blocking) antibod|antibody (recogni|target|bind)|recogni[sz]ed by|human antibod|infected|convalescent|vaccinee|\b\w{3,}mab\b|\bregn\d|ly-cov|\bazd\d|okt3',re.I)
KW_TARGET=re.compile(r'MEMBRANE|TRANSPORT|HYDROLASE|TRANSFERASE|BIOSYNTHETIC|LIPID|OXIDOREDUCTASE|LYASE|ISOMERASE|LIGASE|CHAPERONE|STRUCTURAL|CELL CYCLE|DNA|RNA|TRANSCRIPTION|MOTOR|PHOTOSYNTHESIS|METAL|ELECTRON|SUGAR|CELL ADHESION|PROTEIN FIBRIL|ENDOCYTOSIS|EXOCYTOSIS|SIGNALING',re.I)
PATHO=re.compile(r'virus|viridae|\bphage|plasmodium|toxoplasma|trypanosoma|leishmania|clostridi|bacillus anthracis|staphylococcus aureus|streptococcus pyogenes|neisseria|bordetella|vibrio cholerae|mycobacterium tuberculosis|borrel|chlamydia|helicobacter|pseudomonas aeruginosa|klebsiella|salmonella|shigella|yersinia|schistosoma|candida|babesia|onchocerca|SARS|HIV|influenza|ebola|dengue|zika',re.I)
def classify(title,keywords,entities):
    """entities: list of dict(desc, ab:bool, tcr:bool, length, orgs:str)"""
    title=title or ''; kw=keywords or ''
    abents=[e for e in entities if e['ab']]
    other=[e for e in entities if not e['ab'] and not e['tcr'] and (e['length'] or 0)>=30]
    if not abents: return None
    abdesc=' '.join(e['desc'] or '' for e in abents)
    if not other: return 'Antibody alone'
    if AUX.search(title) or AUX.search(abdesc): return 'Auxiliary'
    if FOCUS.search(title): return 'Antibody of interest'
    if re.search(r'VIRAL|VIRUS|TOXIN|ANTIVIRAL',kw,re.I) or any(PATHO.search(e['orgs'] or '') for e in other): return 'Antibody of interest'
    imm=bool(re.search(r'IMMUN',kw,re.I)); tgt=bool(KW_TARGET.search(kw))
    if imm and not tgt: return 'Antibody of interest'
    if not AB_TITLE.search(title): return 'Auxiliary'
    if tgt and not imm: return 'Auxiliary'
    return 'Unclear'
