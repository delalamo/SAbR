"""Fetch literature metadata (+ open-access full text) from Europe PMC REST for the reference papers.
Cache: scratchpad/deeper/cache/literature_dean/epmc/<key>.{json,xml}"""
import json, os, sys, time, urllib.parse, urllib.request
C='/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad/deeper/cache/literature_dean/epmc'
os.makedirs(C,exist_ok=True)
Q={
 'sabdab2014':'TITLE:"SAbDab: the structural antibody database"',
 'sabdab2022':'TITLE:"SAbDab in the age of biotherapeutics"',
 'sabdab2_2025':'TITLE:"SAbDab2"',
 'abangle2013':'TITLE:"ABangle"',
 'abangle_flex':'TITLE:"VH-VL" AND AUTH:"Deane" AND TITLE:"orientation"',
 'immunebuilder2023':'TITLE:"ImmuneBuilder"',
 'scalop2019':'TITLE:"SCALOP"',
 'nowak2016':'TITLE:"Length-independent structural similarities enrich the antibody CDR canonical class model"',
 'north2011':'TITLE:"A new clustering of antibody CDR loop conformations"',
 'pyigclassify2015':'TITLE:"PyIgClassify: a database of antibody CDR structural classifications"',
 'pyigclassify2':'TITLE:"PyIgClassify2"',
 'kelow2022':'TITLE:"penultimate classification of canonical antibody CDR conformations"',
 'regep2017':'TITLE:"The H3 loop of antibodies shows unique structural characteristics"',
 'gw2024':'TITLE:"canonical forms" AND AUTH:"Deane CM" AND PUB_YEAR:[2023 TO 2025]',
 'oas2018':'TITLE:"Observed Antibody Space: A Resource for Data Mining"',
 'oas2022':'TITLE:"Observed Antibody Space: A diverse database of cleaned"',
 'krawczyk2018':'TITLE:"Structurally Mapping Antibody Repertoires"',
 'kovaltsuk2020':'TITLE:"Structural diversity of B-cell receptor repertoires along the B-cell differentiation axis"',
 'raybould2024lambda':'TITLE:"lambda light chains" AND AUTH:"Raybould"',
 'covabdab':'TITLE:"CoV-AbDab"',
 'croll2021':'TITLE:"Making the invisible enemy visible"',
 'cms2022':'TITLE:"Sequence-assignment validation in cryo-EM models with checkMySequence"',
 'cms2023':'TITLE:"Sequence-assignment validation in protein crystal structure models with checkMySequence"',
 'conkit2022':'TITLE:"Using deep-learning predictions of inter-residue distances for model validation"',
 'conkit2024':'TITLE:"register errors in PDB deposits"',
 'wlodawer2018':'TITLE:"Detect, correct, retract"',
 'lawson2021':'TITLE:"Cryo-EM model validation recommendations based on outcomes of the 2019 EMDataResource challenge"',
 'bailey2018':'TITLE:"Locking the elbow"',
 'ama2':'TITLE:"Antibody modeling assessment II. Structures and models"',
 'weitzner2015':'TITLE:"The origin of CDR H3 structural diversity"',
 'anarci2016':'TITLE:"ANARCI: antigen receptor numbering and receptor classification"',
 'abb2_ablooper':'TITLE:"ABlooper"',
 'igfold':'TITLE:"Fast, accurate antibody structure prediction from deep learning on massive set of natural antibodies"',
 'thera_sabdab':'TITLE:"Thera-SAbDab"',
 'tap2019':'TITLE:"Five computational developability guidelines for therapeutic antibody profiling"',
 'modelangelo':'TITLE:"Automated model building and protein identification in cryo-EM maps"',
 'pintilie2020':'TITLE:"Measurement of atom resolvability in cryo-EM maps with Q-scores"',
 'dunbar2016_abodybuilder':'TITLE:"ABodyBuilder: Automated antibody structure prediction with data-driven accuracy estimation"',
 'marks2017':'TITLE:"Antibody H3 Structure Prediction"',
 'plabdab':'TITLE:"Patent and Literature Antibody Database"',
 'chojnowski_dls':'AUTH:"Chojnowski G" AND (TITLE:"register" OR TITLE:"sequence assignment")',
 'wilman2022':'TITLE:"Machine-designed biotherapeutics"',
 'fab_elbow_em':'TITLE:"Fab" AND TITLE:"cryo-EM" AND TITLE:"elbow"',
 'sars2_ab_review':'TITLE:"structural" AND TITLE:"SARS-CoV-2" AND TITLE:"antibodies" AND TITLE:"landscape"',
}
def get(url):
    for i in range(3):
        try:
            with urllib.request.urlopen(url,timeout=60) as r: return r.read()
        except Exception as e:
            time.sleep(2); err=e
    raise err
for k,q in Q.items():
    fj=f'{C}/{k}.json'
    if not os.path.exists(fj):
        u='https://www.ebi.ac.uk/europepmc/webservices/rest/search?'+urllib.parse.urlencode({'query':q,'format':'json','resultType':'core','pageSize':5})
        try: open(fj,'wb').write(get(u))
        except Exception as e: print(k,'ERR',e); continue
    r=json.load(open(fj)).get('resultList',{}).get('result',[])
    if not r: print(k,'NO HIT'); continue
    a=r[0]
    print(f"{k}: {a.get('authorString','')[:60]} | {a.get('title','')[:90]} | {a.get('journalInfo',{}).get('journal',{}).get('isoabbreviation',a.get('bookOrReportDetails',{}).get('publisher',''))} {a.get('pubYear')} | doi {a.get('doi')} | {a.get('pmcid')} | n={len(r)}")
    pm=a.get('pmcid')
    fx=f'{C}/{k}.xml'
    if pm and not os.path.exists(fx):
        try: open(fx,'wb').write(get(f'https://www.ebi.ac.uk/europepmc/webservices/rest/{pm}/fullTextXML'))
        except Exception as e: print('  no fulltext',e)
