import json, os, time, urllib.parse, urllib.request, sys
C='/tmp/claude-0/-home-user-SAbR/6a27914d-ea1e-53f7-8565-6fae32e14167/scratchpad/deeper/cache/literature_dean/epmc'
def get(url):
    with urllib.request.urlopen(url,timeout=60) as r: return r.read()
extra={'iucrj2024':'PMC11533997','cms2022_actad':'PMC9248842'}
for k,pm in extra.items():
    fx=f'{C}/{k}.xml'
    if not os.path.exists(fx):
        try: open(fx,'wb').write(get(f'https://www.ebi.ac.uk/europepmc/webservices/rest/{pm}/fullTextXML'))
        except Exception as e: print(k,e)
Q={'croll_nsmb':'TITLE:"Making the invisible enemy visible" AND SRC:MED',
   'scalop':'TITLE:"sequence-based antibody canonical loop structure annotation"',
   'pyigclassify2b':'AUTH:"Dunbrack RL" AND TITLE:"PyIgClassify"',
   'kelow_pub':'TITLE:"canonical antibody CDR conformations" AND AUTH:"Dunbrack"',
   'abangle2':'TITLE:"ABangle" OR (TITLE:"VH-VL orientation" AND AUTH:"Deane")',
   'h3_kinked':'AUTH:"Deane CM" AND TITLE:"H3" AND PUB_YEAR:[2019 TO 2025]',
   'ab_cryoem_fab_model':'TITLE:"antibody" AND TITLE:"cryo-EM" AND (TITLE:"model building" OR TITLE:"validation" OR TITLE:"errors")',
   'cdr_length_pdb':'TITLE:"CDR" AND TITLE:"length" AND (TITLE:"Protein Data Bank" OR TITLE:"structural")',
   'germline_pdb_coverage':'(TITLE:"germline" AND TITLE:"structures") AND (TITLE:"antibody" OR TITLE:"antibodies") AND PUB_YEAR:[2015 TO 2026]',
   'teplyakov_germline':'AUTH:"Teplyakov A" AND TITLE:"germline"',
   'abnumber_errors':'TITLE:"numbering" AND TITLE:"antibod" AND PUB_YEAR:[2015 TO 2026]',
   'stanfield_pdb_ab':'AUTH:"Stanfield RL" AND TITLE:"antibod" AND (TITLE:"database" OR TITLE:"PDB")',
   'vhvl_dunbrack':'AUTH:"Dunbrack" AND TITLE:"VH-VL"',
   'nanobody_struct_stats':'TITLE:"nanobody" AND TITLE:"structures" AND (TITLE:"analysis" OR TITLE:"landscape") AND PUB_YEAR:[2019 TO 2026]',
   'emdb_validation':'TITLE:"EMDB" AND TITLE:"validation" AND PUB_YEAR:[2019 TO 2026]',
   'wwpdb_em_validation':'TITLE:"validation" AND TITLE:"3DEM" ',
   'abdb_ab_conformations':'TITLE:"antibody" AND TITLE:"elbow angle"',
   'kink_h3':'TITLE:"kinked" AND TITLE:"H3"',
}
for k,q in Q.items():
    fj=f'{C}/{k}.json'
    if not os.path.exists(fj):
        u='https://www.ebi.ac.uk/europepmc/webservices/rest/search?'+urllib.parse.urlencode({'query':q,'format':'json','resultType':'core','pageSize':8})
        try: open(fj,'wb').write(get(u))
        except Exception as e: print(k,'ERR',e); continue
    r=json.load(open(fj)).get('resultList',{}).get('result',[])
    print('##',k,len(r))
    for a in r[:8]:
        print(f"   {a.get('authorString','')[:50]} | {a.get('title','')[:110]} | {a.get('journalInfo',{}).get('journal',{}).get('isoabbreviation','')} {a.get('pubYear')} | {a.get('doi')} | {a.get('pmcid')}")
