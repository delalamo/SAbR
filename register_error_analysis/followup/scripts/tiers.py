import re
CNS={'nature','science','cell'}
SISTER_PREFIX=('nat','sciadv','sciimmunol','scitranslmed','scisignal')
CELLPRESS={'molcell','immunity','neuron','cellrep','cellreports','cellhostmicrobe','structure','cellchembiol','devcell','cancercell','cellmetab','cellstemcell','currbiol','chem','med','cellres','celldiscov','cellsyst','joule','cellgenomics','molther'}
def tier(j):
    if not isinstance(j,str) or not j.strip() or j.lower().startswith('to be published'): return 'Unpublished'
    k=re.sub(r'[^a-z]','',j.lower())
    if k in CNS: return 'CNS'
    if k.startswith(SISTER_PREFIX) or k in CELLPRESS: return 'CNS sister journal'
    return 'Other journal'
