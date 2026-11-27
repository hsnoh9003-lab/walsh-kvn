"""Extract retained numerical detail and reproduce the terminal observable checks.

Run from an unpacked release: python code/collect_details.py [SI.docx].
The optional original SI is needed only to re-extract the historical full tables.
"""
from pathlib import Path
from zipfile import ZipFile
import json, sys, re
import numpy as np
from lxml import etree as E

ROOT=Path(__file__).resolve().parents[1]
NS={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main','m':'http://schemas.openxmlformats.org/officeDocument/2006/math'}
tables={}
def table(name,header,rows):
    tables[name]=[header]+rows

inp=np.load(ROOT/'input/joint_readout_inputs.npz')
hist=np.load(ROOT/'input/direct_quantum_history.npz')
diag=json.loads((ROOT/'input/observable_diagnostics_reference.json').read_text())
p=inp['direct_p']; y=inp['direct_values']; mu=p@y; ref=inp['target']
cov=(y.T*p)@y-np.outer(mu,mu)
h=inp['M'].T@inp['G']@inp['M']/float(inp['norm'])**2
r=np.sqrt(y[:,2]); factor=np.array([p@r*(p@(y[:,0]/r)),p@r*(p@(y[:,1]/r)),mu[2]])
full_bias=100*np.sqrt((mu-ref)@h@(mu-ref))
factor_bias=100*np.sqrt((factor-ref)@h@(factor-ref))
assert abs(full_bias-0.10720784802292416)<1e-9
assert abs(factor_bias-0.25443931972612477)<1e-9
table('terminal_moments',['observable','circuit_mean','factorized_mean','continuum_mean','single_shot_variance','shots','expected_RMS_error'],[
    [s,float(mu[i]),float(factor[i]),float(ref[i]),float(cov[i,i]),815,float(np.sqrt((mu[i]-ref[i])**2+cov[i,i]/815))]
    for i,s in enumerate(['r_cos_Phi','r_sin_Phi','r_squared'])])
keys=list(diag['history'][0])
table('observable_history',keys,[[v[k] for k in keys] for v in diag['history']])
summary=[['correlated_field_bias',float(full_bias),'percent','matched continuum ensemble; noiseless circuit'],
         ['factorized_field_bias',float(factor_bias),'percent','same field map and r_squared mean'],
         ['force_calibration_start',10,'t_star','FEM-projected modes and surface forces'],
         ['force_calibration_end',60,'t_star','includes the Fig. 2e evaluation interval'],
         ['force_evaluation_start',40,'t_star','saved noiseless ensemble'],
         ['force_evaluation_end',float(hist['times'][-1]),'t_star','saved noiseless ensemble'],
         ['sampling_repetitions',3000,'count','Supplementary Data 8; each step/shot condition'],
         ['sampling_shots_low',1000,'count','steps 0, 11, 22, 33'],
         ['sampling_shots_high',8000,'count','steps 0, 11, 22, 33'],
         ['sampling_seed',20261107,'integer','Supplementary Data 8 replay'],
         ['sampling_max_absolute_MSE_z',2.044,'standard_errors','rounded value reported in SI v30'],
         ['LCHS_angle_normalization_allowance',6.2862e-8,'operator_norm_sum','before success conditioning'],
         ['BE_success_lower_bound',0.99981001,'probability','conservative bound used for conditioning'],
         ['BE_conditional_TV_bound',2e-4/(0.99981001-1e-4),'probability','2 eta/(P_s-eta), eta=1e-4'],
         ['Walsh_accepted_shots',815,'count','1% expected relative field RMS target'],
         ['BE_accepted_shots',813,'count','same field criterion'],
         ['classical_midpoint_field_error',0.002383,'percent','64-node matched characteristic quadrature']]
for k in ['radial_variance_reduction_pct','r2_variance_reduction_pct','field_variance_growth_pct','final_conditional_probe_fraction_pct']:
    summary.append([k,diag[k],'percent','reproduced from saved full joint circuit distributions'])
table('numerical_checks',['quantity','value','unit','definition'],summary)

table('analytical_protocols',['protocol','effective_tiles','levels','output_states_per_batch','input_error_coefficient','input_error_power','batch_cycles','acceptance_formula','acceptance_scope','SI_reference'],[
    ['15-to-1',11,1,1,35,3,11,'(1-e)^15','input-error-only analytical model',26],
    ['116-to-12',44,1,12,41.25,4,99,'(1-e)^116','input-error-only analytical model',26],
    ['116-to-12',81,1,12,41.25,4,50,'(1-e)^116','input-error-only analytical model',26],
    ['225-to-1',176,2,1,1500625,9,15,'(1-e)^15','final-stage approximation; not full two-level acceptance',26],
    ['modified 15-to-1',32,1,1,35,3,11,'(1-e)^15','includes additional correction structures',26],
    ['modified 116-to-12',126,1,12,41.25,4,53,'(1-e)^116','includes additional correction structures',26]])

source=Path(sys.argv[1]) if len(sys.argv)>1 else None
if source:
    with ZipFile(source) as z:doc=E.fromstring(z.read('word/document.xml'))
    ps=doc.findall('.//w:p',NS)
    def text(el):return ''.join(el.xpath('.//w:t/text()|.//m:t/text()',namespaces=NS))
    def top(p):
        while p.getparent().tag!='{'+NS['w']+'}body':p=p.getparent()
        return p
    t6=top(ps[617])
    rows=[[text(c) for c in row.findall('w:tc',NS)] for row in t6.findall('w:tr',NS)]
    hardware=rows[1:]
    for row,symbol in zip(hardware,['t_round','t_react','t_dec','c_circ','p_th','A_L','Q_surface(d)','b','O_1, O_l','','eta','code distances / n_T']):
        row[1]=symbol
        row[3]=row[3].replace('\u2060','')
    hardware[6][2]='2(d+1)^2'
    table('hardware_and_search',['parameter','symbol','value_or_formula','source'],hardware)
    t10=top(ps[968]); rows=[[text(c) for c in row.findall('w:tc',NS)] for row in t10.findall('w:tr',NS)][1:]
    parsed=[]
    for a,b,c,d,e in rows:
        amount,unit=e.rsplit(' ',1)
        parsed.append([a.replace('BE-KvN','BE–KvN'),b,int(c.replace(',','')),int(d),float(amount.replace(',','')),unit,5e-4])
    table('matched_field_designs',['workload','selection','physical_qubits','accepted_shots','complete_task_time','time_unit','physical_error_probability'],parsed)
    # Preserve the full text of shortened numerical paragraphs as provenance.
    ids=[472,473,482,500,502,505,574,584,612,669,670,671,672,691,965,1003,1004]
    table('original_numerical_context',['source','original_paragraph','text'],[['SI v30',i,text(ps[i])] for i in ids])

for name in ['hardware_and_search','matched_field_designs','original_numerical_context']:
    cache=ROOT/'input'/(name+'.json')
    if name in tables:cache.write_text(json.dumps(tables[name],ensure_ascii=False,indent=2)+'\n')
    elif cache.exists():tables[name]=json.loads(cache.read_text())

(ROOT/'tables.json').write_text(json.dumps(tables,ensure_ascii=False,indent=2)+'\n')
report={'correlated_bias_pct':full_bias,'factorized_bias_pct':factor_bias,'tables':{k:len(v)-1 for k,v in tables.items()},'no_new_flow_or_resource_optimization':True}
(ROOT/'checks.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
