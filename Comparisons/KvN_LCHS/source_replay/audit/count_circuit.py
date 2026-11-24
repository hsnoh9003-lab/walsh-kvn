"""Audit the published KvN-LCHS example at pinned QuCF source revision.

Small block-encoding circuits are reconstructed gate by gate. Large QSP,
selector-power and repetition layers are counted algebraically, without
allocating a 216-million-gate list. No full 25-qubit evolution is claimed.
"""
from __future__ import annotations
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
from dataclasses import dataclass,replace
from collections import Counter
import json,math,re,hashlib,csv
import numpy as np
import h5py
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[4]))
from Source.data_locations import shared_data

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parent/'QuCF'
KVN=shared_data('RE_common', 'LCHS_source_inputs')
OUT=ROOT/'results';OUT.mkdir(exist_ok=True)

@dataclass(frozen=True)
class Gate:
    name:str
    t:int
    p:tuple=()
    z:tuple=()
    angles:tuple=()
    inverse:bool=False

def g(name,t,p=(),z=(),angles=()):
    p,z=tuple(sorted(set(p))),tuple(sorted(set(z)))
    assert not set(p)&set(z) and t not in p+z,(name,t,p,z)
    return Gate(name,t,p,z,tuple(float(x) for x in angles))

def controlled(seq,p=(),z=()):
    return [replace(q,p=tuple(sorted(set(q.p)|set(p))),z=tuple(sorted(set(q.z)|set(z)))) for q in seq]

def adjoint(seq):return [replace(q,inverse=not q.inverse) for q in seq[::-1]]

def increment(ts):return [g('X',t,tuple(q for q in ts if q<t)) for t in sorted(ts,reverse=True)]

def qft(ts):
    out=[]
    for i in range(len(ts)-1,-1,-1):
        out.append(g('H',ts[i]))
        for j in range(1,i+1):out.append(g('P',ts[i],[ts[i-j]],angles=[math.pi/(2**j)]))
    for i in range(len(ts)//2):
        a,b=ts[i],ts[-i-1]
        out += [g('X',b,[a]),g('X',a,[b]),g('X',b,[a])]
    return out

def add_fixed(ts,carry,value):
    qs=list(ts)+[carry];n=len(qs);out=qft(qs)
    for i in range(n-1,-1,-1):
        num=n-i
        for j in range(num):
            if (value>>j)&1:out.append(g('P',qs[i],angles=[2*math.pi/2**(num-j)]))
    return out+adjoint(qft(qs))

def comparator(ts,carries,value):
    a=add_fixed(ts,carries[0],value)
    return adjoint(a)+[g('X',carries[1],[carries[0]])]+a

class SmallBuilder:
    def __init__(self):
        self.constants={}
        txt=(KVN/'t2/kvn.oracle').read_text()
        for line in txt.split('CONSTANTS',1)[1].split('END_CONSTANTS')[0].splitlines():
            a=line.split('//')[0].split()
            if len(a)==2:self.constants[a[0]]=float(a[1])
        self.regs={'rx':list(range(7)),'ax':[7,8],'acom':[9],'ae':[10,11,12]}
        self.blocks={}
        self.phis={}
        for key in ('D','Da'):
            with h5py.File(KVN/f'BE-3/qsvt/angles_{key}_nx7_sin_eps15.hdf5') as f:
                self.phis[key]=f['results/phis'][()]

    def value(self,x):return self.constants[x[1:-1]] if x.startswith('<') else float(x)

    def spec(self,tokens,i):
        out=[];other=[]
        n=1
        if re.fullmatch(r'\d+',tokens[i]):n=int(tokens[i]);i+=1
        for _ in range(n):
            word=tokens[i];i+=1
            if '[' in word:
                name,idx=word[:-1].split('[');idx=int(idx)
                picks=[self.regs[name][idx]]
            else:
                name=word;mask=int(self.value(tokens[i]));i+=1
                if mask<0:mask+=2**len(self.regs[name])
                picks=[q for j,q in enumerate(self.regs[name]) if mask>>j&1]
            out+=picks;other += [q for q in self.regs[name] if q not in picks]
        return sorted(out),sorted(other),i

    def controls(self,tokens,i):
        p=[];z=[]
        while i<len(tokens) and tokens[i] not in ('do','end_gate','end_circuit'):
            kind=tokens[i];i+=1
            a,b,i=self.spec(tokens,i)
            if kind=='control':p+=a
            elif kind=='ocontrol':z+=a
            elif kind=='control_e':p+=a;z+=b
            elif kind=='ocontrol_e':z+=a;p+=b
            else:raise ValueError((kind,tokens))
        return p,z

    def sin(self,target,rx):
        a0=self.constants['alpha_x0_D'];a=self.constants['alpha_x1_D']
        return [g('Ry',target,angles=[2*a0])]+[g('Ry',target,[q],angles=[2*a/2**(len(rx)-1-i)]) for i,q in enumerate(rx)]+[g('X',target)]

    def qsvt(self,name,target,bequbits):
        rx=bequbits[:-1];anc=bequbits[-1];be=self.sin(anc,rx);phis=self.phis[name]
        out=[g('H',target),g('X',target,z=[anc]),g('Rz',target,angles=[2*phis[0]]),g('X',target,z=[anc])]
        for i in range(1,len(phis)):
            out+=be if i%2 else adjoint(be)
            if len(phis)%2==0 and i==len(phis)-1:out.append(g('Z',target))
            out += [g('X',target,z=[anc]),g('Rz',target,angles=[2*phis[i]]),g('X',target,z=[anc])]
        return out+[g('H',target)]

    def parse(self,path):
        seq=[];stack=[]
        for line in path.read_text().splitlines():
            line=line.split('//')[0].strip()
            if not line:continue
            t=line.split()
            if t[0]=='with':stack.append(self.controls(t,1));continue
            if t[0]=='end_with':stack.pop();continue
            assert t[0] in ('gate','igate'),line
            inverse=t[0]=='igate';name=t[1];i=2
            if name=='QSVT':
                label=t[i];i+=1;tar,_,i=self.spec(t,i)
                assert t[i]=='Usin_D';i+=1
                be,_,i=self.spec(t,i);block=self.qsvt(label,tar[0],be)
            else:
                ts,_,i=self.spec(t,i)
                if name in ('Rz','Ry','Rc'):
                    n=2 if name=='Rc' else 1
                    ang=[self.value(x) for x in t[i:i+n]];i+=n
                    block=[g(name,q,angles=ang) for q in ts]
                elif name in ('H','X','Y','Z'):block=[g(name,q) for q in ts]
                elif name=='incrementor':block=increment(ts)
                elif name=='decrementor':block=adjoint(increment(ts))
                elif name=='ComparatorFixed':
                    value=int(self.value(t[i]));i+=1
                    carry,_,i=self.spec(t,i);block=comparator(ts,carry,value)
                else:raise ValueError(line)
            pp,zz=self.controls(t,i)
            for p,z in stack:pp+=p;zz+=z
            seq+=controlled(adjoint(block) if inverse else block,pp,zz)
        assert not stack
        return seq

    def be(self,which):
        p=KVN/'BE-3'
        out=self.parse(p/f'UD/add_oracles/BE_{which}_OF_forward.oracle')
        out+=self.parse(p/('OH/OH_Aa.oracle' if which=='a' else 'OH/OH_Bk.oracle'))
        out+=self.parse(p/f'UD/add_oracles/BE_{which}_OM.oracle')
        out+=self.parse(p/f'UD/add_oracles/BE_{which}_OF_backward.oracle')
        return out

def matrix(gate):
    name=gate.name;a=gate.angles
    if name=='X':m=np.array([[0,1],[1,0]],complex)
    elif name=='Y':m=np.array([[0,-1j],[1j,0]])
    elif name=='Z':m=np.diag([1.,-1.]).astype(complex)
    elif name=='H':m=np.array([[1,1],[1,-1]],complex)/np.sqrt(2)
    elif name=='Rz':m=np.diag([np.exp(-.5j*a[0]),np.exp(.5j*a[0])])
    elif name=='Ry':m=np.array([[np.cos(a[0]/2),-np.sin(a[0]/2)],[np.sin(a[0]/2),np.cos(a[0]/2)]],complex)
    elif name=='P':m=np.diag([1,np.exp(1j*a[0])])
    elif name=='P0':m=np.diag([np.exp(1j*a[0]),1])
    elif name=='Rc':m=matrix(g('Ry',0,angles=[a[1]]))@matrix(g('Rz',0,angles=[a[0]]))
    else:raise ValueError(name)
    return m.conj().T if gate.inverse else m

def apply(seq,state):
    state=np.array(state,complex,copy=True);idx=np.arange(len(state))
    for gate in seq:
        mask=((idx>>gate.t)&1)==0
        for q in gate.p:mask&=((idx>>q)&1)==1
        for q in gate.z:mask&=((idx>>q)&1)==0
        a=idx[mask];b=a+(1<<gate.t);m=matrix(gate)
        olda=state[a].copy();oldb=state[b].copy()
        state[a]=m[0,0]*olda+m[0,1]*oldb;state[b]=m[1,0]*olda+m[1,1]*oldb
    return state

# Histograms retain gate type, positive/negative control counts and angle values.
def hist(seq):return Counter((q.name,len(q.p),len(q.z),q.angles) for q in seq)
def plus_control(h,p=0,z=0):return Counter({(n,a+p,b+z,ang):v for (n,a,b,ang),v in h.items()})
def scale(h,k):return Counter({a:k*b for a,b in h.items()})

def qsp_hist(be,phis,outer=0):
    # qsp_ham_opt2: each qubitization W contains controlled BE and BE dagger.
    # W/Wdagger have the same control-count histogram; external controls are disjoint.
    w=plus_control(be,z=1)+plus_control(be,p=1)
    w+=hist([g('H',13),g('P0',13,z=list(range(7,13)),angles=[math.pi]),g('P0',13,angles=[math.pi]),g('H',13)])
    out=scale(plus_control(w,z=1),len(phis)-1)
    # Inverse W is controlled on one rather than zero; correct that half.
    half=(len(phis)-1)//2
    out=scale(plus_control(w,z=1),half)+scale(plus_control(w,p=1),half)
    out+=Counter({('H',0,0,()):4+4*half})
    for a in phis[:-1]:out[('Rz',0,0,(float(a),))]+=2
    out[('Rz',0,0,(float(phis[-1]),))]+=1
    out[('P0',0,0,(-math.pi/2,))]+=half
    out[('P',0,0,(math.pi/2,))]+=half
    return plus_control(out,p=outer)

def full_hist(builder,nt):
    ba=builder.be('a')+[g('X',12),g('Z',12),g('X',12)]
    bk=builder.be('k');bm=bk+[g('X',12),g('Z',12),g('X',12)]
    qsp={}
    for label,be,filename,ctrl in [('a',ba,'QSP_t4741p0_eps12.hdf5',0),('k',bk,'QSP_t6180p0_eps12.hdf5',1),('kmax',bm,'QSP_t3924p2_eps12.hdf5',0)]:
        with h5py.File(KVN/'t2/qsp'/filename) as f:phis=f['angles/QSP-ham'][()]
        qsp[label]=qsp_hist(hist(be),phis,ctrl)
    step=scale(qsp['a'],2)+qsp['kmax']+scale(qsp['k'],127)
    # PREP and inverse PREP each appear 2*N_AA+1=7 times.
    prep=[g('H',q) for q in range(7)]
    kmax=builder.constants['k_max']
    for i in range(128):
        k=(-1+2*i/127)*kmax;angle=2*np.arccos(1/np.sqrt(1+k*k))
        prep.append(g('Ry',7,[q for q in range(7) if i>>q&1],[q for q in range(7) if not i>>q&1],[angle]))
    prep.append(g('X',8,z=[7]))
    reflections=hist([g('Z',8),g('X',8,z=list(range(8))),g('Z',8,z=list(range(8))),g('X',8,z=list(range(8)))])
    overhead=scale(hist(prep),14)+scale(reflections,6)+Counter({('H',0,0,()):7})
    return scale(step,nt)+overhead,{'step':step,'overhead':overhead,**qsp}

def run():
    b=SmallBuilder();checks={}
    for label in ('a','k'):
        gates=b.be(label)
        filename='Ba_OUTPUT.hdf5' if label=='a' else 'Bk_OUTPUT.hdf5'
        with h5py.File(KVN/'BE-3/case-SIN-LCU'/filename) as f:
            # QuCF stores the outputs for successive input columns contiguously.
            expected=int(f['basic/N_gates'][()]);ref=(f['matrix/real'][()]+1j*f['matrix/imag'][()]).reshape(128,128).T
        assert len(gates)==expected,(label,len(gates),expected)
        start=np.zeros((8192,128),complex);start[:128]=np.eye(128)
        actual=apply(gates,start)[:128]
        error=float(np.max(abs(actual-ref)))
        assert error<2e-11,(label,error)
        checks[label]={'STMC':len(gates),'matrix_max_absolute_error':error}
        print('BE check',label,checks[label],flush=True)
    rows=[]
    for name,nt in [('dt1',10),('t2',20)]:
        counts,parts=full_hist(b,nt)
        with h5py.File(KVN/name/'kvn_OUTPUT.hdf5') as f:
            expected=int(f['basic/N_gates'][()]);a=f['states/output-zero-anc-amplitudes-0'][()]
            success=float(np.sum(a['real']**2+a['imag']**2))
        assert sum(counts.values())==expected,(nt,sum(counts.values()),expected)
        raw=Counter()
        for (kind,p,z,ang),number in counts.items():raw[kind]+=number
        row=dict(case=name,physical_simulation_time=nt*.1,time_steps=nt,logical_qubits=25,STMC_total=sum(counts.values()),source_STMC=expected,raw_gate_counts=dict(raw),success_probability_archived=success,full_circuit_rerun=False)
        rows.append(row);print(row,flush=True)
        with (OUT/f'{name}_stmc_histogram.csv').open('w') as f:
            wr=csv.writer(f);wr.writerow(['gate','positive_controls','negative_controls','angles','multiplicity'])
            for key,v in sorted(counts.items()):wr.writerow([*key[:3],json.dumps(key[3]),v])
    result={'source_revision':'f34f89c','block_encoding_checks':checks,'cases':rows}
    (OUT/'source_count_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    return result

if __name__=='__main__':run()
