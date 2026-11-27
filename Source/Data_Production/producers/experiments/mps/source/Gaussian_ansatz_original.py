"""Gaussian initial-state preparation copied in *structure* from wake_POD_KvN_PQC.

A picture block = Ry(all), forward adjacent CX, Ry(all), reverse-direction
adjacent CX. Fit REAL normalized amplitudes, not probabilities. Analytic
Jacobian matches the original wake compare_preparation.py construction.
"""
from pathlib import Path
import json
import numpy as np
from scipy.optimize import least_squares

REFERENCE_DIR = Path(__file__).resolve().parents[1] / "references"


class WakeAnsatz:
    def __init__(self, n_qubits, picture_blocks):
        self.n = int(n_qubits)
        self.stages = 2 * int(picture_blocks)
        if self.n < 1 or self.stages < 2:
            raise ValueError("At least one position qubit and one picture block are needed")
        self.size = 1 << self.n
        self.nparams = self.n * self.stages
        indices = np.arange(self.size)
        self.pairs = [(indices[(indices & (1 << q)) == 0],
                       indices[(indices & (1 << q)) == 0] | (1 << q)) for q in range(self.n)]
        self.perms = {(c,t): indices ^ (((indices >> c) & 1) << t)
                      for c in range(self.n) for t in range(self.n) if c != t}
        self.cache = None

    def edges(self, stage):
        return [(q,q+1) if stage % 2 == 0 else (q+1,q) for q in range(self.n-1)]

    def evaluate(self, angles, *, jacobian=False):
        angles = np.asarray(angles,dtype=float).reshape(self.nparams)
        if self.cache is not None and np.array_equal(self.cache[0],angles):
            state,jac = self.cache[1:]
            return (state,jac) if jacobian else state
        state = np.zeros(self.size)
        state[0] = 1.0
        jac = np.zeros((self.size,self.nparams))
        for stage in range(self.stages):
            for q,(i0,i1) in enumerate(self.pairs):
                k=stage*self.n+q
                co,si = np.cos(angles[k]/2),np.sin(angles[k]/2)
                a,b=state[i0].copy(),state[i1].copy()
                ja,jb=jac[i0].copy(),jac[i1].copy()
                state[i0],state[i1]=co*a-si*b,si*a+co*b
                jac[i0],jac[i1]=co*ja-si*jb,si*ja+co*jb
                jac[i0,k]+=(-si*a-co*b)/2
                jac[i1,k]+=(co*a-si*b)/2
            for edge in self.edges(stage):
                perm=self.perms[edge]
                state,jac=state[perm],jac[perm]
        self.cache=(angles.copy(),state.copy(),jac.copy())
        return (state,jac) if jacobian else state

    def warm_start(self,target):
        state=np.asarray(target,dtype=float).copy()
        for edge in reversed(self.edges(self.stages-1)):
            state=state[self.perms[edge]]
        prob=state**2
        angles=np.zeros(self.nparams)
        for q,(i0,i1) in enumerate(self.pairs):
            angles[(self.stages-1)*self.n+q]=2*np.arctan2(np.sqrt(prob[i1].sum()),np.sqrt(prob[i0].sum()))
        return angles

    def circuit(self,angles):
        from qiskit import QuantumCircuit
        angles=np.asarray(angles,dtype=float).reshape(self.stages,self.n)
        q=QuantumCircuit(self.n,name=f"WakeGaussianPQC_{self.stages//2}blocks")
        for stage in range(self.stages):
            for bit in range(self.n):
                q.ry(float(angles[stage,bit]),bit)
            for c,t in self.edges(stage):
                q.cx(c,t)
        return q


def fit_initial_pqc(target_probability, n_qubits, *, blocks=2, starts=3,
                    max_nfev=600, seed=20261002, reuse_wake_cubic7=True):
    """Fit to the supplied *actual* target, or reuse exact-matching wake fit."""
    p=np.asarray(target_probability,dtype=float).ravel()
    if p.size != 1 << n_qubits or np.any(p<0) or not np.isclose(p.sum(),1.0,atol=1e-9):
        raise ValueError("Invalid normalized probability for the requested position register")
    target=np.sqrt(p)
    ans=WakeAnsatz(n_qubits,blocks)
    preset=REFERENCE_DIR/"wake_cubic7_preparation.npz"
    angles_json=REFERENCE_DIR/"wake_cubic7_gaussian_angles.json"
    if reuse_wake_cubic7 and n_qubits==7 and blocks==2 and preset.exists() and angles_json.exists():
        with np.load(preset) as data:
            preset_target=data["target_probability"]
        if np.allclose(p,preset_target,rtol=1e-11,atol=1e-13):
            angles=np.asarray(json.loads(angles_json.read_text(encoding="utf-8"))["angles_radians"]).ravel()
            actual=ans.evaluate(angles)
            return dict(ansatz=ans,angles=angles,amplitude=actual,source="wake_cubic7_exact_target_fitted",
                        trials=[],**_fit_stats(actual,target))
    rng=np.random.default_rng(seed+int(n_qubits)*100+int(blocks))
    warm=ans.warm_start(target)
    trial_guesses=[warm,warm+rng.normal(0,.1,ans.nparams),warm+rng.normal(0,.5,ans.nparams)]
    while len(trial_guesses)<starts:
        trial_guesses.append(rng.uniform(-np.pi,np.pi,ans.nparams))
    best=None; records=[]
    for idx,guess in enumerate(trial_guesses[:starts]):
        solution=least_squares(lambda a: ans.evaluate(a)-target, guess,
                               jac=lambda a: ans.evaluate(a,jacobian=True)[1],
                               method="trf",max_nfev=max_nfev,
                               ftol=1e-11,xtol=1e-11,gtol=1e-11)
        amplitude=ans.evaluate(solution.x).copy()
        loss=np.linalg.norm(amplitude-target)
        records.append(dict(start=idx,nfev=int(solution.nfev),l2_error=float(loss),
                            converged=bool(solution.success)))
        if best is None or loss<best[0]:
            best=(loss,solution.x.copy(),amplitude)
    return dict(ansatz=ans,angles=best[1],amplitude=best[2],source="fitted_wake_ansatz",
                trials=records,**_fit_stats(best[2],target))


def _fit_stats(actual,target):
    return dict(amplitude_l2_error=float(np.linalg.norm(actual-target)),
                fidelity=float(abs(np.vdot(actual,target))**2),
                probability_l1_pct=float(100*np.sum(abs(abs(actual)**2-abs(target)**2))),
                max_probability_error=float(np.max(abs(abs(actual)**2-abs(target)**2))))


def export_initial_pqc(fit_result, output):
    from qiskit import qpy
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    circuit=fit_result["ansatz"].circuit(fit_result["angles"])
    with (out/"INITIAL_GAUSSIAN_PQC.qpy").open("wb") as f:qpy.dump(circuit,f)
    return circuit
