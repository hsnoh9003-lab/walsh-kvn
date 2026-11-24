"""Replay the archived Re100 flow and its drag/lift coefficients.

Use mirrored Taylor--Hood P2/P1 elements, IMEX Euler startup and BDF2/AB2.
Integrate pressure and viscous stresses on the cylinder at every time step.
Lengths, velocities and times are scaled by D, U and D/U.
"""
import os

for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'

from pathlib import Path
import sys, json, time, argparse, csv, hashlib

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu
from scipy.spatial import cKDTree
from skfem import (
    MeshTri, Basis, FacetBasis, Functional,
    ElementVector, ElementTriP2, ElementTriP1, asm,
)
from skfem.helpers import grad

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'source'))
from solve_steady import (
    diffusion, divergence, mass, convection, convection_jacobian,
    divergence_squared, gradient_squared,
)


# Force forms use the outward FLUID normal n, which points into the cylinder.
# w['pressure']: pressure at quadrature points; w.n[0/1]: n_x/n_y.
# Pressure drag: F_x,p = integral_Gamma p*n_x ds; asm performs the integral.
@Functional
def pressure_x(w):
    return w['pressure'] * w.n[0]


# Pressure lift: F_y,p = integral_Gamma p*n_y ds.
@Functional
def pressure_y(w):
    return w['pressure'] * w.n[1]


# Viscous drag: F_x,nu = -integral_Gamma [nu*(grad(u)+grad(u)^T)*n]_x ds.
@Functional
def viscous_x(w):
    # g[i,j] = du_i/dx_j; base is the velocity field at quadrature points.
    g = grad(w['base'])
    # traction: viscous stress times the fluid normal; nu=1/Re.
    traction = w['nu'] * np.einsum(
        'ij...,j...->i...', g + np.swapaxes(g, 0, 1), w.n
    )
    # Force on the cylinder is opposite to the fluid-domain traction.
    return -traction[0]


# Viscous lift: F_y,nu = -integral_Gamma [nu*(grad(u)+grad(u)^T)*n]_y ds.
@Functional
def viscous_y(w):
    # g: velocity gradient; swapping its first two axes gives grad(u)^T.
    g = grad(w['base'])
    # traction: viscous stress times the fluid normal; index 1 selects y.
    traction = w['nu'] * np.einsum(
        'ij...,j...->i...', g + np.swapaxes(g, 0, 1), w.n
    )
    return -traction[1]


def run(mesh_name, dt, end, case, out, input_path):
    # Inputs: mesh/case labels, time step, final time, output directory and FEM archive.
    # This replay is restricted to the recorded Re100/coarse/train/dt=0.02 case.
    assert mesh_name == 'coarse' and case == 'train'
    assert abs(dt - 0.02) < 1e-14 and end >= 0
    assert abs(end / dt - round(end / dt)) < 1e-9
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    a = np.load(input_path)

    # p: half-domain mesh coordinates; t: triangle connectivity, not pressure/time.
    p = a['mesh_points']
    t = a['mesh_triangles']
    # on identifies symmetry-axis vertices; idx selects vertices to reflect.
    on = np.isclose(p[1], 0)
    idx = np.flatnonzero(~on)
    # mapped: vertex indices for the reflected lower-half triangles.
    mapped = np.arange(p.shape[1])
    mapped[idx] = p.shape[1] + np.arange(len(idx))
    mesh = MeshTri(
        np.column_stack([p, p[:, idx] * np.array([[1], [-1]])]),
        np.column_stack([t, mapped[t]]),
    )
    mesh = mesh.with_boundaries({
        'inlet': lambda x: np.isclose(x[0], -10),
        'outlet': lambda x: np.isclose(x[0], 20),
        'farfield': lambda x: np.isclose(abs(x[1]), 10),
        'cylinder': lambda x: np.linalg.norm(x, axis=0) < .500001,
    })

    # vb/pb: vector P2 velocity and scalar P1 pressure spaces on the full domain.
    vb = Basis(mesh, ElementVector(ElementTriP2()), intorder=5)
    pb = Basis(mesh, ElementTriP1(), intorder=5)
    # nv/np_: numbers of velocity/pressure degrees of freedom.
    nv = vb.N
    np_ = pb.N
    # half: archived velocity space; loc: full-domain velocity-node coordinates.
    half = Basis(MeshTri(p, t), ElementVector(ElementTriP2()), intorder=5)
    loc = vb.doflocs[:, ::2]
    # jj: matching half-domain node indices; dist: matching errors.
    dist, jj = cKDTree(half.doflocs[:, ::2].T).query(
        np.vstack([loc[0], abs(loc[1])]).T
    )
    assert dist.max() < 1e-9

    # u0: steady velocity coefficients, interleaved ux/uy; ux is even, uy is odd in y.
    u0 = np.zeros(nv)
    u0[::2] = a['velocity'][::2][jj]
    u0[1::2] = a['velocity'][1::2][jj] * np.where(loc[1] < 0, -1, 1)
    # M/A/B: mass, diffusion (without nu), divergence; Z: zero pressure block.
    M = asm(mass, vb)
    A = asm(diffusion, vb)
    B = asm(divergence, vb, pb)
    Z = sp.csr_matrix((np_, np_))
    assert nv == 37640 and np_ == 4753 and mesh.nelements == 9314

    # fbv/fbp evaluate velocity/pressure on the full cylinder boundary Gamma.
    # Body force: F = integral_Gamma [p*n - nu*(grad(u)+grad(u)^T)*n] ds.
    fbv = FacetBasis(
        mesh, ElementVector(ElementTriP2()), facets=mesh.boundaries['cylinder'], intorder=5
    )
    fbp = FacetBasis(
        mesh, ElementTriP1(), facets=mesh.boundaries['cylinder'], intorder=5
    )

    def force_components(velocity, pressure=None):
        # kw: velocity evaluated on the boundary and nu=1/Re=0.01.
        kw = {'base': fbv.interpolate(velocity), 'nu': 0.01}
        # C_D=F_x/(rho*U^2*D/2), C_L=F_y/(rho*U^2*D/2); rho=U=D=1.
        # cdv/clv: viscous drag/lift coefficients; factor 2 is normalization.
        cdv = float(2*asm(viscous_x, fbv, **kw))
        clv = float(2*asm(viscous_y, fbv, **kw))
        # The perturbed t=0 velocity has no pressure solution yet; total forces stay unset.
        if pressure is None:
            return dict(
                C_D=None, C_L=None, C_D_pressure=None, C_L_pressure=None,
                C_D_viscous=cdv, C_L_viscous=clv, pressure_valid=False,
            )
        # pf: pressure at boundary quadrature points; cdp/clp: pressure coefficients.
        pf = fbp.interpolate(pressure)
        # C_D,p = 2*integral_Gamma p*n_x ds; C_L,p = 2*integral_Gamma p*n_y ds.
        cdp = float(2*asm(pressure_x, fbv, pressure=pf))
        clp = float(2*asm(pressure_y, fbv, pressure=pf))
        # Total coefficients: C_D=cdp+cdv, C_L=clp+clv.
        # pressure_valid records pressure availability, not a convergence test.
        return dict(
            C_D=cdp+cdv, C_L=clp+clv, C_D_pressure=cdp, C_L_pressure=clp,
            C_D_viscous=cdv, C_L_viscous=clv, pressure_valid=True,
        )

    # pidx/pdist: matching half-domain pressure nodes and their coordinate errors.
    pdist, pidx = cKDTree(p.T).query(
        np.vstack([pb.doflocs[0], abs(pb.doflocs[1])]).T
    )
    assert pdist.max() < 1e-9
    # p0: steady pressure, reflected evenly; steady_force checks the archived base state.
    p0 = a['pressure'][pidx]
    steady_force = force_components(u0, p0)
    assert abs(steady_force['C_D'] - 1.1382409242337448) < 1e-10
    assert abs(steady_force['C_L']) < 1e-10
    (out / 'steady_force.json').write_text(json.dumps(steady_force, indent=2) + '\n')
    print('STEADY_FORCE', json.dumps(steady_force), flush=True)

    # Velocity BCs: inlet/top/bottom (1,0), cylinder (0,0), inherited from u0.
    # Outlet: nu*du/dn - p*n=0 (natural). Pressure degrees of freedom remain free.
    # fixed/free: prescribed velocity indices and free indices of [velocity, pressure].
    fixed = vb.get_dofs(['inlet', 'farfield', 'cylinder']).all()
    free = np.setdiff1d(np.arange(nv + np_), fixed)
    # bc: prescribed velocity coefficients.
    bc = u0[fixed].copy()

    # Streamfunction presets: amplitude, Gaussian centre (cx,cy), and width.
    # The archived presets are retained, but the assertions above permit only train.
    cases = {
        'train': (.02, 2., 0., .7),
        'test_a': (.035, 2.7, .25, .8),
        'test_b': (-.015, 1.8, -.3, .6),
        'steady': (0., 2., 0., .7),
        'holdout_d': (.01, 2.3, -.2, .65),
        'holdout_e': (-.03, 2.5, .15, .75),
        'holdout_f': (.05, 3., 0., .9),
    }
    amp, cx, cy, width = cases[case]
    # x/y: full-domain velocity-node coordinates.
    x, y = loc
    assert cases[case] == (.02, 2., 0., .7)
    # C2 cutoff chi=z^3*(10-15*z+6*z^2); z=clip((x-0.7)/0.5,0,1).
    # cut vanishes near the cylinder; dcut is d(chi)/dx.
    z = np.clip((x-.7)/.5, 0, 1)
    cut = z**3 * (10 - 15*z + 6*z*z)
    dcut = np.where((x > .7) & (x < 1.2), 30*z*z*(1-z)**2/.5, 0.)
    # Gaussian: g(x,y) = exp(-((x-cx)^2+(y-cy)^2)/width^2).
    g = np.exp(-((x-cx)**2 + (y-cy)**2)/width**2)
    # perturb = (dpsi/dy, -dpsi/dx), where psi=amp*cut*g.
    perturb = np.zeros(nv)
    perturb[::2] = amp*cut*g*(-2*(y-cy)/width**2)
    perturb[1::2] = -amp*g*(dcut - 2*(x-cx)/width**2*cut)
    perturb[fixed] = 0

    # Project the nodal perturbation onto B*delta_u=0 with zero prescribed velocity.
    # Kp: coupled mass/constraint matrix; lp: LU factorization on free indices.
    Kp = sp.bmat([[M, -B.T], [B, Z]], format='csc')
    lp = splu(Kp[free][:, free])
    # vp: [projected perturbation, constraint multiplier]; rhs: projection RHS.
    vp = np.zeros(nv + np_)
    rhs = np.r_[M @ perturb, np.zeros(np_)]
    vp[free] = lp.solve(rhs[free])
    # u: initial velocity = steady field + projected perturbation.
    u = u0 + vp[:nv]
    del lp, Kp

    # N0: fixed convection Jacobian at the steady velocity.
    N0 = asm(convection_jacobian, vb, base=vb.interpolate(u0))

    def factor(alpha):
        # K: [alpha*M/dt + nu*A + N0, -B.T; B, 0], with nu=0.01.
        # alpha=1 for Euler startup and 3/2 for BDF2.
        K = sp.bmat([
            [alpha*M/dt + .01*A + N0, -B.T],
            [B, Z],
        ], format='csc')
        return splu(K[free][:, free]), K[free][:, fixed] @ bc

    # lu1/lu: startup/BDF2 factors; kbc1/kbc: prescribed-boundary RHS corrections.
    lu1, kbc1 = factor(1.)
    lu, kbc = factor(1.5)

    # gx/gy: observation-grid axes; xx/yy: grid coordinates, separate from the FEM mesh.
    gx = np.linspace(-1, 7, 128)
    gy = np.linspace(-2, 2, 64)
    xx, yy = np.meshgrid(gx, gy)
    # mask selects fluid points; points holds their coordinates for interpolation.
    mask = xx*xx + yy*yy >= .25
    points = np.vstack([xx[mask], yy[mask]])
    # probes: fixed FEM-to-grid evaluation matrix, built in blocks to bound memory.
    probes = sp.vstack([
        vb.probes(points[:, j:j+128]) for j in range(0, points.shape[1], 128)
    ]).tocsr()
    # splits: points per block; each block stores all ux values followed by all uy values.
    splits = [min(128, points.shape[1]-j) for j in range(0, points.shape[1], 128)]

    def snap(v):
        # vals: blockwise sampled velocities; res: per-block arrays of shape (2, size).
        vals = probes @ v
        res = []
        offset = 0
        for size in splits:
            res.append(vals[offset:offset+2*size].reshape(2, size))
            offset += 2*size
        # arr: (component, y, x) snapshot; cylinder-interior entries remain zero.
        arr = np.zeros((2, 64, 128), np.float32)
        arr[:, mask] = np.concatenate(res, axis=1)
        return arr

    # uold: previous velocity; conv/cold: current/previous remainders C(u)-N0*u.
    uold = u.copy()
    conv = asm(convection, vb, base=vb.interpolate(u)) - N0 @ u
    cold = conv.copy()
    # stride: snapshot interval in steps (12 at dt=0.02); steps: total step count.
    stride = max(1, round(.25/dt))
    steps = round(end/dt)
    # ts/fields: sampled times/velocities; diag: sampled divergence/speed records.
    ts = [0.]
    fields = [snap(u)]
    diag = []

    # meta: recorded settings and diagnostics; historical metadata fields are preserved.
    meta = {
        'reynolds': 100,
        'domain': [-10, 20, -10, 10],
        'dt': dt,
        'end': end,
        'case': case,
        'perturbation': cases[case],
        'velocity_dofs': int(nv),
        'pressure_dofs': int(np_),
        'triangles': int(mesh.nelements),
        'scheme': 'BDF2/AB2: steady convection Jacobian implicit, nonlinear remainder explicit; IMEX Euler first step',
        'original_OpenFOAM_boundary_match': False,
        'new_independent_CFD': True,
        'initial_weak_divergence_l2': float(np.linalg.norm(B @ u)),
        'initial_perturbation_mass_norm': float(np.sqrt(vp[:nv] @ (M @ vp[:nv]))),
        'boundaries': {
            'inlet_top_bottom': 'u=(1,0)',
            'cylinder': 'u=0',
            'outlet': 'nu*du/dn-p*n=0',
        },
    }
    meta['force_diagnostic'] = {
        'normal': 'outward from fluid domain; into cylinder',
        'body_force_integrand': 'p*n - nu*(grad(u)+grad(u).T)*n',
        'coefficient_factor': 2.0,
        'rho': 1.0,
        'U_infinity': 1.0,
        'diameter': 1.0,
        'nu': 0.01,
        'surface_quadrature_order': 5,
        'initial_perturbed_pressure_available': False,
        't0_note': 'Only viscous force from perturbed velocity is reported; steady pressure is not substituted.',
        'original_solver_sha256': 'e993920dcd763f367f71d77515a83c88ff22b57e95ef3adb906ba01dda988194',
        'input_sha256': hashlib.sha256(Path(input_path).read_bytes()).hexdigest(),
    }
    (out / 'config.json').write_text(json.dumps(meta, indent=2) + '\n')

    # forces: per-step force records; fstream/writer stream the same records to CSV.
    # At t=0, the perturbed velocity has no computed pressure, so total C_D/C_L are unset.
    forces = [dict(step=0, time=0.0, **force_components(u, None))]
    fstream = (out / 'forces.csv').open('w', newline='')
    writer = csv.DictWriter(fstream, fieldnames=list(forces[0]))
    writer.writeheader()
    writer.writerow(forces[0])
    fstream.flush()
    # final_pressure: latest solved pressure; selected_steps: archived full-state save times.
    final_pressure = None
    selected_steps = {
        round(s/dt) for s in [29.04, 40.08, 41.52, 51.36, 70.08, 90.0] if s <= end+1e-12
    }
    print('START', out.name, nv, np_, round(time.monotonic()-start, 2), flush=True)

    for k in range(1, steps+1):
        # Let R(u)=C(u)-N0*u. Euler RHS: M*u/dt-R(u).
        # BDF2 RHS: M*(2*u-0.5*uold)/dt-2*R(u)+R(uold).
        rhs_v = (M @ u/dt - conv) if k == 1 else (M @ (2*u-.5*uold)/dt - 2*conv + cold)
        rhs = np.r_[rhs_v, np.zeros(np_)]
        # state: [velocity, pressure] coefficients; continuity RHS is zero.
        state = np.zeros(nv + np_)
        state[fixed] = bc
        state[free] = (lu1 if k == 1 else lu).solve(
            rhs[free] - (kbc1 if k == 1 else kbc)
        )
        # unew/final_pressure: velocity and pressure at the same new time k*dt.
        unew = state[:nv]
        final_pressure = state[nv:].copy()
        if not np.all(np.isfinite(unew)) or np.max(abs(unew)) > 10:
            raise RuntimeError(f'Unstable CFD at step {k}')
        uold, u = u, unew
        cold, conv = conv, asm(convection, vb, base=vb.interpolate(u)) - N0 @ u

        # row: surface-integrated C_D/C_L and their components for this time step.
        row = dict(step=k, time=k*dt, **force_components(u, final_pressure))
        forces.append(row)
        writer.writerow(row)
        if k in selected_steps:
            np.savez_compressed(
                out / f'full_state_step_{k:05d}.npz', time=k*dt,
                velocity=u, previous_velocity=uold, pressure=final_pressure,
            )
        if k % max(1, round(5/dt)) == 0:
            fstream.flush()
        if k % stride == 0 or k == steps:
            ts.append(k*dt)
            fields.append(snap(u))
            # divnorm: weak divergence residual norm; speed: peak nodal speed.
            divnorm = float(np.linalg.norm(B @ u))
            speed = float(np.max(np.hypot(u[::2], u[1::2])))
            diag.append({'step': k, 'time': k*dt, 'weak_divergence_l2': divnorm, 'max_speed': speed})
        if k % max(1, round(5/dt)) == 0:
            print('STEP', k, 'time', k*dt, 'elapsed', round(time.monotonic()-start, 1),
                  'speed', float(np.max(abs(u))), flush=True)
            np.savez_compressed(
                out / 'snapshots.npz', times=ts, fields=np.array(fields),
                grid_x=gx, grid_y=gy, fluid_mask=mask, steady=snap(u0),
            )

    # Save the sampled fields, full final FEM state, force history and diagnostics.
    np.savez_compressed(
        out / 'snapshots.npz', times=ts, fields=np.array(fields),
        grid_x=gx, grid_y=gy, fluid_mask=mask, steady=snap(u0),
    )
    fstream.flush()
    fstream.close()
    np.savez_compressed(
        out / 'final_state.npz', velocity=u, previous_velocity=uold,
        pressure=np.array([]) if final_pressure is None else final_pressure,
        mesh_points=mesh.p, mesh_triangles=mesh.t, time=end,
    )
    (out / 'forces.json').write_text(json.dumps(forces, indent=2, allow_nan=False) + '\n')
    (out / 'diagnostics.json').write_text(json.dumps(diag, indent=2) + '\n')
    meta['elapsed_seconds'] = time.monotonic()-start
    meta['completed'] = True
    (out / 'summary.json').write_text(json.dumps(meta, indent=2) + '\n')
    print('DONE', out.name, meta['elapsed_seconds'], flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--mesh', default='coarse')
    p.add_argument('--dt', type=float, default=.02)
    p.add_argument('--end', type=float, default=90.)
    p.add_argument('--case', default='train')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--input', type=Path, default=ROOT.parent / 'inputs/steady_coarse.npz')
    a = p.parse_args()
    run(a.mesh, a.dt, a.end, a.case, a.out, a.input)
