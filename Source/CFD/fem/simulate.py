"""Transient cylinder-wake flow on a mirrored Taylor--Hood P2/P1 mesh.

Start from a steady field plus a divergence-free projected perturbation.
Use IMEX Euler startup, then BDF2/AB2 with a fixed steady-state convection
Jacobian implicit and the nonlinear remainder explicit. Length, velocity
and time are nondimensionalized by D, U and D/U.
"""
import os

for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'

from pathlib import Path
import sys, json, time, argparse

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu
from scipy.spatial import cKDTree
from skfem import MeshTri, Basis, ElementVector, ElementTriP2, ElementTriP1, asm

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[2]))
from Source.CFD.inputs import steady_input
from Source.CFD.fem.steady import (
    diffusion, divergence, mass, convection, convection_jacobian,
    divergence_squared, gradient_squared,
)


# Streamfunction presets: amplitude, Gaussian centre (cx,cy), and width.
# amp is not the maximum velocity perturbation.
CASES = {
    'train': (.02, 2., 0., .7),
    'test_a': (.035, 2.7, .25, .8),
    'test_b': (-.015, 1.8, -.3, .6),
    'steady': (0., 2., 0., .7),
    'holdout_d': (.01, 2.3, -.2, .65),
    'holdout_e': (-.03, 2.5, .15, .75),
    'holdout_f': (.05, 3., 0., .9),
}


def validate_inputs(dt, end, case, reynolds):
    """Reject invalid run settings before creating output or assembling matrices."""
    for name, value in (('dt', dt), ('end', end), ('reynolds', reynolds)):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f'--{name} must be a finite number greater than zero.')
    if end < dt:
        raise ValueError('--end must be at least --dt to run one full time step.')
    if case not in CASES:
        raise ValueError(f'Unknown case {case!r}; choose from: {", ".join(CASES)}.')


def run(mesh_name, dt, end, case, reynolds=100., input_path=None):
    # Inputs: mesh label, time step, final time, perturbation case and Reynolds number.
    validate_inputs(dt, end, case, reynolds)
    initial = steady_input(mesh_name, reynolds, input_path)
    out = ROOT / 'cfd' / f'Re{reynolds:g}_{mesh_name}_dt{dt:g}_{case}'
    out.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    a = np.load(initial)

    # p: half-domain vertex coordinates; t: triangle connectivity (not pressure/time).
    p = a['mesh_points']
    t = a['mesh_triangles']
    on = np.isclose(p[1], 0)
    idx = np.flatnonzero(~on)

    # Reflect off-axis vertices; mapped gives their indices in the full mesh.
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

    # FEM spaces: vector P2 velocity and scalar P1 pressure.
    vb = Basis(mesh, ElementVector(ElementTriP2()), intorder=5)
    pb = Basis(mesh, ElementTriP1(), intorder=5)
    # Numbers of velocity and pressure degrees of freedom.
    nv = vb.N
    np_ = pb.N

    # Match each full-domain velocity node to its reflected half-domain node.
    half = Basis(MeshTri(p, t), ElementVector(ElementTriP2()), intorder=5)
    loc = vb.doflocs[:, ::2]
    dist, jj = cKDTree(half.doflocs[:, ::2].T).query(
        np.vstack([loc[0], abs(loc[1])]).T
    )
    assert dist.max() < 1e-9

    # Steady velocity coefficients: interleaved ux, uy; uy changes sign under reflection.
    u0 = np.zeros(nv)
    u0[::2] = a['velocity'][::2][jj]
    u0[1::2] = a['velocity'][1::2][jj] * np.where(loc[1] < 0, -1, 1)

    # Mass, diffusion, divergence and zero pressure-pressure blocks; A excludes viscosity.
    M = asm(mass, vb)
    A = asm(diffusion, vb)
    B = asm(divergence, vb, pb)
    Z = sp.csr_matrix((np_, np_))

    # Prescribed velocity: inlet/top/bottom (1,0), cylinder (0,0).
    # Outlet traction is natural: nu*du/dn - p*n=0. Pressure DOFs remain free.
    fixed = vb.get_dofs(['inlet', 'farfield', 'cylinder']).all()
    free = np.setdiff1d(np.arange(nv + np_), fixed)
    bc = u0[fixed].copy()

    amp, cx, cy, width = CASES[case]
    x, y = loc

    # C2 cutoff chi(x) vanishes for x<=0.7, including the cylinder.
    # z is a clipped cutoff coordinate; dcut is d(chi)/dx.
    z = np.clip((x - .7) / .5, 0, 1)
    cut = z**3 * (10 - 15*z + 6*z*z)
    dcut = np.where((x > .7) & (x < 1.2), 30*z*z*(1-z)**2 / .5, 0.)

    # Gaussian: g(x,y) = exp(-((x-cx)^2 + (y-cy)^2) / width^2).
    g = np.exp(-((x-cx)**2 + (y-cy)**2) / width**2)
    # Velocity perturbation from psi=amp*cut*g: (dpsi/dy, -dpsi/dx).
    perturb = np.zeros(nv)
    perturb[::2] = amp * cut * g * (-2*(y-cy) / width**2)
    perturb[1::2] = -amp * g * (dcut - 2*(x-cx) / width**2 * cut)
    perturb[fixed] = 0

    # Mass-matrix projection onto B*delta_u=0; lp stores the free-DOF LU factors.
    Kp = sp.bmat([[M, -B.T], [B, Z]], format='csc')
    lp = splu(Kp[free][:, free])
    # vp contains [projected velocity perturbation, constraint multiplier].
    vp = np.zeros(nv + np_)
    rhs = np.r_[M @ perturb, np.zeros(np_)]
    vp[free] = lp.solve(rhs[free])
    u = u0 + vp[:nv]
    del lp, Kp

    # Fixed convection Jacobian at the steady field.
    N0 = asm(convection_jacobian, vb, base=vb.interpolate(u0))

    def factor(alpha):
        # alpha=1 for Euler startup and 3/2 for BDF2; viscosity is 1/Re.
        K = sp.bmat([
            [alpha*M/dt + (1/reynolds)*A + N0, -B.T],
            [B, Z],
        ], format='csc')
        return splu(K[free][:, free]), K[free][:, fixed] @ bc

    # Time-step LU factors and prescribed-boundary contributions to the free rows.
    lu1, kbc1 = factor(1.)
    lu, kbc = factor(1.5)

    # Observation grid, separate from the FEM mesh; mask excludes the cylinder.
    gx = np.linspace(-1, 7, 128)
    gy = np.linspace(-2, 2, 64)
    xx, yy = np.meshgrid(gx, gy)
    mask = xx*xx + yy*yy >= .25
    points = np.vstack([xx[mask], yy[mask]])
    # Fixed FEM-to-grid evaluation matrix, built in blocks to bound point-location memory.
    probes = sp.vstack([
        vb.probes(points[:, j:j+128]) for j in range(0, points.shape[1], 128)
    ]).tocsr()
    # Vector Basis probes are component-major in EACH block. Restore that order.
    splits = [min(128, points.shape[1]-j) for j in range(0, points.shape[1], 128)]

    def snap(v):
        vals = probes @ v
        res = []
        offset = 0
        for size in splits:
            res.append(vals[offset:offset+2*size].reshape(2, size))
            offset += 2*size
        arr = np.zeros((2, 64, 128), np.float32)
        arr[:, mask] = np.concatenate(res, axis=1)
        return arr

    # Current/previous velocities and explicit convection remainders C(u)-N0*u.
    uold = u.copy()
    conv = asm(convection, vb, base=vb.interpolate(u)) - N0 @ u
    cold = conv.copy()
    # Save approximately every 0.24 time units; ts/fields hold sampled times/velocities.
    stride = max(1, round(.24/dt))
    steps = round(end/dt)
    ts = [0.]
    fields = [snap(u)]
    diag = []

    # Run settings and initial diagnostics saved with the results, not solver inputs.
    meta = {
        'reynolds': reynolds,
        'viscosity': 1/reynolds,
        'domain': [-10, 20, -10, 10],
        'dt': dt,
        'saved_interval': max(1, round(.24/dt))*dt,
        'end': end,
        'case': case,
        'perturbation': CASES[case],
        'velocity_dofs': int(nv),
        'pressure_dofs': int(np_),
        'triangles': int(mesh.nelements),
        'scheme': 'BDF2/AB2: steady convection Jacobian implicit, nonlinear remainder explicit; IMEX Euler first step',
        'initial_weak_divergence_l2': float(np.linalg.norm(B @ u)),
        'initial_perturbation_mass_norm': float(np.sqrt(vp[:nv] @ (M @ vp[:nv]))),
        'boundaries': {
            'inlet_top_bottom': 'u=(1,0)',
            'cylinder': 'u=0',
            'outlet': 'nu*du/dn-p*n=0',
        },
    }
    (out / 'config.json').write_text(json.dumps(meta, indent=2) + '\n')
    print('START', out.name, nv, np_, round(time.monotonic()-start, 2), flush=True)

    for k in range(1, steps+1):
        # Euler startup or BDF2/AB2 velocity RHS; continuity RHS is zero.
        rhs_v = (M @ u/dt - conv) if k == 1 else (M @ (2*u - .5*uold)/dt - 2*conv + cold)
        rhs = np.r_[rhs_v, np.zeros(np_)]
        # Coupled state: [velocity coefficients, pressure coefficients].
        state = np.zeros(nv + np_)
        state[fixed] = bc
        state[free] = (lu1 if k == 1 else lu).solve(
            rhs[free] - (kbc1 if k == 1 else kbc)
        )
        unew = state[:nv]
        if not np.all(np.isfinite(unew)) or np.max(abs(unew)) > 10:
            peak = int(np.argmax(np.hypot(unew[::2], unew[1::2])))
            np.savez_compressed(out / 'failed_state.npz', velocity=unew, coordinates=loc, time=k*dt)
            raise RuntimeError(f'Unstable CFD at step {k}; peak coordinate {loc[:,peak]}')

        uold, u = u, unew
        cold, conv = conv, asm(convection, vb, base=vb.interpolate(u)) - N0 @ u
        if k % stride == 0 or k == steps:
            ts.append(k*dt)
            fields.append(snap(u))
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

    np.savez_compressed(
        out / 'snapshots.npz', times=ts, fields=np.array(fields),
        grid_x=gx, grid_y=gy, fluid_mask=mask, steady=snap(u0),
    )
    np.savez_compressed(out / 'final_state.npz', velocity=u, previous_velocity=uold)
    (out / 'diagnostics.json').write_text(json.dumps(diag, indent=2) + '\n')
    # Add runtime/completion to the settings record after successful integration.
    meta['elapsed_seconds'] = time.monotonic()-start
    meta['completed'] = True
    (out / 'summary.json').write_text(json.dumps(meta, indent=2) + '\n')
    print('DONE', out.name, meta['elapsed_seconds'], flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--mesh', default='coarse')
    p.add_argument('--dt', type=float, default=.02)
    p.add_argument('--end', type=float, default=80.)
    p.add_argument('--case', choices=CASES, default='train')
    p.add_argument('--reynolds', type=float, default=100.)
    p.add_argument('--input', type=Path, help='Explicit steady field; otherwise use the local/Zenodo dataset')
    a = p.parse_args()
    try:
        validate_inputs(a.dt, a.end, a.case, a.reynolds)
    except ValueError as error:
        p.error(str(error))
    run(a.mesh, a.dt, a.end, a.case, a.reynolds, a.input)
