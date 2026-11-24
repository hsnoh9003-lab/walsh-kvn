"""Replay selected Fourier–Walsh matching layers from explicitly supplied inputs.

Source is the archived implementation with the previous runtime string patch
applied statically; input, model and continuum reference are now arguments.
"""

import argparse
import ctypes
import csv
import json
from pathlib import Path
import subprocess
import time

import numpy as np

HERE = Path(__file__).resolve().parent


def harmonics(axis, r, z, coefficients):
    """Exact coefficients of 1, cos(2 theta), sin(2 theta)."""
    c = coefficients
    a = c[1,0] + c[6,0]*z
    b = c[2,0] + c[8,0]*z
    d = c[1,1] + c[6,1]*z
    e = c[2,1] + c[8,1]*z
    if axis == 0:
        return r*(a+e)/2, r*(a-e)/2, r*(b+d)/2
    if axis == 1:
        return (d-b)/2, (b+d)/2, -(a-e)/2
    return (z*(c[3,2]+c[9,2]*z) + r*r*(c[4,2]+c[7,2])/2,
            np.broadcast_to(r*r*(c[4,2]-c[7,2])/2,(len(z),r.shape[-1])),
            np.broadcast_to(r*r*c[5,2]/2,(len(z),r.shape[-1])))


def rotation_tables(r, z, modes, coefficients, dt, stencil):
    nr,nt,nz = len(r),len(modes),len(z)
    shape = (nz,max(nr,nt),2)
    layers, tables = [], []

    def add(axis,distance,parity,harmonic,mode_parity,angle):
        layers.append((axis,distance,parity,harmonic,mode_parity))
        table = np.zeros(shape)
        angles = np.broadcast_to(angle,(nz,nt if axis == 1 else nr))
        width = angles.shape[1]
        table[:,:width,0] = np.cos(angles)
        table[:,:width,1] = np.sin(angles)
        tables.append(table.ravel())

    for axis in (0,1,2):
        if axis == 1:
            mean,co,si = harmonics(axis,1.,z[:,None],coefficients)
            add(axis,0,0,0,0,(dt/2)*mean*modes[None,:])
            for harmonic,weight in ((1,co),(2,si)):
                for mode_parity in (0,1):
                    add(axis,0,0,harmonic,mode_parity,dt*weight*(modes[None,:]+1)/4)
            continue
        width = np.diff(r if axis == 0 else z)[0]
        differences = ((1,.5),) if stencil == 2 else ((1,2/3),(2,-1/12))
        for distance,weight in differences:
            if stencil == 2:
                rface = r[None,:] + (width/2 if axis == 0 else 0.)
                zface = z[:,None] + (width/2 if axis == 2 else 0.)
                values = harmonics(axis,rface,zface,coefficients)
            else:
                centred = harmonics(axis,r[None,:],z[:,None],coefficients)
                values = []
                for value in centred:
                    if axis == 0:
                        averaged = (value[:,:-distance]+value[:,distance:])/2
                        values.append(np.pad(averaged,((0,0),(0,distance))))
                    else:
                        averaged = (value[:-distance,:]+value[distance:,:])/2
                        values.append(np.pad(averaged,((0,distance),(0,0))))
            for parity in (0,1):
                for harmonic,value in enumerate(values):
                    for mode_parity in ((0,) if harmonic == 0 else (0,1)):
                        full_time = axis == 2 and distance == differences[-1][0] and parity == 1 and harmonic == 2 and mode_parity == 1
                        substep = dt if full_time else dt/2
                        scale = substep*weight/width/(1 if harmonic == 0 else 2)
                        add(axis,distance,parity,harmonic,mode_parity,scale*value)
    # Pack each layer without padding between active rows.
    for j,(axis,*_) in enumerate(layers):
        if nt != nr:
            table = tables[j].reshape(shape)
            width = nt if axis == 1 else nr
            tables[j] = np.pad(table[:,:width].copy().ravel(),(0,2*nz*(max(nr,nt)-width)))
    return np.array(layers,dtype=np.int32),np.array(tables,dtype=np.float64)


def kernel():
    library = HERE.parents[1] / "Build" / "KvN" / "matching.so"
    library.parent.mkdir(parents=True, exist_ok=True)
    if not library.is_file():
        subprocess.run(["g++", "-O3", "-std=c++17", "-shared", "-fPIC",
                        str(HERE / "matching.cpp"), "-o", str(library)], check=True)
    lib = ctypes.CDLL(str(library))
    step = lib.matching_step
    step.argtypes = [np.ctypeslib.ndpointer(dtype=dtype,flags="C_CONTIGUOUS")
                     for dtype in (np.complex128,np.int32,np.float64)] + [ctypes.c_int]*5
    step.restype = None
    return step


def replay(args):
    output = args.output
    output.mkdir(parents=True,exist_ok=True)
    model = args.model
    nr,nt,nz = args.grid
    if nt % 4:
        raise ValueError("The Fourier-mode matching requires Ntheta divisible by four")
    if args.steps % 66:
        raise ValueError("Steps must be divisible by 66 for the archived reference")
    dr = (args.rrange[1]-args.rrange[0])/nr
    dz = (args.zrange[1]-args.zrange[0])/nz
    r = args.rrange[0]+dr*(np.arange(nr)+.5)
    z = args.zrange[0]+dz*(np.arange(nz)+.5)
    delta = 2*np.pi/nt
    theta = delta*np.arange(nt)
    edges = (np.linspace(*args.rrange,nr+1),theta[0]-delta/2+delta*np.arange(nt+1),np.linspace(*args.zrange,nz+1))
    start,end = model.times
    start=args.start
    initial_norm=args.initial_probability_integral
    amplitude=args.prepared_amplitude
    if not args.no_save_preparation:
        np.save(output / "initial_amplitude_theta.npy",amplitude)
    # Stored modes are centred at zero by an index permutation equivalent to an MSB X.
    # Only the initial and final transforms are part of the implemented propagation.
    # Interior inverse transforms below are classical statevector diagnostics.
    state = np.ascontiguousarray(np.fft.fftshift(np.fft.fft(amplitude,axis=2,norm="ortho"),axes=2))
    del amplitude
    dt = (end-start)/args.steps
    modes = np.arange(-nt//2,nt//2)
    layers,tables = rotation_tables(r,z,modes,model.coefficients,dt,args.stencil)
    step = kernel()
    name = "early_reference.csv" if args.restart is None else "late_reference.csv"
    archive = np.genfromtxt(args.reference_csv,names=True,delimiter=",")
    expected = np.column_stack([archive[f"reference_mean_{q}"] for q in "xyz"])
    xy = (r[:,None]*np.cos(theta[None,:]),r[:,None]*np.sin(theta[None,:]))
    means,norms,r2,boundary = [],[],[],[]
    terminal = {}
    print(f"Starting {nr}x{nt}x{nz}, {len(layers)} layers, {args.steps} steps, dt={dt:.8f}",flush=True)
    tick = time.monotonic()
    for i in range(args.steps+1):
        if i % (args.steps//66) == 0:
            transformed = np.fft.ifft(np.fft.ifftshift(state,axes=2),axis=2,norm="ortho")
            p = transformed.real**2 + transformed.imag**2
            del transformed
            planar = p.sum(axis=0)
            zprob = p.sum(axis=(1,2))
            mean = np.array([(planar*xy[0]).sum(),(planar*xy[1]).sum(),z @ zprob])
            means.append(mean)
            norms.append(float(p.sum()))
            r2.append(float(planar.sum(axis=1) @ (r*r)))
            boundary.append(float(p[[0,-1]].sum()+p[1:-1][:,[0,-1]].sum()))
            if i == args.steps:
                second = np.zeros((3,3))
                for j in (0,1):
                    for k in range(j,2):
                        second[j,k] = second[k,j] = (planar*xy[j]*xy[k]).sum()
                    second[j,2] = second[2,j] = np.einsum("zrt,rt,z->",p,xy[j],z)
                second[2,2] = zprob @ (z*z)
                cov = second - mean[:,None]*mean[None,:]
                modes_field = model.modes[:,model.mask]
                norm_field = np.linalg.norm(model.base[model.mask]+expected[-1]@modes_field)
                metric = (modes_field @ modes_field.T)/(norm_field*norm_field)
                terminal = dict(outcome_mean=mean.tolist(),reference_outcome_mean=expected[-1].tolist(),
                                covariance=cov.tolist(),field_metric=metric.tolist(),probability_sum=norms[-1],
                                grid_centres=[r.tolist(),theta.tolist(),z.tolist()])
                (output / "terminal_readout.json").write_text(json.dumps(terminal,indent=2)+"\n")
            del p
        if i == args.steps:
            break
        step(state,layers,tables,nr,nt,nz,len(layers),tables.shape[1])
        if (i+1) % max(1,args.steps//8) == 0:
            print(f"step {i+1}/{args.steps}, {time.monotonic()-tick:.1f}s",flush=True)
    means = np.asarray(means)
    r2 = np.asarray(r2)
    error = 100*model.field_error(means,expected)
    summary = dict(method="r,k,z Fourier-Galerkin matching, endpoint-only QFT",stencil=args.stencil,
                   grid_r_theta_z=list(args.grid),rrange=args.rrange,zrange=args.zrange,steps=args.steps,
                   time_window=[float(start),float(end)],dt=float(dt),initial_probability_integral=initial_norm,
                   initial_field_error_percent=float(error[0]),terminal_field_error_percent=float(error[-1]),
                   peak_sampled_field_error_percent=float(error.max()),max_norm_drift=float(np.max(np.abs(np.array(norms)-1))),
                   max_sampled_outer_cell_mass=float(max(boundary)),terminal_means=means[-1].tolist(),
                   reference_terminal_means=expected[-1].tolist(),wall_seconds=time.monotonic()-tick,
                   matching_layers=int(len(layers)),substeps_per_step=int(2*len(layers)-1),
                   theta_grid="theta_j = 2*pi*j/Ntheta",fourier_mode_grid="-Ntheta/2,...,Ntheta/2-1",
                   initial_amplitude_axis_order=["z","r","theta"],qft_per_propagation=0,
                   endpoint_qft_count=2,initial_cell_quadrature_order=args.quadrature if args.restart is None else None,
                   restart_nodes_outside_center_hull=0 if args.restart is not None else None,initial_preparation="archive Cartesian cell-density pullback" if args.restart is None else "common t*=40 ODE quadrature deposition")
    (output / "summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    with (output / "trajectory.csv").open("w",newline="") as file:
        columns = ["time","field_relative_percent","walsh_mean_z_minus_r2","reference_mean_z_minus_r2",
                   *[f"{role}_mean_{axis}" for axis in "xyz" for role in ("walsh","reference")],"outer_cell_mass"]
        writer = csv.writer(file)
        writer.writerow(columns)
        for i,t in enumerate(archive["time"]):
            writer.writerow([t,error[i],means[i,2]-r2[i],archive["reference_mean_z_minus_r2"][i],
                             *np.column_stack((means[i],expected[i])).ravel(),boundary[i]])
    print(json.dumps(summary),flush=True)


