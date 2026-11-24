"""Steady symmetric Re=100 cylinder flow, solved by damped Newton iteration.

Taylor-Hood P2/P1 finite elements on the upper half-domain. The full field is
obtained by reflection. This is a new Navier-Stokes solve, not a snapshot fit.

All distances are in cylinder diameters; inlet speed and diameter are one.
The source data's full OpenFOAM outer-domain setup is not publicly specified in
the inspected release. Outer-domain choices below are explicit reconstruction
assumptions and must be assessed by domain/mesh sensitivity, not called exact
reproduction of the source discretization.
"""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key]='1'
from pathlib import Path
import argparse,json,time
import numpy as np
import triangle
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve,splu,eigs,LinearOperator
from scipy.optimize import brentq
from skfem import MeshTri,Basis,FacetBasis,ElementVector,ElementTriP2,ElementTriP1
from skfem import BilinearForm,LinearForm,Functional,asm
from skfem.helpers import dot,ddot,grad,div

ROOT=Path(__file__).resolve().parent


def save_json(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False,
        default=lambda x:x.item() if isinstance(x,np.generic) else str(x))+'\n')


def sample(basis,vector,points):
    """Bound point-location memory when transferring between refined meshes."""
    interpolator=basis.interpolator(vector)
    return np.concatenate([interpolator(points[:,j:j+256])
                           for j in range(0,points.shape[1],256)],axis=-1)


def make_mesh(xmin,xmax,height,factor):
    # Polygonal cylinder boundary; its refinement is included in grid checks.
    count=int(np.ceil(np.pi*.5/(.045*factor)))
    angles=np.linspace(np.pi,0,count+1)
    arc=np.column_stack([.5*np.cos(angles),.5*np.sin(angles)])
    polygon=np.vstack([[xmin,0],arc,[xmax,0],[xmax,height],[xmin,height]])
    segments=np.column_stack([np.arange(len(polygon)),np.roll(np.arange(len(polygon)),-1)])
    cloud=[polygon]
    # Piecewise mesh grading: fine body and near wake, coarse far field.
    specs=[(xmin,xmax,0,height,.8*factor,0),(-3,12,0,5,.32*factor,1),(-1.5,8,0,2,.15*factor,2)]
    for lo,hi,bottom,top,h,zone in specs:
        xx,yy=np.meshgrid(np.arange(max(lo,xmin)+h/2,min(hi,xmax),h),
                         np.arange(bottom+h/2,min(top,height),h))
        points=np.column_stack([xx.ravel(),yy.ravel()])
        rad=np.linalg.norm(points,axis=1)
        keep=rad>1.0
        if zone==0:keep &= ~((points[:,0]>-3)&(points[:,0]<12)&(points[:,1]<5))
        if zone==1:keep &= ~((points[:,0]>-1.5)&(points[:,0]<8)&(points[:,1]<2))
        cloud.append(points[keep])
    radius=.5+.04*factor
    while radius<1.12:
        h=min(.15,.045+(.15-.045)*(radius-.5)/.6)*factor
        theta=np.linspace(0,np.pi,int(np.ceil(np.pi*radius/h))+1)[1:-1]
        cloud.append(np.column_stack([radius*np.cos(theta),radius*np.sin(theta)]))
        radius+=h
    vertices=np.vstack(cloud)
    raw=triangle.triangulate(dict(vertices=vertices,segments=segments),f'pq28a{.5*(.8*factor)**2:.8f}Q')
    mesh=MeshTri(raw['vertices'].T,raw['triangles'].T)
    mesh=mesh.with_boundaries(dict(
        inlet=lambda x:np.isclose(x[0],xmin),
        outlet=lambda x:np.isclose(x[0],xmax),
        farfield=lambda x:np.isclose(x[1],height),
        symmetry=lambda x:np.isclose(x[1],0)&(np.abs(x[0])>=.5-1e-8),
        cylinder=lambda x:np.sqrt(x[0]**2+x[1]**2)<.500001))
    return mesh


@BilinearForm
def diffusion(u,v,w):return ddot(grad(u),grad(v))


@BilinearForm
def divergence(u,q,w):return div(u)*q


@BilinearForm
def mass(u,v,w):return dot(u,v)


@LinearForm
def convection(v,w):
    u=w['base']
    return dot(np.einsum('ij...,j...->i...',grad(u),u),v)


@BilinearForm
def convection_jacobian(du,v,w):
    u=w['base']
    variation=np.einsum('ij...,j...->i...',grad(u),du)+np.einsum('ij...,j...->i...',grad(du),u)
    return dot(variation,v)


@Functional
def divergence_squared(w):return div(w['base'])**2


@Functional
def gradient_squared(w):return ddot(grad(w['base']),grad(w['base']))


@Functional
def drag_half(w):
    g=grad(w['base'])
    stress_viscous=w['nu']*np.einsum('ij...,j...->i...',g+np.swapaxes(g,0,1),w.n)
    return w['pressure']*w.n[0]-stress_viscous[0]


def solve_case(config,output,warm=None,stability=False):
    started=time.monotonic()
    output.mkdir(parents=True,exist_ok=True)
    save_json(output/'config.json',config)
    mesh=make_mesh(config['xmin'],config['xmax'],config['height'],config['mesh_factor'])
    velocity_basis=Basis(mesh,ElementVector(ElementTriP2()),intorder=5)
    pressure_basis=Basis(mesh,ElementTriP1(),intorder=5)
    nu_dofs,np_dofs=velocity_basis.N,pressure_basis.N
    print('MESH',mesh.nvertices,mesh.nelements,'DOFS',nu_dofs,np_dofs,flush=True)
    A=asm(diffusion,velocity_basis)
    B=asm(divergence,velocity_basis,pressure_basis)
    zero=sp.csr_matrix((np_dofs,np_dofs))
    fixed=np.unique(np.r_[velocity_basis.get_dofs(['inlet','farfield','cylinder']).all(),
                          velocity_basis.get_dofs('symmetry').all(['u^2'])])
    free=np.setdiff1d(np.arange(nu_dofs+np_dofs),fixed)
    state=np.zeros(nu_dofs+np_dofs)
    state[velocity_basis.get_dofs(['inlet','farfield']).all(['u^1'])]=1
    state[velocity_basis.get_dofs('cylinder').all()]=0
    prescribed=state[fixed].copy()
    if warm is None:
        K=sp.bmat([[A,-B.T],[B,zero]],format='csr')
        state[free]=spsolve(K[free][:,free],-K[free][:,fixed]@state[fixed])
        reynolds_steps=[1,5,10,20,40,60,80,100]
    else:
        archive=np.load(warm)
        wm=MeshTri(archive['mesh_points'],archive['mesh_triangles'])
        wb=Basis(wm,ElementVector(ElementTriP2()),intorder=5)
        pb=Basis(wm,ElementTriP1(),intorder=5)
        # Warm start is optional; only points inside the previous rectangle.
        coordinates=velocity_basis.doflocs[:,::2]
        inside=(coordinates[0]>=wm.p[0].min()+1e-9)&(coordinates[0]<=wm.p[0].max()-1e-9)&(coordinates[1]<=wm.p[1].max()-1e-9)
        inside &= np.linalg.norm(coordinates,axis=0)>.50001
        state[0:nu_dofs:2]=1
        if inside.any():
            values=sample(wb,archive['velocity'],coordinates[:,inside])
            state[np.arange(0,nu_dofs,2)[inside]]=values[0]
            state[np.arange(1,nu_dofs,2)[inside]]=values[1]
        pc=pressure_basis.doflocs
        insidep=(pc[0]>=wm.p[0].min()+1e-9)&(pc[0]<=wm.p[0].max()-1e-9)&(pc[1]<=wm.p[1].max()-1e-9)&(np.linalg.norm(pc,axis=0)>.50001)
        if insidep.any():state[nu_dofs+np.flatnonzero(insidep)]=sample(pb,archive['pressure'],pc[:,insidep])
        state[fixed]=prescribed
        reynolds_steps=[100]
    history=[]
    for reynolds in reynolds_steps:
        viscosity=1/reynolds
        K=sp.bmat([[viscosity*A,-B.T],[B,zero]],format='csr')
        def residual(vector):
            conv=asm(convection,velocity_basis,base=velocity_basis.interpolate(vector[:nu_dofs]))
            return K@vector+np.r_[conv,np.zeros(np_dofs)]
        for iteration in range(45):
            res=residual(state)
            norm=float(np.linalg.norm(res[free]))
            print('NEWTON',reynolds,iteration,'residual',norm,'seconds',round(time.monotonic()-started,1),flush=True)
            history.append(dict(reynolds=reynolds,iteration=iteration,free_residual_l2=norm))
            if norm<1e-10:break
            N=asm(convection_jacobian,velocity_basis,base=velocity_basis.interpolate(state[:nu_dofs]))
            J=sp.bmat([[viscosity*A+N,-B.T],[B,zero]],format='csr')
            update=np.zeros_like(state)
            update[free]=spsolve(J[free][:,free],-res[free])
            alpha=1.
            for backtrack in range(14):
                trial=state+alpha*update
                trial[fixed]=prescribed
                newnorm=np.linalg.norm(residual(trial)[free])
                if newnorm<norm*(1-1e-4*alpha) or newnorm<1e-10:break
                alpha*=.5
            if not np.isfinite(newnorm) or newnorm>=norm:
                raise RuntimeError(f'Newton line search failed at Re={reynolds}')
            state=trial
        else:raise RuntimeError(f'Newton did not converge at Re={reynolds}')
    velocity=state[:nu_dofs]
    pressure=state[nu_dofs:]
    np.savez_compressed(output/'finite_element_solution.npz',mesh_points=mesh.p,mesh_triangles=mesh.t,
                        velocity=velocity,pressure=pressure)
    gx,gy=np.linspace(-1,7,128),np.linspace(-2,2,64)
    xx,yy=np.meshgrid(gx,gy)
    fluid=xx*xx+yy*yy>=.25
    points=np.vstack([xx[fluid],np.abs(yy[fluid])])
    values=sample(velocity_basis,velocity,points)
    values[1]*=np.sign(yy[fluid])
    grid=np.zeros((2,len(gy),len(gx)))
    grid[:,fluid]=values
    pgrid=np.zeros_like(xx)
    pgrid[fluid]=sample(pressure_basis,pressure,points)
    np.savez_compressed(output/'steady_grid.npz',velocity=grid,pressure=pgrid,grid_x=gx,grid_y=gy,fluid_mask=fluid)
    trace_x=np.linspace(.50001,min(15,config['xmax']-.1),1000)
    trace_u=sample(velocity_basis,velocity,np.vstack([trace_x,np.zeros_like(trace_x)]))[0]
    crossings=np.flatnonzero((trace_u[:-1]<0)&(trace_u[1:]>=0))
    reattachment=None
    if len(crossings):
        j=crossings[0]
        reattachment=float(brentq(lambda x:float(velocity_basis.interpolator(velocity)(np.array([[x],[0.]]))[0,0]),trace_x[j],trace_x[j+1]))
    vfield=velocity_basis.interpolate(velocity)
    divergence_l2=float(np.sqrt(asm(divergence_squared,velocity_basis,base=vfield)))
    gradient_l2=float(np.sqrt(asm(gradient_squared,velocity_basis,base=vfield)))
    fbv=FacetBasis(mesh,ElementVector(ElementTriP2()),facets=mesh.boundaries['cylinder'],intorder=5)
    fbp=FacetBasis(mesh,ElementTriP1(),facets=mesh.boundaries['cylinder'],intorder=5)
    cd=float(4*asm(drag_half,fbv,base=fbv.interpolate(velocity),pressure=fbp.interpolate(pressure),nu=.01))
    result=dict(config=config,vertices=mesh.nvertices,triangles=mesh.nelements,
                velocity_dofs=nu_dofs,pressure_dofs=np_dofs,
                final_free_residual_l2=history[-1]['free_residual_l2'],
                divergence_l2=divergence_l2,velocity_gradient_l2=gradient_l2,
                divergence_relative_to_gradient=divergence_l2/gradient_l2,
                reattachment_x=reattachment,recirculation_length=None if reattachment is None else reattachment-.5,
                drag_coefficient=cd,elapsed_seconds=time.monotonic()-started,
                steady_solver='P2/P1 Taylor-Hood, damped Newton with Reynolds continuation',
                source_outer_boundary_match_verified=False,
                boundary_conditions=dict(inlet='u=(1,0)',farfield_top='u=(1,0)',cylinder='u=(0,0)',
                   symmetry='v=0, natural du/dy=0',outlet='natural nu*du/dn-p*n=0'),
                history=history)
    if stability:
        # Antisymmetric perturbations on the half-domain: delta u_x=0 on y=0;
        # delta u_y has natural traction. This is the vortex-shedding parity.
        perturb_fixed=np.unique(np.r_[velocity_basis.get_dofs(['inlet','farfield','cylinder']).all(),
                                      velocity_basis.get_dofs('symmetry').all(['u^1'])])
        perturb_free=np.setdiff1d(np.arange(nu_dofs+np_dofs),perturb_fixed)
        N=asm(convection_jacobian,velocity_basis,base=vfield)
        J=sp.bmat([[.01*A+N,-B.T],[B,zero]],format='csr')
        M=sp.block_diag([asm(mass,velocity_basis),zero],format='csr')
        jf=J[perturb_free][:,perturb_free].tocsc()
        mf=M[perturb_free][:,perturb_free].tocsc()
        sigma=.1+.7j
        print('STABILITY factorization',len(perturb_free),flush=True)
        lu=splu((-jf-sigma*mf).astype(complex).tocsc())
        operator=LinearOperator(jf.shape,matvec=lambda x:lu.solve(mf@x),dtype=complex)
        values,vectors=eigs(operator,k=6,which='LM',tol=1e-9,maxiter=500,ncv=30,
                            v0=np.random.default_rng(27).normal(size=jf.shape[0]).astype(complex))
        eigenvalues=sigma+1/values
        records=[]
        for k,value in enumerate(eigenvalues):
            vec=vectors[:,k]
            residual=np.linalg.norm(jf@vec+value*(mf@vec))/(np.linalg.norm(jf@vec)+abs(value)*np.linalg.norm(mf@vec))
            records.append(dict(real=float(value.real),imag=float(value.imag),relative_residual=float(residual)))
        result['antisymmetric_eigenvalues']=records
        result['unstable_eigenvalue_detected']=any(x['real']>1e-4 and x['relative_residual']<1e-7 for x in records)
        print('EIGENVALUES',records,flush=True)
    result['elapsed_seconds']=time.monotonic()-started
    save_json(output/'summary.json',result)
    np.savez_compressed(output/'centerline.npz',x=trace_x,u=trace_u)
    print('DONE',json.dumps({k:v for k,v in result.items() if k!='history'},
        default=lambda x:x.item() if isinstance(x,np.generic) else str(x)),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--name',default='domain10_coarse')
    parser.add_argument('--xmin',type=float,default=-10.)
    parser.add_argument('--xmax',type=float,default=20.)
    parser.add_argument('--height',type=float,default=10.)
    parser.add_argument('--mesh-factor',type=float,default=1.)
    parser.add_argument('--warm',type=Path)
    parser.add_argument('--stability',action='store_true')
    args=parser.parse_args()
    config=dict(reynolds=100,inlet_speed=1.,diameter=1.,viscosity=.01,xmin=args.xmin,xmax=args.xmax,
                height=args.height,mesh_factor=args.mesh_factor,quadrature_order=5,
                domain_status='Explicit far-field reconstruction assumption; exact original outer boundaries unavailable.')
    solve_case(config,ROOT/'results'/args.name,args.warm,args.stability)
