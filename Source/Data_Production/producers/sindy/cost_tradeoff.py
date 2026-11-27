"""Secondary cost/accuracy analysis, added after the accuracy-only experiment.

This uses an analytical Walsh rotation-support upper bound on a common
Cartesian tensor grid. It is not a resource re-estimate of the polar wake
circuits in the manuscript.
"""
from pathlib import Path
from itertools import product
from math import comb
import csv,json
import numpy as np
import run_experiment as experiment

ROOT=Path(__file__).resolve().parent

def support_bound(model,bits_per_coordinate):
    """Sum of bounds for one polynomial pair layer along each coordinate.

    A monomial with powers p_j can contribute only Walsh subsets with at most
    p_j address bits from coordinate j. Distinct monomial supports are unioned
    before counting; coordinate i has n-1 pair-address bits and the others n.
    """
    counts=[];naive_cnot=[]
    for axis in range(3):
        widths=[bits_per_coordinate]*3;widths[axis]-=1
        weights=set()
        for powers in model['p'][model['C'][:,axis]!=0]:
            weights.update(product(*(range(min(int(p),m)+1) for p,m in zip(powers,widths))))
        number=0;cnot=0
        for weight in weights:
            multiplicity=int(np.prod([comb(m,w) for m,w in zip(widths,weight)]))
            number+=multiplicity;cnot+=2*sum(weight)*multiplicity
        counts.append(number);naive_cnot.append(cnot)
    return counts,naive_cnot

def main():
    protocol=dict(
        timing='Secondary analysis added after the primary accuracy-only experiment, following the request to assess synthesis cost.',
        data='Same CFD projected states, fitted coefficients, ridge, thresholds and blocked validation scores as the primary experiment.',
        grid='Common illustrative Cartesian grid: 64 cells (6 address bits) per retained POD coordinate; sensitivity at 16,32,128,256 cells.',
        cost='Sum over three coordinates of the Walsh support upper bound for one interior polynomial adjacent-pair layer per coordinate.',
        definition='For equation i, set m_i=n-1 and m_j=n otherwise. Union all integer weight tuples w with 0<=w_j<=p_j for at least one active monomial p. Sum product_j binomial(m_j,w_j) over that union.',
        comparison='Bound uses the actual active monomial structure of each full-data refit. It allows coefficient cancellations to reduce realized counts.',
        exclusions='No angular coordinate transform, boundary correction, address shifts, splitting multiplicities, step count, input preparation, rotation precision, shots, error correction, or factory supply. This is not the reported polar circuit or its T-count/runtime.',
        decision='Preserve the original accuracy-only winner; report the nondominated cost/error frontier. No scalar penalty or accuracy tolerance is chosen to force a degree.',
        error_scope='Validation error is against fixed-basis CFD projections. The manuscript 1% quantum-versus-ROM target is a different error and is not used as the ROM selection tolerance.')
    (ROOT/'cost_protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    data=np.load(ROOT/'inputs/baseline_inputs.npz')
    candidates=list(csv.DictReader((ROOT/'results/validation_candidates.csv').open()))
    rows=[];sensitivity=[]
    for candidate in candidates:
        degree=int(candidate['library_degree']);threshold=float(candidate['threshold'])
        model=experiment.fit(data['training_coordinates'],data['training_times'],degree,threshold)
        counts,cnots=support_bound(model,6)
        row=dict(library_degree=degree,threshold=threshold,active_degree=model['active_degree'],
            active_terms=model['active_terms'],active_above_quadratic=model['active_above_quadratic'],
            validation_error_percent=float(candidate['mean_validation_field_error_percent']),
            rotation_support_bound=sum(counts),bound_equation_1=counts[0],bound_equation_2=counts[1],bound_equation_3=counts[2],
            separate_parity_network_CNOT_bound=sum(cnots))
        rows.append(row)
        for n in [4,5,6,7,8]:
            bound,_=support_bound(model,n)
            sensitivity.append(dict(library_degree=degree,threshold=threshold,cells_per_coordinate=2**n,
                bits_per_coordinate=n,rotation_support_bound=sum(bound),validation_error_percent=row['validation_error_percent']))
    for row in rows:
        row['pareto_nondominated']=not any(
            other['rotation_support_bound']<=row['rotation_support_bound'] and
            other['validation_error_percent']<=row['validation_error_percent'] and
            (other['rotation_support_bound']<row['rotation_support_bound'] or
             other['validation_error_percent']<row['validation_error_percent']) for other in rows)
    experiment.write_csv(ROOT/'results/cost_accuracy_candidates.csv',rows)
    experiment.write_csv(ROOT/'results/cost_grid_sensitivity.csv',sensitivity)
    selected=[]
    for degree in experiment.DEGREES:
        best=min((r for r in rows if r['library_degree']==degree),key=lambda r:(r['validation_error_percent'],r['threshold']))
        selected.append(best)
    frontier=sorted([r for r in rows if r['pareto_nondominated']],key=lambda r:(r['rotation_support_bound'],r['validation_error_percent'],r['library_degree']))
    experiment.write_csv(ROOT/'results/cost_accuracy_best_by_degree.csv',selected)
    experiment.write_csv(ROOT/'results/cost_accuracy_frontier.csv',frontier)
    original=next(r for r in rows if r['library_degree']==2 and r['threshold']==0)
    report=dict(protocol=protocol,best_by_library_degree=selected,frontier=frontier,
        deployed_quadratic=original,
        note='The cost proxy supports assessing an accuracy/cost compromise; it does not show that the deployed quadratic degree was learned or that a new hardware-optimal model has been found.')
    (ROOT/'results/cost_summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(best_by_library_degree=selected,frontier=frontier,deployed_quadratic=original),indent=2))

if __name__=='__main__':main()
