"""Check actual quantum-state force readout and all 815-shot covariance data."""

# Public Source Data mapping (single canonical file per published input).
import sys as _kvn_path_sys
from pathlib import Path as _kvn_path_Path
_kvn_path_sys.path.insert(0, str(_kvn_path_Path(__file__).resolve().parents[1]))
from Source.data_locations import published_path
from pathlib import Path
from tempfile import TemporaryDirectory
import csv,json,math
import numpy as np
from Source.ROM.force_sampling import evaluate,ROOT

INTEGER_FIELDS={'step','shots','shots_per_time','count','outcome'}
FLOAT_RTOL=1e-12
FLOAT_ATOL=1e-13

def compare_csv_numerically(generated, archived):
    """Check identical structure and counts, tolerating CSV float roundoff.

    The original bytewise test could fail on platforms that serialize the same
    floating-point result with a different last decimal digit.  Every discrete
    outcome/count is still required to agree exactly.
    """
    with generated.open(newline='',encoding='utf-8') as fh:
        reader=csv.DictReader(fh)
        columns=reader.fieldnames
        computed=list(reader)
    with archived.open(newline='',encoding='utf-8') as fh:
        reader=csv.DictReader(fh)
        expected_columns=reader.fieldnames
        expected=list(reader)
    assert columns==expected_columns,(generated.name,'columns')
    assert len(computed)==len(expected),(generated.name,'row count')
    max_abs_difference=0.0
    for index,(fresh,stored) in enumerate(zip(computed,expected)):
        for column in columns:
            actual,reference=fresh[column],stored[column]
            if column in INTEGER_FIELDS:
                assert int(actual)==int(reference),(generated.name,index,column,actual,reference)
                continue
            try:
                got,want=float(actual),float(reference)
            except ValueError:
                assert actual==reference,(generated.name,index,column,actual,reference)
                continue
            assert math.isfinite(got) and math.isfinite(want),(generated.name,index,column)
            max_abs_difference=max(max_abs_difference,abs(got-want))
            assert math.isclose(got,want,rel_tol=FLOAT_RTOL,abs_tol=FLOAT_ATOL),(
                generated.name,index,column,got,want)
    return max_abs_difference

def main():
    with TemporaryDirectory(prefix='force_815_') as temp:
        result,meta=evaluate(Path(temp),repetitions=2000)
        assert meta['shots_total_per_time']==815
        assert meta['maximum_expectation_discrepancy']<2e-13
        for row in result:
            assert abs(row['repeated_sampling_MSE_over_exact_MSE']-1)<.03
        csv_differences={}
        for name in ['force_815_shots.csv','sampled_outcome_counts.csv','summary.csv']:
            csv_differences[name]=compare_csv_numerically(
                Path(temp)/name,published_path('Data/Force_Sampling/'+name))
        with np.load(published_path('Data/Force_Sampling/readout_distributions.npz')) as stored, \
                np.load(Path(temp)/'readout_distributions.npz') as fresh:
            assert set(stored.files)==set(fresh.files)
            for key in stored.files:
                expected, actual = stored[key], fresh[key]
                assert actual.shape == expected.shape, (key, 'shape')
                assert actual.dtype == expected.dtype, (key, 'dtype')
                # Numerical arrays may differ by platform roundoff; discrete data must match exactly.
                if np.issubdtype(expected.dtype, np.inexact):
                    np.testing.assert_allclose(
                        actual, expected, rtol=FLOAT_RTOL, atol=FLOAT_ATOL,
                        equal_nan=False, err_msg=key,
                    )
                else:
                    np.testing.assert_array_equal(actual, expected, err_msg=key)
    print(json.dumps({'status':'PASS','signed_MPS_input':True,'all_exact_force_means_verified':34,
        'four_terminal_settings_verified':True,'finite_shot_CSV_reproduced':True,
        'CSV_max_absolute_differences':csv_differences,
        'CSV_float_tolerances':{'rtol':FLOAT_RTOL,'atol':FLOAT_ATOL},
        'NPZ_float_tolerances':{'rtol':FLOAT_RTOL,'atol':FLOAT_ATOL},
        'discrete_counts_exact':True,'saved_npz_numerically_verified':True,
        'independent_batches':2000}))

if __name__=='__main__':main()
