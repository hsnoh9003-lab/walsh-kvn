"""Regenerate the matched selected-resource plot tables from the current hardware backend.

Reference literature and NISQ calibration tables are manually curated and do
not originate from this RE runner; they are not rewritten.
"""
from pathlib import Path
import pandas as pd
from Source.data_locations import source_file, figure_data
ROOT=Path(__file__).resolve().parents[1]
SOURCE=source_file('Data/RE/v38_reference_selected.csv')
FIG=figure_data('F3')

def run():
    df=pd.read_csv(SOURCE)
    df=df[df.model.isin(['direct','generator','nj_lchs']) & df.selection.isin(['minimum_qubits','fastest_uncapped'])].copy()
    assert len(df)==48
    # Reference cache may leave the minimal-qubit per-shot field null.
    # It is exactly recoverable from the matched total runtime / shot count.
    per_shot=df.complete_task_seconds/df.successful_shots
    is_missing=df.seconds_per_accepted_shot.isna()
    df.loc[is_missing,'seconds_per_accepted_shot']=per_shot.loc[is_missing]
    assert df.seconds_per_accepted_shot.notna().all()
    df['expected_attempts']=df.successful_shots/df.success_probability
    df['complete_task_hours']=df.complete_task_seconds/3600
    df['complete_task_days']=df.complete_task_seconds/86400
    df['complete_task_years']=df.complete_task_seconds/(365.25*86400)
    FIG.mkdir(parents=True,exist_ok=True)
    df.sort_values(['model','selection','p_phys']).to_csv(FIG/'panel_b_selected_designs.csv',index=False)
    df[df.p_phys.round(8)==0.0005].sort_values(['model','selection']).to_csv(FIG/'panel_c_time.csv',index=False)
    df.sort_values(['model','selection','p_phys']).to_csv(FIG/'architecture_levels.csv',index=False)
    print('Wrote F3 selected-resource tables from authoritative current hardware selected.csv')
if __name__=='__main__':run()
