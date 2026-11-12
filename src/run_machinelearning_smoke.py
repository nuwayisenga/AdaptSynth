"""Run a one-epoch pipeline check. Outputs are preliminary, not research results.

Run from the project: .venv/bin/python -u src/run_machinelearning_smoke.py
"""
import os,time,traceback
os.environ['MPLBACKEND']='Agg'
os.environ.setdefault('MPLCONFIGDIR', '/tmp/adaptsynth-matplotlib')
import nbformat
from pathlib import Path
root=Path(__file__).resolve().parents[1]
os.chdir(root)
nb=nbformat.read(root/'machinelearning_pipeline.ipynb',as_version=4)
ns={'display':lambda x: print(x.to_string() if hasattr(x,'to_string') else x,flush=True)}
import matplotlib.pyplot as plt
plt.show=lambda: plt.close('all')
start=time.perf_counter()
for i,c in enumerate(nb.cells):
 if c.cell_type!='code':continue
 print(f'\nSTART cell {i} elapsed={time.perf_counter()-start:.1f}s',flush=True)
 t=time.perf_counter()
 try:
  exec(compile(c.source,f'notebook_cell_{i}','exec'),ns)
  if i==2:
   ns.update(EPOCHS=1,MAX_ROWS_PER_CLASS=100,RFE_STEP=0.2,OUTPUT_DIR=root/'results_machinelearning_smoke')
   import torch
   torch.set_num_threads(2)
  if i==12:
   ns['scores'].to_csv(root/'results_machinelearning_smoke/feature_scores.csv')
   ns['vote_table'].to_csv(root/'results_machinelearning_smoke/voting_table.csv')
  if i==17:print(ns['results'].to_string(index=False),flush=True)
 except Exception:
  traceback.print_exc();raise
 print(f'DONE cell {i} seconds={time.perf_counter()-t:.1f}',flush=True)
print('COMPLETE',time.perf_counter()-start,flush=True)
