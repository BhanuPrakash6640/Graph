import glob
import json
import os
import yaml

runs = glob.glob('experiments/results/exp1_gnn/*')
completed = {}
for r in runs:
    met_file = os.path.join(r, 'metrics.json')
    cfg_file = os.path.join(r, 'config.yaml')
    if os.path.exists(met_file) and os.path.exists(cfg_file):
        try:
            with open(met_file) as f:
                met = json.load(f)
            with open(cfg_file) as f:
                cfg = yaml.safe_load(f)
            key = f"{cfg['model']}|{cfg['split']}|{cfg['feature_set']}|{cfg['seed']}"
            completed[key] = {
                'pr': round(met.get('test_at_opt', {}).get('pr_auc', 0), 4),
                'f1': round(met.get('test_at_opt', {}).get('f1', 0), 4),
                'thr': round(met.get('best_threshold', 0.5), 3),
                'ep': met.get('best_epoch', 0)
            }
        except Exception:
            pass

with open('completed_runs.json', 'w') as f:
    json.dump(completed, f)
print(f"Wrote {len(completed)} entries to completed_runs.json")

