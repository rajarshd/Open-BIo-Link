import json
import wandb
from tqdm import tqdm

api = wandb.Api()
sweep = api.sweep("rajarshd/pr-cbr/21j6vk1m")
all_finished_runs = {}
finished_run_ctr = 0
for run in tqdm(sweep.runs):
    if run.state == 'finished':
        run_config = run.config
        run_summary = run.summary._json_dict
        all_finished_runs.setdefault(run_config['specific_rel'], []).append({'config': run_config,
                                                                             'summary': run_summary})
        finished_run_ctr += 1

print(f"{finished_run_ctr} runs stored")
with open("sweep_21j6vk1m_finished_runs.json", 'w') as fout:
    json.dump(all_finished_runs, fout, indent=1)


def sorting_priority(run_):
    conf, summ = run_["config"], run_["summary"]
    sort_key = (summ.get('hits_10', 0.0), summ.get('mrr', 0.0), -conf.get('cheat_neighbors', 1))
    return sort_key


all_finished_runs_sorted = {}
for specific_rel, runs in all_finished_runs.items():
    all_finished_runs_sorted[specific_rel] = sorted(runs, key=sorting_priority, reverse=True)

with open("sweep_21j6vk1m_finished_runs_sorted.json", 'w') as fout:
    json.dump(all_finished_runs_sorted, fout, indent=1)


def get_clip_key(run_):
    return (run_['summary'].get('hits_10', 0.0), run_['summary'].get('mrr', 0.0))


def clip_runs(runs_, best_key):
    clipped_runs = []
    print([get_clip_key(run__) for run__ in runs_])
    for run_ in runs_:
        if get_clip_key(run_) < best_key:
            break
        clipped_runs.append(run_)
    return clipped_runs


clipped_finished_runs_sorted = {}
for specific_rel, runs in all_finished_runs_sorted.items():
    best_scores = get_clip_key(runs[0])
    print(specific_rel)
    clipped_finished_runs_sorted[specific_rel] = clip_runs(runs, best_scores)

with open("sweep_21j6vk1m_finished_runs_clipped.json", 'w') as fout:
    json.dump(clipped_finished_runs_sorted, fout, indent=1)

useful_config_keys = ["aggr_type1", "aggr_type2", "use_only_precision_scores",
                      "k_adj", "max_num_programs", "cheat_neighbors"]
per_rel_config = {}
for specific_rel, runs in clipped_finished_runs_sorted.items():
    best_run_config = runs[0]['config']
    best_config_for_rel = {c_k: best_run_config[c_k] for c_k in useful_config_keys}
    rel_name = specific_rel if int(specific_rel) < 28 else f"{int(specific_rel)-28}_inv"
    per_rel_config[rel_name] = best_config_for_rel

with open("sweep_21j6vk1m_best_per_rel_config.json", 'w') as fout:
    json.dump(per_rel_config, fout, indent=1)

# issues for: 16, 25_inv, 20_inv
