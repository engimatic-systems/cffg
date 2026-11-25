"""Run the registered two-level T-maze cue-and-commit study, retaining every trial."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import jax
import numpy as np

from cffg.hierarchy import Hierarchy
from cffg.runtime_types import RunStatus


def transition(flip):
    return np.array([[1-flip, flip], [flip, 1-flip]])


def cue_update(predicted, likelihood):
    mass = predicted * likelihood
    return mass / mass.sum()


def maze_run(config, seed):
    rng = np.random.Generator(np.random.PCG64(seed))
    context = int(rng.integers(2))
    k = config['child_ticks_per_period']
    flip = config['parent_flip_probability']
    p = config['cue_accuracy']
    slow_transition = transition(flip)
    average_transition = transition(flip / k)
    emission = np.eye(2)
    cue_laws = transition(1-p)
    uniform = np.array(config['first_parent_prior'])
    child = np.repeat(cue_laws.T[:, :, None], 2, axis=2)
    model = Hierarchy.from_tables(uniform, slow_transition, uniform, child, emission, thickness=k)
    flat = uniform.copy()
    reference = uniform.copy()
    trials = []
    max_error = 0.
    for period in range(config['periods']):
        if period:
            if rng.random() < flip:
                context = 1-context
            reference = slow_transition @ reference
        for tick in range(k):
            cue = context if rng.random() < p else 1-context
            likelihood = cue_laws[cue]
            reference = cue_update(reference, likelihood)
            flat = cue_update(average_transition @ flat, likelihood)
            result = model.step(cue)
            row = dict(period=period, tick=tick, true_arm=context, cue=cue,
                       flat_arm=int(np.argmax(likelihood)),
                       memory_flat_arm=int(np.argmax(flat)), status=result.status.value)
            trials.append(row)
            if result.status != RunStatus.CONVERGED:
                return dict(seed=seed, trials=trials, passed=False, artifact_hash=model.artifact_hash)
            posterior = np.asarray(result.parent)
            error = float(np.max(abs(posterior-reference)))
            max_error = max(error,max_error)
            row.update(parent=posterior.tolist(), hierarchy_arm=int(np.argmax(posterior)),
                       flat_with_memory=flat.tolist(), joint_energy=result.free_energy)
    metrics = {field:float(np.mean([t[field] == t['true_arm'] for t in trials]))
               for field in ['hierarchy_arm','flat_arm','memory_flat_arm']}
    metrics['last_parent_mass'] = float(np.mean([t['parent'][t['true_arm']] for t in trials if t['tick']==k-1]))
    return dict(seed=seed, artifact_hash=model.artifact_hash, passed=bool(max_error <= 1024*np.finfo(float).eps),
                max_abs_posterior_error=max_error, metrics=metrics, trials=trials)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if not jax.config.x64_enabled:
        parser.error('Set JAX_ENABLE_X64=1 for the registered float64 reference comparison')
    protocol = Path(__file__).with_name('drift_protocol.json')
    config = json.loads(protocol.read_text())
    root = Path(__file__).resolve().parents[2]
    revision = subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    paths = [*sorted((root/'src/cffg').glob('hierarchy*.py')),Path(__file__).resolve(),protocol]
    hashes = {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    report = dict(revision=revision, source_sha256=hashes, protocol=config, cells=[])
    for seed in config['seeds']:
        report['cells'].append(maze_run(config,seed))
    passed = all(run['passed'] for run in report['cells'])
    if passed:
        scores = {key:float(np.mean([r['metrics'][key] for r in report['cells']]))
                  for key in report['cells'][0]['metrics']}
        passed = scores['last_parent_mass']>=.75 and scores['hierarchy_arm']-scores['flat_arm']>=.06
        report['scores'] = scores
    report['passed'] = passed
    args.output.write_text(json.dumps(report,allow_nan=False,separators=(',',':'))+'\n')
    if not passed:
        raise SystemExit('Study did not pass. See the recorded trial statuses, discrepancies and scores.')
    print(json.dumps(report['scores'],sort_keys=True))


if __name__ == '__main__':
    main()
