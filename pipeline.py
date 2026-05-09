import json
import os
import subprocess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--start", type=int, default=0)
parser.add_argument("--end", type=int, default=None)

args = parser.parse_args()

scenario_file = "scenario.json"
default_env = "a living room"

with open(scenario_file, 'r') as f:
    scenarios = json.load(f)

scenarios = scenarios[args.start:args.end]

print(f"Processing scenarios {args.start} to {args.end}")

success_count = 0
failed_scenarios = []

for s in scenarios:
    name = s.get('name')
    desc = s.get('desc')
    env = s.get('env', default_env)

    if not name or not desc:
        continue

    print(f"\nProcessing: {name}")

    obj_diff_path = os.path.join("output", name, "obj_diff.npz")

    if not os.path.exists(obj_diff_path):
        print(f"Skipping: {obj_diff_path} not found")
        failed_scenarios.append(name)
        continue

    final_output = os.path.join("output", name, "radarllm_6d.npy")

    # skip already completed
    if os.path.exists(final_output):
        print(f"Already exists: {name}")
        continue

    cmd = ["python", "run.py", "-o", desc, "-e", env, "-n", name]

    result = subprocess.run(cmd)

    if result.returncode == 0:
        print(f"Success: {name}")
        success_count += 1
    else:
        print(f"FAILED: {name}")
        failed_scenarios.append(name)

print("=" * 30)
print(f"Success: {success_count}")
print(f"Failed: {len(failed_scenarios)}")