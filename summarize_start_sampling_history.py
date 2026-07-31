import csv
import json
import sys

with open(sys.argv[1]) as f:
    rows = list(csv.DictReader(f))

r = rows[-1]
print(f"episodes:       {len(rows)}")
print(f"hard success:   {float(r['hard_success_ema']):.3f}")
print(f"random success: {float(r['random_success_ema']):.3f}")
print(f"next p(hard):   {float(r['hard_start_probability_next']):.3f}")
print("\nHard-pose probabilities:")

probs = json.loads(r["pose_sampling_probability_json"])
success = json.loads(r["pose_success_ema_json"])

for name, probability in sorted(probs.items(), key=lambda x: x[1], reverse=True):
    print(f"{probability:6.2%}  success={success[name]:.3f}  {name}")