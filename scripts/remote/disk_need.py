"""Estimate the disk the rest of the pipeline needs, in GB.

Called by pipeline.sh before the sweep. Sized against remaining work: a resumed run
already has the corpus and some arms on disk, and costing those again would make the
guard refuse work that fits.
"""

import json
import os
import sys

TOTAL_RUNS = 39
GB_PER_RUN = 1.3  # best.pt + final.pt, with milestones disabled for the sweep
GB_CORPUS = 20
GB_REPRO = 10

data_dir, results, out_dir = sys.argv[1:4]

done = 0
if os.path.exists(results):
    arms = json.load(open(results))["arms"]
    done = sum(1 for a in arms if a["status"] in ("completed", "diverged"))

need = (TOTAL_RUNS - done) * GB_PER_RUN
need += 0 if os.path.exists(os.path.join(data_dir, "meta.json")) else GB_CORPUS
need += 0 if os.path.exists(os.path.join(out_dir, "gpt2-124m-repro", "final.pt")) else GB_REPRO
print(int(need) + 5)  # headroom
