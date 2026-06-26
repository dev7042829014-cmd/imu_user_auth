"""
eval_m2.py
==========
Evaluate a trained M2 ("slow/fast tremor split") checkpoint using the EXISTING
evaluation harness in eval_verification.py — without modifying any original file.

It registers m2/m2a/m2b with build_model and extends the --model choices, then
hands off to eval_verification.main(). Gallery/probe protocol, EER / FAR / FRR /
rank-1 reporting and the optional score-matrix heatmap all come from the shared
harness unchanged.

Examples
--------
  python eval_m2.py --model m2a --split test \
      --checkpoint checkpoints_verification/m2a.pt \
      --data_dirs dataset --cache_dir cache_verification

  python eval_m2.py --model m2b --split test \
      --checkpoint checkpoints_verification/m2b.pt \
      --data_dirs dataset --cache_dir cache_verification \
      --plot_dir plots
"""

from __future__ import annotations

import eval_verification as ev
import m2_model


def main():
    m2_model.register()
    parser = ev.build_argparser()
    m2_model.extend_model_choices(parser)
    ev.main(parser.parse_args())


if __name__ == "__main__":
    main()
