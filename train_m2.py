"""
train_m2.py
===========
Train the M2 "slow/fast (tremor) split" embedding using the EXISTING training
pipeline 
"""

from __future__ import annotations

import train_verification as tv
import m2_model


def main():
    m2_model.register()                          # m2/m2a/m2b -> M2 encoder
    parser = tv.build_argparser()                # reuse the exact same CLI
    m2_model.extend_model_choices(parser)        # allow --model m2/m2a/m2b
    tv.main(parser.parse_args())


if __name__ == "__main__":
    main()
