#!/bin/sh
# python run_experiments.py --exp_name intrinsic_motivation_usefulness --games MontezumaRevenge --frames 50000000 --selector random
# python run_experiments.py --exp_name Montezuma10M-im-v-cells --games MontezumaRevenge --seeds 0 --frame 10000000 --widths 8 24 --heights 8 --depths 16 --selector StochAccept
# python run_experiments.py --exp_name IntrinsicMotivation-NumCells-Pong --games Pong --seeds 0 --frame 5000000 --widths 8 24 --heights 8 --depths 16 --selector random StochAccept
# python run_experiments.py --exp_name IntrinsicMotivation-NumCells-Montezuma --games MontezumaRevenge --seeds 0 --frame 10000000 --widths 8 24 --heights 8 --depths 16 --selector random StochAccept

python run_experiments.py --exp-name IntrinsicMotivation-NumCells-Montezuma --games MontezumaRevenge --seeds 0 --frames 10000000 --width 24 --heights 8 --depths 16 --selector Random --max-cells 200000
python run_experiments.py --exp-name IntrinsicMotivation-NumCells-Montezuma --games MontezumaRevenge --seeds 0 --frames 10000000 --width 12 --heights 8 --depths 16 --selector Random --max-cells 200000
python run_experiments.py --exp-name IntrinsicMotivation-NumCells-Montezuma --games MontezumaRevenge --seeds 0 --frames 10000000 --width 12 --heights 8 --depths 16 --selector StochasticAcceptance --max-cells 200000
python run_experiments.py --exp-name IntrinsicMotivation-NumCells-Montezuma --games MontezumaRevenge --seeds 0 --frames 10000000 --width 24 --heights 8 --depths 16 --selector StochasticAcceptance --max-cells 200000
