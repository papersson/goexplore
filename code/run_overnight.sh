#!/bin/sh
python run_experiments.py --exp_name stochaccept_10M --games Pong --frames 10000000
python run_experiments.py --exp_name roulette_5M --games Pong --frames 5000000
