# GoExplore

## Installation

Tested using Python version 3.8.3.

First clone the repository, then run

```console
sudo apt-get install libgl1
pip install -r requirements.txt
```

## Example usage

To run experiment "Test" with 10000 frames on Pong, run the following command:

```console
python code/run_experiments.py --exp-name Test --games Pong --frames 10000
```

This will produce a folder ```experiments/<timestamp>\_Test```. This folder contains 4 files:
 - a .trajectory file, which contains the pickled trajectory to the best cell.
 - a .json file, which contains metadata pertaining to the experiment, such as the highscore; the trajectory length; the archive size; the duration; and the number of frames processed.
 - a .svg file, which plots the how the algorithm progresses over time.
 - a .data file, which contains the pickled raw data used to produce the plots.

Run the help command for more details:

```console
python code/run_experiments.py -h
```
