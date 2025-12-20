"""
Copyright © 2022 Laboratoire Adhesion et Inflammation, Authored by Remy Torro.
"""

import argparse
import datetime
import os
import sys
from art import tprint
from celldetective.signals import analyze_signals
import pandas as pd
from celldetective import logger
from celldetective.log_manager import PositionLogger, setup_global_logging

setup_global_logging()

tprint("Signals")

parser = argparse.ArgumentParser(description="Classify and regress the signals based on the provided model.",
								formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-p',"--position", required=True, help="Path to the position")
parser.add_argument('-m',"--model", required=True, help="Path to the model")
parser.add_argument("--mode", default="target", help="Cell population of interest")
parser.add_argument("--use_gpu", default="True", choices=["True","False"],help="use GPU")

args = parser.parse_args()
process_arguments = vars(args)
pos = str(process_arguments['position'])
model = str(process_arguments['model'])
mode = str(process_arguments['mode'])
use_gpu = process_arguments['use_gpu']
if use_gpu=='True' or use_gpu=='true' or use_gpu=='1':
	use_gpu = True
else:
	use_gpu = False

column_labels = {'track': "TRACK_ID", 'time': 'FRAME', 'x': 'POSITION_X', 'y': 'POSITION_Y'}

if mode.lower()=="target" or mode.lower()=="targets":
	table_name = "trajectories_targets.csv"

elif mode.lower()=="effector" or mode.lower()=="effectors":
	table_name = "trajectories_effectors.csv"
else:
	table_name = f"trajectories_{mode}.csv"



# Load trajectories, add centroid if not in trajectory
trajectories_path = os.path.join(pos, 'output', 'tables', table_name)
if os.path.exists(trajectories_path):
	trajectories = pd.read_csv(trajectories_path)
else:
	logger.error('The trajectories table could not be found. Abort.')
	exit(1)

with PositionLogger(pos):
	logger.info(f'Starting signal analysis with model: {model}, mode: {mode}')
	
	trajectories = analyze_signals(trajectories.copy(), model, interpolate_na=True, selected_signals=None, column_labels = column_labels, plot_outcome=True, output_dir=os.path.join(pos,'output', ''))
	trajectories = trajectories.sort_values(by=[column_labels['track'], column_labels['time']])
	trajectories.to_csv(os.path.join(pos, 'output', 'tables', table_name), index=False)


