import argparse
import os
from celldetective.relative_measurements import measure_pair_signals_at_position, extract_neighborhoods_from_pickles
from celldetective.utils import ConfigSectionMap, extract_experiment_channels

from pathlib import Path, PurePath

import pandas as pd

from art import tprint


tprint("Measure pairs")

parser = argparse.ArgumentParser(description="Measure features and intensities in a multichannel timeseries.",
								 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-p', "--position", required=True, help="Path to the position")

args = parser.parse_args()
process_arguments = vars(args)
pos = str(process_arguments['position'])

instruction_file = os.sep.join(['configs', "neighborhood_instructions.json"])

# Locate experiment config
parent1 = Path(pos).parent
expfolder = parent1.parent
config = PurePath(expfolder, Path("config.ini"))
assert os.path.exists(config), 'The configuration file for the experiment could not be located. Abort.'
print("Configuration file: ", config)

# from exp config fetch spatial calib, channel names
movie_prefix = ConfigSectionMap(config, "MovieSettings")["movie_prefix"]
spatial_calibration = float(ConfigSectionMap(config, "MovieSettings")["pxtoum"])
time_calibration = float(ConfigSectionMap(config, "MovieSettings")["frametomin"])
len_movie = float(ConfigSectionMap(config, "MovieSettings")["len_movie"])
channel_names, channel_indices = extract_experiment_channels(expfolder)
nbr_channels = len(channel_names)

# from tracking instructions, fetch btrack config, features, haralick, clean_traj, idea: fetch custom timeline?
instr_path = PurePath(expfolder, Path(f"{instruction_file}"))
previous_pair_table_path = pos + os.sep.join(['output', 'tables', 'trajectories_pairs.csv'])


previous_neighborhoods = []
associated_reference_population = []


neighborhoods_to_measure = extract_neighborhoods_from_pickles(pos)
all_df_pairs = []
if os.path.exists(previous_pair_table_path):
	df_0 = pd.read_csv(previous_pair_table_path)
	previous_neighborhoods = [c.replace('status_','') for c in list(df_0.columns) if c.startswith('status_neighborhood')]
	for n in previous_neighborhoods:
		associated_reference_population.append(df_0.loc[~df_0['status_'+n].isnull(),'reference_population'].values[0])
	print(f'{previous_neighborhoods=} {associated_reference_population=}')
	all_df_pairs.append(df_0)
for k,neigh_protocol in enumerate(neighborhoods_to_measure):
	if neigh_protocol['description'] not in previous_neighborhoods:
		df_pairs = measure_pair_signals_at_position(pos, neigh_protocol)
		print(f'{df_pairs=}')
		if 'REFERENCE_ID' in list(df_pairs.columns):
			all_df_pairs.append(df_pairs)
	elif neigh_protocol['description'] in previous_neighborhoods and neigh_protocol['reference'] != associated_reference_population[previous_neighborhoods.index(neigh_protocol['description'])]:
		df_pairs = measure_pair_signals_at_position(pos, neigh_protocol)
		if 'REFERENCE_ID' in list(df_pairs.columns):
			all_df_pairs.append(df_pairs)			

print(f'{len(all_df_pairs)} neighborhood measurements sets were computed...')

if len(all_df_pairs)>1:
	print('Merging...')
	df_pairs = all_df_pairs[0]
	for i in range(1,len(all_df_pairs)):
		cols = [c1 for c1,c2 in zip(list(df_pairs.columns), list(all_df_pairs[i].columns)) if c1==c2]
		df_pairs = pd.merge(df_pairs.round(decimals=6), all_df_pairs[i].round(decimals=6), how="outer", on=cols)
elif len(all_df_pairs)==1:
	df_pairs = all_df_pairs[0]

print('Writing table...')
df_pairs = df_pairs.sort_values(by=['reference_population', 'neighbor_population', 'REFERENCE_ID', 'NEIGHBOR_ID', 'FRAME'])
df_pairs.to_csv(previous_pair_table_path, index=False)
print('Done.')

