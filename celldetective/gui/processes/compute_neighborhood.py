from multiprocessing import Process
import time
import os

from celldetective.io import locate_labels, get_position_table, get_position_pickle

from tqdm import tqdm
import numpy as np
import pandas as pd
from art import tprint

from celldetective.neighborhood import set_live_status, contact_neighborhood, compute_attention_weight, \
    compute_neighborhood_metrics, mean_neighborhood_after_event, \
    mean_neighborhood_before_event
from celldetective.utils import extract_identity_col
from scipy.spatial.distance import cdist


class NeighborhoodProcess(Process):

    def __init__(self, queue=None, process_args=None):

        super().__init__()

        self.queue = queue

        if process_args is not None:
            for key, value in process_args.items():
                setattr(self, key, value)

        self.column_labels = {'track': "TRACK_ID", 'time': 'FRAME', 'x': 'POSITION_X', 'y': 'POSITION_Y'}

        tprint("Neighborhood")

        self.sum_done = 0
        self.t0 = time.time()

    def mask_contact_neighborhood(self, setA, setB, labelsA, labelsB, distance, mode='two-pop', status=None,
                                  not_status_option=None, compute_cum_sum=True,
                                  attention_weight=True, symmetrize=True, include_dead_weight=True,
                                  column_labels={'track': "TRACK_ID", 'time': 'FRAME', 'x': 'POSITION_X',
                                                 'y': 'POSITION_Y',
                                                 'mask_id': 'class_id'}):

        if setA is not None and setB is not None:
            setA, setB, status = set_live_status(setA, setB, status, not_status_option)
        else:
            return None, None

        # Check distance option
        if not isinstance(distance, list):
            distance = [distance]

        cl = []
        for s in [setA, setB]:

            # Check whether data can be tracked
            temp_column_labels = column_labels.copy()

            id_col = extract_identity_col(s)
            temp_column_labels.update({'track': id_col})
            if id_col == 'ID':
                compute_cum_sum = False

            cl.append(temp_column_labels)

        setA = setA.loc[~setA[cl[0]['track']].isnull(), :].copy()
        setB = setB.loc[~setB[cl[1]['track']].isnull(), :].copy()

        if labelsB is None:
            labelsB = [None] * len(labelsA)

        for d in distance:
            # loop over each provided distance

            if mode == 'two-pop':
                neigh_col = f'neighborhood_2_contact_{d}_px'
            elif mode == 'self':
                neigh_col = f'neighborhood_self_contact_{d}_px'

            setA[neigh_col] = np.nan
            setA[neigh_col] = setA[neigh_col].astype(object)

            setB[neigh_col] = np.nan
            setB[neigh_col] = setB[neigh_col].astype(object)

            # Loop over each available timestep
            timeline = np.unique(
                np.concatenate([setA[cl[0]['time']].to_numpy(), setB[cl[1]['time']].to_numpy()])).astype(
                int)



            for t in tqdm(timeline):

                index_A = list(setA.loc[setA[cl[0]['time']] == t].index)
                dataA = setA.loc[setA[cl[0]['time']] == t, [cl[0]['x'], cl[0]['y'], cl[0]['track'], cl[0]['mask_id'],
                                                            status[0]]].to_numpy()
                coordinates_A = dataA[:, [0, 1]];
                ids_A = dataA[:, 2];
                mask_ids_A = dataA[:, 3];
                status_A = dataA[:, 4];

                index_B = list(setB.loc[setB[cl[1]['time']] == t].index)
                dataB = setB.loc[setB[cl[1]['time']] == t, [cl[1]['x'], cl[1]['y'], cl[1]['track'], cl[1]['mask_id'],
                                                            status[1]]].to_numpy()
                coordinates_B = dataB[:, [0, 1]];
                ids_B = dataB[:, 2];
                mask_ids_B = dataB[:, 3];
                status_B = dataB[:, 4]

                if len(coordinates_A) > 0 and len(coordinates_B) > 0:

                    # compute distance matrix
                    dist_map = cdist(coordinates_A, coordinates_B, metric="euclidean")
                    intersection_map = np.zeros_like(dist_map).astype(float)

                    # Do the mask contact computation
                    lblA = labelsA[t]
                    lblA = np.where(np.isin(lblA, mask_ids_A), lblA, 0.)

                    lblB = labelsB[t]
                    if lblB is not None:
                        lblB = np.where(np.isin(lblB, mask_ids_B), lblB, 0.)

                    contact_pairs = contact_neighborhood(lblA, labelsB=lblB, border=d, connectivity=2)

                    # Put infinite distance to all non-contact pairs (something like this)
                    plot_map = False
                    flatA = lblA.flatten()
                    if lblB is not None:
                        flatB = lblB.flatten()

                    if len(contact_pairs) > 0:
                        mask = np.ones_like(dist_map).astype(bool)

                        indices_to_keep = []
                        for cp in contact_pairs:

                            cp = np.abs(cp)
                            mask_A, mask_B = cp
                            idx_A = np.where(mask_ids_A == int(mask_A))[0][0]
                            idx_B = np.where(mask_ids_B == int(mask_B))[0][0]

                            intersection = 0
                            if lblB is not None:
                                intersection = len(flatA[(flatA == int(mask_A)) & (flatB == int(mask_B))])

                            indices_to_keep.append([idx_A, idx_B, intersection])
                            print(f'Ref cell #{ids_A[idx_A]} matched with neigh. cell #{ids_B[idx_B]}...')
                            print(f'Computed intersection: {intersection} px...')

                        if len(indices_to_keep) > 0:
                            indices_to_keep = np.array(indices_to_keep)
                            mask[indices_to_keep[:, 0], indices_to_keep[:, 1]] = False
                            if mode == 'self':
                                mask[indices_to_keep[:, 1], indices_to_keep[:, 0]] = False
                            dist_map[mask] = 1.0E06
                            intersection_map[indices_to_keep[:, 0], indices_to_keep[:, 1]] = indices_to_keep[:, 2]
                            plot_map = True
                        else:
                            dist_map[:, :] = 1.0E06
                    else:
                        dist_map[:, :] = 1.0E06

                    d_filter = 1.0E05
                    if attention_weight:
                        weights, closest_A = compute_attention_weight(dist_map, d_filter, status_A, ids_A, axis=1,
                                                                      include_dead_weight=include_dead_weight)

                    # Target centric
                    for k in range(dist_map.shape[0]):

                        col = dist_map[k, :]
                        col_inter = intersection_map[k, :]
                        col[col == 0.] = 1.0E06

                        neighs_B = np.array([ids_B[i] for i in np.where((col <= d_filter))[0]])
                        status_neigh_B = np.array([status_B[i] for i in np.where((col <= d_filter))[0]])
                        dist_B = [round(col[i], 2) for i in np.where((col <= d_filter))[0]]
                        intersect_B = [round(col_inter[i], 2) for i in np.where((col <= d_filter))[0]]

                        if len(dist_B) > 0:
                            closest_B_cell = neighs_B[np.argmin(dist_B)]

                        if symmetrize and attention_weight:
                            n_neighs = float(len(neighs_B))
                            if not include_dead_weight:
                                n_neighs_alive = len(np.where(status_neigh_B == 1)[0])
                                neigh_count = n_neighs_alive
                            else:
                                neigh_count = n_neighs
                            if neigh_count > 0:
                                weight_A = 1. / neigh_count
                            else:
                                weight_A = np.nan

                            if not include_dead_weight and status_A[k] == 0:
                                weight_A = 0

                        neighs = []
                        setA.at[index_A[k], neigh_col] = []
                        for n in range(len(neighs_B)):

                            # index in setB
                            n_index = np.where(ids_B == neighs_B[n])[0][0]
                            # Assess if neigh B is closest to A
                            if attention_weight:
                                if closest_A[n_index] == ids_A[k]:
                                    closest = True
                                else:
                                    closest = False

                            if symmetrize:
                                # Load neighborhood previous data
                                sym_neigh = setB.loc[index_B[n_index], neigh_col]
                                if neighs_B[n] == closest_B_cell:
                                    closest_b = True
                                else:
                                    closest_b = False
                                if isinstance(sym_neigh, list):
                                    sym_neigh.append({'id': ids_A[k], 'distance': dist_B[n], 'status': status_A[k],
                                                      'intersection': intersect_B[n]})
                                else:
                                    sym_neigh = [{'id': ids_A[k], 'distance': dist_B[n], 'status': status_A[k],
                                                  'intersection': intersect_B[n]}]
                                if attention_weight:
                                    sym_neigh[-1].update({'weight': weight_A, 'closest': closest_b})

                            # Write the minimum info about neighborhing cell B
                            neigh_dico = {'id': neighs_B[n], 'distance': dist_B[n], 'status': status_neigh_B[n],
                                          'intersection': intersect_B[n]}
                            if attention_weight:
                                neigh_dico.update({'weight': weights[n_index], 'closest': closest})

                            if compute_cum_sum:
                                # Compute the integrated presence of the neighboring cell B
                                assert cl[1][
                                           'track'] == 'TRACK_ID', 'The set B does not seem to contain tracked data. The cumulative time will be meaningless.'
                                past_neighs = [[ll['id'] for ll in l] if len(l) > 0 else [None] for l in setA.loc[
                                    (setA[cl[0]['track']] == ids_A[k]) & (
                                                setA[cl[0]['time']] <= t), neigh_col].to_numpy()]
                                past_neighs = [item for sublist in past_neighs for item in sublist]

                                if attention_weight:
                                    past_weights = [[ll['weight'] for ll in l] if len(l) > 0 else [None] for l in
                                                    setA.loc[
                                                        (setA[cl[0]['track']] == ids_A[k]) & (
                                                                setA[cl[0]['time']] <= t), neigh_col].to_numpy()]
                                    past_weights = [item for sublist in past_weights for item in sublist]

                                cum_sum = len(np.where(past_neighs == neighs_B[n])[0])
                                neigh_dico.update({'cumulated_presence': cum_sum + 1})

                                if attention_weight:
                                    cum_sum_weighted = np.sum(
                                        [w if l == neighs_B[n] else 0 for l, w in zip(past_neighs, past_weights)])
                                    neigh_dico.update(
                                        {'cumulated_presence_weighted': cum_sum_weighted + weights[n_index]})

                            if symmetrize:
                                setB.at[index_B[n_index], neigh_col] = sym_neigh

                            neighs.append(neigh_dico)

                        setA.at[index_A[k], neigh_col] = neighs

                self.sum_done += 1 / len(timeline) * 100
                mean_exec_per_step = (time.time() - self.t0) / (self.sum_done * len(timeline)/ 100 + 1)
                pred_time = (len(timeline) - (self.sum_done * len(timeline) / 100 + 1)) * mean_exec_per_step
                print(f"{self.sum_done=} {pred_time=}")
                self.queue.put([self.sum_done, pred_time])

        return setA, setB



    def compute_contact_neighborhood_at_position(self, pos, distance, population=['targets', 'effectors'], theta_dist=None,
                                                 img_shape=(2048, 2048), return_tables=False, clear_neigh=False,
                                                 event_time_col=None,
                                                 neighborhood_kwargs={'mode': 'two-pop', 'status': None,
                                                                      'not_status_option': None,
                                                                      'include_dead_weight': True,
                                                                      "compute_cum_sum": False,
                                                                      "attention_weight": True, 'symmetrize': True}):

        pos = pos.replace('\\', '/')
        pos = rf"{pos}"
        assert os.path.exists(pos), f'Position {pos} is not a valid path.'

        if isinstance(population, str):
            population = [population, population]

        if not isinstance(distance, list):
            distance = [distance]
        if not theta_dist is None and not isinstance(theta_dist, list):
            theta_dist = [theta_dist]

        if theta_dist is None:
            theta_dist = [0 for d in distance]  # 0.9*d
        assert len(theta_dist) == len(distance), 'Incompatible number of distances and number of edge thresholds.'

        if population[0] == population[1]:
            neighborhood_kwargs.update({'mode': 'self'})
        if population[1] != population[0]:
            neighborhood_kwargs.update({'mode': 'two-pop'})

        df_A, path_A = get_position_table(pos, population=population[0], return_path=True)
        df_B, path_B = get_position_table(pos, population=population[1], return_path=True)
        if df_A is None or df_B is None:
            return None

        if clear_neigh:
            if os.path.exists(path_A.replace('.csv', '.pkl')):
                os.remove(path_A.replace('.csv', '.pkl'))
            if os.path.exists(path_B.replace('.csv', '.pkl')):
                os.remove(path_B.replace('.csv', '.pkl'))
            df_pair, pair_path = get_position_table(pos, population='pairs', return_path=True)
            if df_pair is not None:
                os.remove(pair_path)

        df_A_pkl = get_position_pickle(pos, population=population[0], return_path=False)
        df_B_pkl = get_position_pickle(pos, population=population[1], return_path=False)

        if df_A_pkl is not None:
            pkl_columns = np.array(df_A_pkl.columns)
            neigh_columns = np.array([c.startswith('neighborhood') for c in pkl_columns])
            cols = list(pkl_columns[neigh_columns]) + ['FRAME']

            id_col = extract_identity_col(df_A_pkl)
            cols.append(id_col)
            on_cols = [id_col, 'FRAME']

            print(f'Recover {cols} from the pickle file...')
            try:
                df_A = pd.merge(df_A, df_A_pkl.loc[:, cols], how="outer", on=on_cols)
                print(df_A.columns)
            except Exception as e:
                print(f'Failure to merge pickle and csv files: {e}')

        if df_B_pkl is not None and df_B is not None:
            pkl_columns = np.array(df_B_pkl.columns)
            neigh_columns = np.array([c.startswith('neighborhood') for c in pkl_columns])
            cols = list(pkl_columns[neigh_columns]) + ['FRAME']

            id_col = extract_identity_col(df_B_pkl)
            cols.append(id_col)
            on_cols = [id_col, 'FRAME']

            print(f'Recover {cols} from the pickle file...')
            try:
                df_B = pd.merge(df_B, df_B_pkl.loc[:, cols], how="outer", on=on_cols)
            except Exception as e:
                print(f'Failure to merge pickle and csv files: {e}')

        labelsA = locate_labels(pos, population=population[0])
        if population[1] == population[0]:
            labelsB = None
        else:
            labelsB = locate_labels(pos, population=population[1])

        if clear_neigh:
            unwanted = df_A.columns[df_A.columns.str.contains('neighborhood')]
            df_A = df_A.drop(columns=unwanted)
            unwanted = df_B.columns[df_B.columns.str.contains('neighborhood')]
            df_B = df_B.drop(columns=unwanted)

        print(f"Distance: {distance} for mask contact")
        df_A, df_B = self.mask_contact_neighborhood(df_A, df_B, labelsA, labelsB, distance, **neighborhood_kwargs)
        if df_A is None or df_B is None or len(df_A) == 0:
            return None

        for td, d in zip(theta_dist, distance):

            if neighborhood_kwargs['mode'] == 'two-pop':
                neigh_col = f'neighborhood_2_contact_{d}_px'
            elif neighborhood_kwargs['mode'] == 'self':
                neigh_col = f'neighborhood_self_contact_{d}_px'

            df_A.loc[df_A['class_id'].isnull(), neigh_col] = np.nan

            # edge_filter_A = (df_A['POSITION_X'] > td)&(df_A['POSITION_Y'] > td)&(df_A['POSITION_Y'] < (img_shape[0] - td))&(df_A['POSITION_X'] < (img_shape[1] - td))
            # edge_filter_B = (df_B['POSITION_X'] > td)&(df_B['POSITION_Y'] > td)&(df_B['POSITION_Y'] < (img_shape[0] - td))&(df_B['POSITION_X'] < (img_shape[1] - td))
            # df_A.loc[~edge_filter_A, neigh_col] = np.nan
            # df_B.loc[~edge_filter_B, neigh_col] = np.nan

            df_A = compute_neighborhood_metrics(df_A, neigh_col, metrics=['inclusive', 'intermediate'],
                                                decompose_by_status=True)
            if 'TRACK_ID' in list(df_A.columns):
                if not np.all(df_A['TRACK_ID'].isnull()):
                    df_A = mean_neighborhood_before_event(df_A, neigh_col, event_time_col,
                                                          metrics=['inclusive', 'intermediate'])
                    if event_time_col is not None:
                        df_A = mean_neighborhood_after_event(df_A, neigh_col, event_time_col,
                                                             metrics=['inclusive', 'intermediate'])
                    print('Done...')

        if not population[0] == population[1]:
            # Remove neighborhood column from neighbor table, rename with actual population name
            for td, d in zip(theta_dist, distance):
                if neighborhood_kwargs['mode'] == 'two-pop':
                    neigh_col = f'neighborhood_2_contact_{d}_px'
                    new_neigh_col = neigh_col.replace('_2_', f'_({population[0]}-{population[1]})_')
                    df_A = df_A.rename(columns={neigh_col: new_neigh_col})
                elif neighborhood_kwargs['mode'] == 'self':
                    neigh_col = f'neighborhood_self_contact_{d}_px'
                df_B = df_B.drop(columns=[neigh_col])
            df_B.to_pickle(path_B.replace('.csv', '.pkl'))

        cols_to_rename = [c for c in list(df_A.columns) if
                          c.startswith('intermediate_count_') or c.startswith('inclusive_count_') or c.startswith(
                              'exclusive_count_') or c.startswith('mean_count_')]
        new_col_names = [c.replace('_2_', f'_({population[0]}-{population[1]})_') for c in cols_to_rename]
        new_name_map = {}
        for k, c in enumerate(cols_to_rename):
            new_name_map.update({c: new_col_names[k]})
        df_A = df_A.rename(columns=new_name_map)

        print(f'{df_A.columns=}')
        df_A.to_pickle(path_A.replace('.csv', '.pkl'))

        unwanted = df_A.columns[df_A.columns.str.startswith('neighborhood_')]
        df_A2 = df_A.drop(columns=unwanted)
        df_A2.to_csv(path_A, index=False)

        if not population[0] == population[1]:
            unwanted = df_B.columns[df_B.columns.str.startswith('neighborhood_')]
            df_B_csv = df_B.drop(unwanted, axis=1, inplace=False)
            df_B_csv.to_csv(path_B, index=False)

        if return_tables:
            return df_A, df_B

    def run(self):
        print(f"Launching the neighborhood computation...")
        if self.protocol['neighborhood_type']=="distance_threshold":
            self.compute_neighborhood_at_position(self.pos,
                                             self.protocol['distance'],
                                             population=self.protocol['population'],
                                             theta_dist=None,
                                             img_shape=self.img_shape,
                                             return_tables=False,
                                             clear_neigh=self.protocol['clear_neigh'],
                                             event_time_col=self.protocol['event_time_col'],
                                             neighborhood_kwargs=self.protocol['neighborhood_kwargs'],
                                             )
            print(f"Computation done!")
        elif self.protocol["neighborhood_type"]=="mask_contact":
            print(f"Compute contact neigh!!")
            self.compute_contact_neighborhood_at_position(self.pos,
                                             self.protocol['distance'],
                                             population=self.protocol['population'],
                                             theta_dist=None,
                                             img_shape=self.img_shape,
                                             return_tables=False,
                                             clear_neigh=self.protocol['clear_neigh'],
                                             event_time_col=self.protocol['event_time_col'],
                                             neighborhood_kwargs=self.protocol['neighborhood_kwargs'],
                                             )
            print(f"Computation done!")


        # self.indices = list(range(self.img_num_channels.shape[1]))
        # chunks = np.array_split(self.indices, self.n_threads)
        #
        # self.timestep_dataframes = []
        # with concurrent.futures.ThreadPoolExecutor(max_workers=self.n_threads) as executor:
        #     results = executor.map(self.parallel_job,
        #                            chunks)  # list(map(lambda x: executor.submit(self.parallel_job, x), chunks))
        #     try:
        #         for i, return_value in enumerate(results):
        #             print(f'Thread {i} completed...')
        #             self.timestep_dataframes.extend(return_value)
        #     except Exception as e:
        #         print("Exception: ", e)
        #
        # print('Measurements successfully performed...')
        #
        # if len(self.timestep_dataframes) > 0:
        #
        #     df = pd.concat(self.timestep_dataframes)
        #
        #     if self.trajectories is not None:
        #         df = df.sort_values(by=[self.column_labels['track'], self.column_labels['time']])
        #         df = df.dropna(subset=[self.column_labels['track']])
        #     else:
        #         df['ID'] = np.arange(len(df))
        #         df = df.sort_values(by=[self.column_labels['time'], 'ID'])
        #
        #     df = df.reset_index(drop=True)
        #     df = _remove_invalid_cols(df)
        #
        #     df.to_csv(self.pos + os.sep.join(["output", "tables", self.table_name]), index=False)
        #     print(f'Measurement table successfully exported in  {os.sep.join(["output", "tables"])}...')
        #     print('Done.')
        # else:
        #     print('No measurement could be performed. Check your inputs.')
        #     print('Done.')

        # Send end signal
        self.queue.put("finished")
        self.queue.close()

    def end_process(self):

        self.terminate()
        self.queue.put("finished")

    def abort_process(self):

        self.terminate()
        self.queue.put("error")