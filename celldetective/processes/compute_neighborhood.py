from multiprocessing import Process
import time
import os

from celldetective.utils.image_loaders import locate_labels
from celldetective.utils.data_loaders import get_position_table, get_position_pickle

from tqdm import tqdm
import numpy as np
import pandas as pd
from art import tprint

from celldetective.neighborhood import (
    _fill_distance_neighborhood_at_t,
    set_live_status,
    compute_attention_weight,
    compute_neighborhood_metrics,
    mean_neighborhood_after_event,
    mean_neighborhood_before_event,
    _compute_mask_contact_dist_map,
    _fill_contact_neighborhood_at_t,
)
from celldetective.utils.data_cleaning import extract_identity_col
from scipy.spatial.distance import cdist
from celldetective.relative_measurements import measure_pair_signals_at_position


class NeighborhoodProcess(Process):

    def __init__(self, queue=None, process_args=None):

        super().__init__()

        self.queue = queue

        if process_args is not None:
            for key, value in process_args.items():
                setattr(self, key, value)

        self.column_labels = {
            "track": "TRACK_ID",
            "time": "FRAME",
            "x": "POSITION_X",
            "y": "POSITION_Y",
        }

        tprint("Neighborhood")

        if not hasattr(self, "well_progress"):
            self.well_progress = 0
        if not hasattr(self, "pos_progress"):
            self.pos_progress = 0
        if not hasattr(self, "measure_pairs"):
            self.measure_pairs = False

        self.sum_done = 0
        self.t0 = time.time()

    def mask_contact_neighborhood(
        self,
        setA,
        setB,
        labelsA,
        labelsB,
        distance,
        mode="two-pop",
        status=None,
        not_status_option=None,
        compute_cum_sum=True,
        attention_weight=True,
        symmetrize=True,
        include_dead_weight=True,
        column_labels={
            "track": "TRACK_ID",
            "time": "FRAME",
            "x": "POSITION_X",
            "y": "POSITION_Y",
            "mask_id": "class_id",
        },
    ):

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
            temp_column_labels.update({"track": id_col})
            if id_col == "ID":
                compute_cum_sum = False

            cl.append(temp_column_labels)

        setA = setA.loc[~setA[cl[0]["track"]].isnull(), :].copy()
        setB = setB.loc[~setB[cl[1]["track"]].isnull(), :].copy()

        if labelsB is None:
            labelsB = [None] * len(labelsA)

        for d in distance:
            # loop over each provided distance
            if mode == "two-pop":
                neigh_col = f"neighborhood_2_contact_{d}_px"
            elif mode == "self":
                neigh_col = f"neighborhood_self_contact_{d}_px"
            else:
                print("Please provide a valid mode between `two-pop` and `self`...")
                return None

            setA[neigh_col] = np.nan
            setA[neigh_col] = setA[neigh_col].astype(object)

            setB[neigh_col] = np.nan
            setB[neigh_col] = setB[neigh_col].astype(object)

            # Loop over each available timestep
            timeline = np.unique(
                np.concatenate(
                    [setA[cl[0]["time"]].to_numpy(), setB[cl[1]["time"]].to_numpy()]
                )
            ).astype(int)

            self.sum_done = 0
            self.t0 = time.time()

            for t in tqdm(timeline):

                setA_t = setA.loc[setA[cl[0]["time"]] == t, :].copy()
                setB_t = setB.loc[setB[cl[1]["time"]] == t, :].copy()

                if len(setA_t) > 0 and len(setB_t) > 0:
                    dist_map, intersection_map = _compute_mask_contact_dist_map(
                        setA_t,
                        setB_t,
                        labelsA[t],
                        labelsB[t],
                        distance=d,
                        mode=mode,
                        column_labelsA=cl[0],
                        column_labelsB=cl[1],
                    )

                    d_filter = 1.0e05
                    if attention_weight:
                        status_A = setA_t[status[0]].to_numpy()
                        ids_A = setA_t[cl[0]["track"]].to_numpy()
                        weights, closest_A = compute_attention_weight(
                            dist_map,
                            d_filter,
                            status_A,
                            ids_A,
                            axis=1,
                            include_dead_weight=include_dead_weight,
                        )
                    else:
                        weights = None
                        closest_A = None

                    _fill_contact_neighborhood_at_t(
                        t,
                        setA,
                        setB,
                        dist_map,
                        intersection_map=intersection_map,
                        attention_weight=attention_weight,
                        include_dead_weight=include_dead_weight,
                        symmetrize=symmetrize,
                        compute_cum_sum=compute_cum_sum,
                        weights=weights,
                        closest_A=closest_A,
                        neigh_col=neigh_col,
                        column_labelsA=cl[0],
                        column_labelsB=cl[1],
                        statusA=status[0],
                        statusB=status[1],
                        d_filter=d_filter,
                    )

                self.sum_done += 1 / len(timeline) * 100
                mean_exec_per_step = (time.time() - self.t0) / (
                    self.sum_done * len(timeline) / 100 + 1
                )
                pred_time = (
                    len(timeline) - (self.sum_done * len(timeline) / 100 + 1)
                ) * mean_exec_per_step
                self.queue.put(
                    {
                        "frame_progress": self.sum_done,
                        "frame_time": f"Time left: {round(pred_time, 1)}s",
                        "well_progress": self.well_progress,
                        "pos_progress": self.pos_progress,
                    }
                )

        return setA, setB

    def distance_cut_neighborhood(
        self,
        setA,
        setB,
        distance,
        mode="two-pop",
        status=None,
        not_status_option=None,
        compute_cum_sum=True,
        attention_weight=True,
        symmetrize=True,
        include_dead_weight=True,
        column_labels={
            "track": "TRACK_ID",
            "time": "FRAME",
            "x": "POSITION_X",
            "y": "POSITION_Y",
        },
    ):
        # Check live_status option
        if setA is not None and setB is not None:
            setA, setB, status = set_live_status(setA, setB, status, not_status_option)
        else:
            return None, None

        # Check distance option
        if not isinstance(distance, list):
            distance = [distance]

        for d in distance:
            # loop over each provided distance

            if mode == "two-pop":
                neigh_col = f"neighborhood_2_circle_{d}_px"
            elif mode == "self":
                neigh_col = f"neighborhood_self_circle_{d}_px"

            cl = []
            for s in [setA, setB]:

                # Check whether data can be tracked
                temp_column_labels = column_labels.copy()

                id_col = extract_identity_col(s)
                temp_column_labels.update({"track": id_col})
                if id_col == "ID":
                    compute_cum_sum = (
                        False  # if no tracking data then cum_sum is not relevant
                    )
                cl.append(temp_column_labels)

                # Remove nan tracks (cells that do not belong to a track)
                s[neigh_col] = np.nan
                s[neigh_col] = s[neigh_col].astype(object)
                s.dropna(subset=[cl[-1]["track"]], inplace=True)

            # Loop over each available timestep
            timeline = np.unique(
                np.concatenate(
                    [setA[cl[0]["time"]].to_numpy(), setB[cl[1]["time"]].to_numpy()]
                )
            ).astype(int)

            self.sum_done = 0
            self.t0 = time.time()

            for t in tqdm(timeline):

                coordinates_A = setA.loc[
                    setA[cl[0]["time"]] == t, [cl[0]["x"], cl[0]["y"]]
                ].to_numpy()
                ids_A = setA.loc[setA[cl[0]["time"]] == t, cl[0]["track"]].to_numpy()
                status_A = setA.loc[setA[cl[0]["time"]] == t, status[0]].to_numpy()

                coordinates_B = setB.loc[
                    setB[cl[1]["time"]] == t, [cl[1]["x"], cl[1]["y"]]
                ].to_numpy()
                ids_B = setB.loc[setB[cl[1]["time"]] == t, cl[1]["track"]].to_numpy()

                if len(ids_A) > 0 and len(ids_B) > 0:

                    # compute distance matrix
                    dist_map = cdist(coordinates_A, coordinates_B, metric="euclidean")

                    if attention_weight:
                        weights, closest_A = compute_attention_weight(
                            dist_map,
                            d,
                            status_A,
                            ids_A,
                            axis=1,
                            include_dead_weight=include_dead_weight,
                        )

                    _fill_distance_neighborhood_at_t(
                        t,
                        setA,
                        setB,
                        dist_map,
                        attention_weight=attention_weight,
                        include_dead_weight=include_dead_weight,
                        symmetrize=symmetrize,
                        compute_cum_sum=compute_cum_sum,
                        weights=weights,
                        closest_A=closest_A,
                        neigh_col=neigh_col,
                        column_labelsA=cl[0],
                        column_labelsB=cl[1],
                        statusA=status[0],
                        statusB=status[1],
                        distance=d,
                    )

                self.sum_done += 1 / len(timeline) * 100
                mean_exec_per_step = (time.time() - self.t0) / (
                    self.sum_done * len(timeline) / 100 + 1
                )
                pred_time = (
                    len(timeline) - (self.sum_done * len(timeline) / 100 + 1)
                ) * mean_exec_per_step
                self.queue.put(
                    {
                        "frame_progress": self.sum_done,
                        "frame_time": f"Time left: {round(pred_time, 1)}s",
                        "well_progress": self.well_progress,
                        "pos_progress": self.pos_progress,
                    }
                )

        return setA, setB

    def compute_neighborhood_at_position(
        self,
        pos,
        distance,
        population=["targets", "effectors"],
        theta_dist=None,
        img_shape=(2048, 2048),
        return_tables=False,
        clear_neigh=False,
        event_time_col=None,
        neighborhood_kwargs={
            "mode": "two-pop",
            "status": None,
            "not_status_option": None,
            "include_dead_weight": True,
            "compute_cum_sum": False,
            "attention_weight": True,
            "symmetrize": True,
        },
    ):

        pos = pos.replace("\\", "/")
        pos = rf"{pos}"
        assert os.path.exists(pos), f"Position {pos} is not a valid path."

        if isinstance(population, str):
            population = [population, population]

        if not isinstance(distance, list):
            distance = [distance]
        if not theta_dist is None and not isinstance(theta_dist, list):
            theta_dist = [theta_dist]

        if theta_dist is None:
            theta_dist = [0.9 * d for d in distance]
        assert len(theta_dist) == len(
            distance
        ), "Incompatible number of distances and number of edge thresholds."

        if population[0] == population[1]:
            neighborhood_kwargs.update({"mode": "self"})
        if population[1] != population[0]:
            neighborhood_kwargs.update({"mode": "two-pop"})

        df_A, path_A = get_position_table(
            pos, population=population[0], return_path=True
        )
        df_B, path_B = get_position_table(
            pos, population=population[1], return_path=True
        )
        if df_A is None or df_B is None:
            return None

        if clear_neigh:
            if os.path.exists(path_A.replace(".csv", ".pkl")):
                os.remove(path_A.replace(".csv", ".pkl"))
            if os.path.exists(path_B.replace(".csv", ".pkl")):
                os.remove(path_B.replace(".csv", ".pkl"))
            df_pair, pair_path = get_position_table(
                pos, population="pairs", return_path=True
            )
            if df_pair is not None:
                os.remove(pair_path)

        df_A_pkl = get_position_pickle(pos, population=population[0], return_path=False)
        df_B_pkl = get_position_pickle(pos, population=population[1], return_path=False)

        if df_A_pkl is not None:
            pkl_columns = np.array(df_A_pkl.columns)
            neigh_columns = np.array(
                [c.startswith("neighborhood") for c in pkl_columns]
            )
            cols = list(pkl_columns[neigh_columns]) + ["FRAME"]

            id_col = extract_identity_col(df_A_pkl)
            cols.append(id_col)
            on_cols = [id_col, "FRAME"]

            print(f"Recover {cols} from the pickle file...")
            try:
                df_A = pd.merge(df_A, df_A_pkl.loc[:, cols], how="outer", on=on_cols)
                print(df_A.columns)
            except Exception as e:
                print(f"Failure to merge pickle and csv files: {e}")

        if df_B_pkl is not None and df_B is not None:
            pkl_columns = np.array(df_B_pkl.columns)
            neigh_columns = np.array(
                [c.startswith("neighborhood") for c in pkl_columns]
            )
            cols = list(pkl_columns[neigh_columns]) + ["FRAME"]

            id_col = extract_identity_col(df_B_pkl)
            cols.append(id_col)
            on_cols = [id_col, "FRAME"]

            print(f"Recover {cols} from the pickle file...")
            try:
                df_B = pd.merge(df_B, df_B_pkl.loc[:, cols], how="outer", on=on_cols)
            except Exception as e:
                print(f"Failure to merge pickle and csv files: {e}")

        if clear_neigh:
            unwanted = df_A.columns[df_A.columns.str.contains("neighborhood")]
            df_A = df_A.drop(columns=unwanted)
            unwanted = df_B.columns[df_B.columns.str.contains("neighborhood")]
            df_B = df_B.drop(columns=unwanted)

        df_A, df_B = self.distance_cut_neighborhood(
            df_A, df_B, distance, **neighborhood_kwargs
        )

        if df_A is None or df_B is None or len(df_A) == 0:
            return None

        for td, d in zip(theta_dist, distance):

            if neighborhood_kwargs["mode"] == "two-pop":
                neigh_col = f"neighborhood_2_circle_{d}_px"

            elif neighborhood_kwargs["mode"] == "self":
                neigh_col = f"neighborhood_self_circle_{d}_px"

            # edge_filter_A = (df_A['POSITION_X'] > td)&(df_A['POSITION_Y'] > td)&(df_A['POSITION_Y'] < (img_shape[0] - td))&(df_A['POSITION_X'] < (img_shape[1] - td))
            # edge_filter_B = (df_B['POSITION_X'] > td)&(df_B['POSITION_Y'] > td)&(df_B['POSITION_Y'] < (img_shape[0] - td))&(df_B['POSITION_X'] < (img_shape[1] - td))
            # df_A.loc[~edge_filter_A, neigh_col] = np.nan
            # df_B.loc[~edge_filter_B, neigh_col] = np.nan

            print("Count neighborhood...")
            df_A = compute_neighborhood_metrics(
                df_A,
                neigh_col,
                metrics=["inclusive", "exclusive", "intermediate"],
                decompose_by_status=True,
            )
            # if neighborhood_kwargs['symmetrize']:
            # 	df_B = compute_neighborhood_metrics(df_B, neigh_col, metrics=['inclusive','exclusive','intermediate'], decompose_by_status=True)
            print("Done...")

            if "TRACK_ID" in list(df_A.columns):
                if not np.all(df_A["TRACK_ID"].isnull()):
                    print("Estimate average neighborhood before/after event...")
                    df_A = mean_neighborhood_before_event(
                        df_A, neigh_col, event_time_col
                    )
                    if event_time_col is not None:
                        df_A = mean_neighborhood_after_event(
                            df_A, neigh_col, event_time_col
                        )
                    print("Done...")

        if not population[0] == population[1]:
            # Remove neighborhood column from neighbor table, rename with actual population name
            for td, d in zip(theta_dist, distance):
                if neighborhood_kwargs["mode"] == "two-pop":
                    neigh_col = f"neighborhood_2_circle_{d}_px"
                    new_neigh_col = neigh_col.replace(
                        "_2_", f"_({population[0]}-{population[1]})_"
                    )
                    df_A = df_A.rename(columns={neigh_col: new_neigh_col})
                elif neighborhood_kwargs["mode"] == "self":
                    neigh_col = f"neighborhood_self_circle_{d}_px"
                df_B = df_B.drop(columns=[neigh_col])
            df_B.to_pickle(path_B.replace(".csv", ".pkl"))

        cols_to_rename = [
            c
            for c in list(df_A.columns)
            if c.startswith("intermediate_count_")
            or c.startswith("inclusive_count_")
            or c.startswith("exclusive_count_")
            or c.startswith("mean_count_")
        ]
        new_col_names = [
            c.replace("_2_", f"_({population[0]}-{population[1]})_")
            for c in cols_to_rename
        ]
        new_name_map = {}
        for k, c in enumerate(cols_to_rename):
            new_name_map.update({c: new_col_names[k]})
        df_A = df_A.rename(columns=new_name_map)

        df_A.to_pickle(path_A.replace(".csv", ".pkl"))

        unwanted = df_A.columns[df_A.columns.str.startswith("neighborhood_")]
        df_A2 = df_A.drop(columns=unwanted)
        df_A2.to_csv(path_A, index=False)

        if not population[0] == population[1]:
            unwanted = df_B.columns[df_B.columns.str.startswith("neighborhood_")]
            df_B_csv = df_B.drop(unwanted, axis=1, inplace=False)
            df_B_csv.to_csv(path_B, index=False)

        if return_tables:
            return df_A, df_B

    def compute_contact_neighborhood_at_position(
        self,
        pos,
        distance,
        population=["targets", "effectors"],
        theta_dist=None,
        img_shape=(2048, 2048),
        return_tables=False,
        clear_neigh=False,
        event_time_col=None,
        neighborhood_kwargs={
            "mode": "two-pop",
            "status": None,
            "not_status_option": None,
            "include_dead_weight": True,
            "compute_cum_sum": False,
            "attention_weight": True,
            "symmetrize": True,
        },
    ):

        pos = pos.replace("\\", "/")
        pos = rf"{pos}"
        assert os.path.exists(pos), f"Position {pos} is not a valid path."

        if isinstance(population, str):
            population = [population, population]

        if not isinstance(distance, list):
            distance = [distance]
        if not theta_dist is None and not isinstance(theta_dist, list):
            theta_dist = [theta_dist]

        if theta_dist is None:
            theta_dist = [0 for d in distance]  # 0.9*d
        assert len(theta_dist) == len(
            distance
        ), "Incompatible number of distances and number of edge thresholds."

        if population[0] == population[1]:
            neighborhood_kwargs.update({"mode": "self"})
        if population[1] != population[0]:
            neighborhood_kwargs.update({"mode": "two-pop"})

        df_A, path_A = get_position_table(
            pos, population=population[0], return_path=True
        )
        df_B, path_B = get_position_table(
            pos, population=population[1], return_path=True
        )
        if df_A is None or df_B is None:
            return None

        if clear_neigh:
            if os.path.exists(path_A.replace(".csv", ".pkl")):
                os.remove(path_A.replace(".csv", ".pkl"))
            if os.path.exists(path_B.replace(".csv", ".pkl")):
                os.remove(path_B.replace(".csv", ".pkl"))
            df_pair, pair_path = get_position_table(
                pos, population="pairs", return_path=True
            )
            if df_pair is not None:
                os.remove(pair_path)

        df_A_pkl = get_position_pickle(pos, population=population[0], return_path=False)
        df_B_pkl = get_position_pickle(pos, population=population[1], return_path=False)

        if df_A_pkl is not None:
            pkl_columns = np.array(df_A_pkl.columns)
            neigh_columns = np.array(
                [c.startswith("neighborhood") for c in pkl_columns]
            )
            cols = list(pkl_columns[neigh_columns]) + ["FRAME"]

            id_col = extract_identity_col(df_A_pkl)
            cols.append(id_col)
            on_cols = [id_col, "FRAME"]

            print(f"Recover {cols} from the pickle file...")
            try:
                df_A = pd.merge(df_A, df_A_pkl.loc[:, cols], how="outer", on=on_cols)
                print(df_A.columns)
            except Exception as e:
                print(f"Failure to merge pickle and csv files: {e}")

        if df_B_pkl is not None and df_B is not None:
            pkl_columns = np.array(df_B_pkl.columns)
            neigh_columns = np.array(
                [c.startswith("neighborhood") for c in pkl_columns]
            )
            cols = list(pkl_columns[neigh_columns]) + ["FRAME"]

            id_col = extract_identity_col(df_B_pkl)
            cols.append(id_col)
            on_cols = [id_col, "FRAME"]

            print(f"Recover {cols} from the pickle file...")
            try:
                df_B = pd.merge(df_B, df_B_pkl.loc[:, cols], how="outer", on=on_cols)
            except Exception as e:
                print(f"Failure to merge pickle and csv files: {e}")

        labelsA = locate_labels(pos, population=population[0])
        if population[1] == population[0]:
            labelsB = None
        else:
            labelsB = locate_labels(pos, population=population[1])

        if clear_neigh:
            unwanted = df_A.columns[df_A.columns.str.contains("neighborhood")]
            df_A = df_A.drop(columns=unwanted)
            unwanted = df_B.columns[df_B.columns.str.contains("neighborhood")]
            df_B = df_B.drop(columns=unwanted)

        print(f"Distance: {distance} for mask contact")
        df_A, df_B = self.mask_contact_neighborhood(
            df_A, df_B, labelsA, labelsB, distance, **neighborhood_kwargs
        )
        if df_A is None or df_B is None or len(df_A) == 0:
            return None

        for td, d in zip(theta_dist, distance):

            if neighborhood_kwargs["mode"] == "two-pop":
                neigh_col = f"neighborhood_2_contact_{d}_px"
            elif neighborhood_kwargs["mode"] == "self":
                neigh_col = f"neighborhood_self_contact_{d}_px"
            else:
                print("Invalid mode...")
                return None

            df_A.loc[df_A["class_id"].isnull(), neigh_col] = np.nan

            # edge_filter_A = (df_A['POSITION_X'] > td)&(df_A['POSITION_Y'] > td)&(df_A['POSITION_Y'] < (img_shape[0] - td))&(df_A['POSITION_X'] < (img_shape[1] - td))
            # edge_filter_B = (df_B['POSITION_X'] > td)&(df_B['POSITION_Y'] > td)&(df_B['POSITION_Y'] < (img_shape[0] - td))&(df_B['POSITION_X'] < (img_shape[1] - td))
            # df_A.loc[~edge_filter_A, neigh_col] = np.nan
            # df_B.loc[~edge_filter_B, neigh_col] = np.nan

            df_A = compute_neighborhood_metrics(
                df_A,
                neigh_col,
                metrics=["inclusive", "intermediate"],
                decompose_by_status=True,
            )
            if "TRACK_ID" in list(df_A.columns):
                if not np.all(df_A["TRACK_ID"].isnull()):
                    df_A = mean_neighborhood_before_event(
                        df_A,
                        neigh_col,
                        event_time_col,
                        metrics=["inclusive", "intermediate"],
                    )
                    if event_time_col is not None:
                        df_A = mean_neighborhood_after_event(
                            df_A,
                            neigh_col,
                            event_time_col,
                            metrics=["inclusive", "intermediate"],
                        )
                    print("Done...")

        if not population[0] == population[1]:
            # Remove neighborhood column from neighbor table, rename with actual population name
            for td, d in zip(theta_dist, distance):
                if neighborhood_kwargs["mode"] == "two-pop":
                    neigh_col = f"neighborhood_2_contact_{d}_px"
                    new_neigh_col = neigh_col.replace(
                        "_2_", f"_({population[0]}-{population[1]})_"
                    )
                    df_A = df_A.rename(columns={neigh_col: new_neigh_col})
                elif neighborhood_kwargs["mode"] == "self":
                    neigh_col = f"neighborhood_self_contact_{d}_px"
                else:
                    print("Invalid mode...")
                    return None
                df_B = df_B.drop(columns=[neigh_col])
            df_B.to_pickle(path_B.replace(".csv", ".pkl"))

        cols_to_rename = [
            c
            for c in list(df_A.columns)
            if c.startswith("intermediate_count_")
            or c.startswith("inclusive_count_")
            or c.startswith("exclusive_count_")
            or c.startswith("mean_count_")
        ]
        new_col_names = [
            c.replace("_2_", f"_({population[0]}-{population[1]})_")
            for c in cols_to_rename
        ]
        new_name_map = {}
        for k, c in enumerate(cols_to_rename):
            new_name_map.update({c: new_col_names[k]})
        df_A = df_A.rename(columns=new_name_map)

        print(f"{df_A.columns=}")
        df_A.to_pickle(path_A.replace(".csv", ".pkl"))

        unwanted = df_A.columns[df_A.columns.str.startswith("neighborhood_")]
        df_A2 = df_A.drop(columns=unwanted)
        df_A2.to_csv(path_A, index=False)

        if not population[0] == population[1]:
            unwanted = df_B.columns[df_B.columns.str.startswith("neighborhood_")]
            df_B_csv = df_B.drop(unwanted, axis=1, inplace=False)
            df_B_csv.to_csv(path_B, index=False)

        if return_tables:
            return df_A, df_B

    def run(self):
        self.queue.put({"status": "Computing neighborhood..."})
        print(f"Launching the neighborhood computation...")
        if self.protocol["neighborhood_type"] == "distance_threshold":
            self.compute_neighborhood_at_position(
                self.pos,
                self.protocol["distance"],
                population=self.protocol["population"],
                theta_dist=None,
                img_shape=self.img_shape,
                return_tables=False,
                clear_neigh=self.protocol["clear_neigh"],
                event_time_col=self.protocol["event_time_col"],
                neighborhood_kwargs=self.protocol["neighborhood_kwargs"],
            )
            print(f"Computation done!")
        elif self.protocol["neighborhood_type"] == "mask_contact":
            print(f"Compute contact neigh!!")
            self.compute_contact_neighborhood_at_position(
                self.pos,
                self.protocol["distance"],
                population=self.protocol["population"],
                theta_dist=None,
                img_shape=self.img_shape,
                return_tables=False,
                clear_neigh=self.protocol["clear_neigh"],
                event_time_col=self.protocol["event_time_col"],
                neighborhood_kwargs=self.protocol["neighborhood_kwargs"],
            )
            print(f"Computation done!")

        if self.measure_pairs:
            self.queue.put({"status": "Measuring pairs..."})
            print(f"Measuring pairs...")

            distances = self.protocol["distance"]
            if not isinstance(distances, list):
                distances = [distances]

            for d in distances:
                # Construct the protocol dictionary expected by measure_pair_signals_at_position
                if self.protocol["population"][0] == self.protocol["population"][1]:
                    mode = "self"
                else:
                    mode = "two-pop"

                if self.protocol["neighborhood_type"] == "distance_threshold":
                    neigh_type = "circle"
                    if mode == "two-pop":
                        neigh_col = f"neighborhood_2_circle_{d}_px"
                    elif mode == "self":
                        neigh_col = f"neighborhood_self_circle_{d}_px"
                elif self.protocol["neighborhood_type"] == "mask_contact":
                    neigh_type = "contact"
                    if mode == "two-pop":
                        neigh_col = f"neighborhood_2_contact_{d}_px"
                    elif mode == "self":
                        neigh_col = f"neighborhood_self_contact_{d}_px"

                pair_protocol = {
                    "reference": self.protocol["population"][0],
                    "neighbor": self.protocol["population"][1],
                    "type": neigh_type,
                    "distance": d,
                    "description": neigh_col,
                }

                print(f"Processing pairs for {neigh_col}...")
                df_pairs = measure_pair_signals_at_position(self.pos, pair_protocol)

                if df_pairs is not None:
                    if "REFERENCE_ID" in list(df_pairs.columns):
                        previous_pair_table_path = self.pos + os.sep.join(
                            ["output", "tables", "trajectories_pairs.csv"]
                        )

                        if os.path.exists(previous_pair_table_path):
                            df_prev = pd.read_csv(previous_pair_table_path)
                            cols = [
                                c
                                for c in list(df_prev.columns)
                                if c in list(df_pairs.columns)
                            ]
                            df_pairs = pd.merge(df_prev, df_pairs, how="outer", on=cols)

                        try:
                            df_pairs = df_pairs.sort_values(
                                by=[
                                    "reference_population",
                                    "neighbor_population",
                                    "REFERENCE_ID",
                                    "NEIGHBOR_ID",
                                    "FRAME",
                                ]
                            )
                        except KeyError:
                            pass

                        df_pairs.to_csv(previous_pair_table_path, index=False)
                        print(f"Pair measurements saved to {previous_pair_table_path}")

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
