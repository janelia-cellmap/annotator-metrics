import os
from typing import Tuple, Union
import h5py
import requests
import pandas
from io import StringIO
import re
import numpy as np


class Row:
    def __init__(self, df_row: pandas.Series):
        self.df_row = df_row
        self._is_valid = True
        try:
            self.__get_useful_columns()
            self.__get_organelle_info()
        except Exception as e:
            self._is_valid = False

    def __get_column(
        self, column: str
    ) -> Union[int, str, np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        c = self.df_row[column]
        if "x" in c:
            return np.array([int(c["x"]), int(c["y"]), int(c["z"])], dtype=int)
        elif "x min" in c:
            try:
                output = (
                    np.array(
                        [int(c["x min"]), int(c["y min"]), int(c["z min"])], dtype=int
                    ),
                    np.array(
                        [int(c["x max"]), int(c["y max"]), int(c["z max"])], dtype=int
                    ),
                )
            except:
                with h5py.File(self.gt_path, "r") as f:
                    im = f["volumes"]["labels"]["gt"][:]
                    output = (np.array([0, 0, 0]), np.array(np.shape(im)))
            return output
        else:
            # Way to treat it when  contains eg unnamed 0_level_0
            val = c[c.keys()[0]]
            if not isinstance(val, str):
                return int(val)
            elif "Z:" in val:
                return val.replace("\\", "/").replace("Z:", "/groups/cellmap/cellmap")
            return val

    def __get_useful_columns(self):
        group_crop = self.__get_column("group")
        self.group = group_crop.rsplit("_", 1)[0]
        self.crop = group_crop.rsplit("_", 1)[1]
        self.raw_path = self.__get_column("raw data")
        self.cell_name = self.raw_path.split("/")[-1].split(".n5")[0]
        self.gt_path = self.__get_column("crop pathway")
        self.converted_4nm_coordinates = self.__get_column("converted 4nm coordinates")
        self.original_crop_size = self.__get_column("original crop size (pixels)")
        self.raw_resolution = self.__get_column("raw resolution (nm)")
        self.gt_resolution = self.__get_column("groundtruth annotation resolution (nm)")
        self.correct_resolution = self.__get_column(
            "correct annotation resolution (nm)"
        )
        self.mins, self.maxs = self.__get_column("coordinates within crop")

    def __get_organelle_info(self):
        self.organelle_info = {}
        # self.organelle_labels_with_predictions = []
        for h, c in self.df_row.iteritems():
            if c == "X":
                organelle_name = h[0].split("(")[0].split(" ")[0]
                organelle_label = int(h[0].split("(")[1].split(")")[0])
                self.organelle_info[organelle_name] = organelle_label

                # for the case of manually annotated lumen, we label the corresponding whole-organelle predictions/refinements with
                # that value, and overlay eg. membranes on top of that
                # organelle_name = organelle_name.split("-lum")[0]
                # df_row = prediction_paths_df.loc[
                #     (prediction_paths_df["Group"] == f"{self.group}_{self.crop}")
                #     & (prediction_paths_df["Dataset"] == self.dataset)
                #     & (prediction_paths_df["Class"] == organelle_name)
                # ]
                # if not df_row.empty and os.path.exists(
                #     df_row["Prediction Pathway"].values[0]
                # ):
                # then there are predictions for this specific organelle

                #    self.organelle_labels_with_predictions.append(organelle_label)

        # lumen are only specifically labeled by annotators, otherwise they are the remnants of organelle and mem predictions
        # so a predict
        combined_labels = {
            # mem, lumen, other
            "mito": [3, 4, 5],
            "er": [16, 17, 18, 19],
            "eres": [18, 19],
            "golgi": [6, 7],
            "vesicle": [8, 9],
            "endo": [10, 11],
            "lyso": [12, 13],
            "mt": [36, 30],
            "np": [23, 22],
            "nucleus": [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
            "ne": [20, 21],
        }

        all_values = set(self.organelle_info.values())
        for organelle, labels in combined_labels.items():
            if set(all_values).intersection(set(labels)):
                self.organelle_info[organelle] = labels

        # for organelle in self.organelle_info.keys():
        #     # these are all the annotated ones
        #     df_row = prediction_paths_df.loc[
        #         (prediction_paths_df["Group"] == f"{self.group}_{self.crop}")
        #         & (prediction_paths_df["Dataset"] == self.dataset)
        #         & (prediction_paths_df["Class"] == organelle_name)
        #     ]
        #     if not df_row.empty and os.path.exists(
        #         df_row["Prediction Pathway"].values[0]
        #     ):
        #         # then there are predictions for this specific organelle
        #         self.organelle_labels_with_predictions.append(
        #             labels_dict[organelle_name]
        #         )

    def is_valid(self):
        return self._is_valid


class MaskInformation:
    def __init__(
        self,
        group: str = None,
        crop: Union[list, str] = "all",
        base_path_to_check: str = None,
    ):
        self.__get_df_from_doc()
        self.__get_organelle_info()

        # filter by group and crop
        if group:
            filtered_rows = []
            for row in self.rows:
                if row.group in group:
                    if crop != "all":
                        if row.crop in crop:
                            filtered_rows.append(row)
                    else:
                        filtered_rows.append(row)
            self.rows = filtered_rows

        # filter to check if path exists
        if base_path_to_check:
            filtered_rows = []
            for row in self.rows:
                path = f"{base_path_to_check}/{row.group}/{row.crop}"
                if os.path.exists(f"{path}") or os.path.exists(f"{path}.n5"):
                    filtered_rows.append(row)
            self.rows = filtered_rows

    def __get_df_from_doc(self):
        received = requests.get(
            "https://docs.google.com/spreadsheets/d/1GID90G3kUOM9qhuvlNc6Sqlf45ORMHDIaHBivYNzx3E/export?format=csv&id=1GID90G3kUOM9qhuvlNc6Sqlf45ORMHDIaHBivYNzx3E&gid=585168624"
        )
        mask_information = received.text

        firstline = mask_information.partition("\n")[0]
        column_names = firstline.split(",")
        previous_column_name = column_names[0]
        for i, column_name in enumerate(column_names):
            if column_name:
                previous_column_name = column_name
            else:
                column_names[i] = previous_column_name
        updated_firstline = (",").join(column_names)
        mask_information = mask_information.replace(firstline, updated_firstline)

        self.df = pandas.read_csv(StringIO(("").join(mask_information)), header=[0, 1])
        self.rows = []
        for _, r in self.df.iterrows():
            row = Row(r)
            if row.is_valid():
                self.rows.append(row)

    def __get_organelle_info(self):
        self.all_organelle_names = []
        self.all_organelle_labels = []
        for c in self.df.columns:
            c = c[0]
            if re.search(r"\(\d\)", c) or re.search(r"\(\d\d\)", c):
                self.all_organelle_names.append(c.split(" (")[0])
                self.all_organelle_labels.append(int(c[c.find("(") + 1 : c.find(")")]))


def get_prediction_paths_df():
    received = requests.get(
        "https://docs.google.com/spreadsheets/d/1GID90G3kUOM9qhuvlNc6Sqlf45ORMHDIaHBivYNzx3E/export?format=csv&id=1GID90G3kUOM9qhuvlNc6Sqlf45ORMHDIaHBivYNzx3E&gid=733638280"
    )

    prediction_information = received.text

    prediction_paths_df = pandas.read_csv(StringIO(("").join(prediction_information)))
    for _, row in prediction_paths_df.iterrows():
        if type(row.Group) != str:
            row.Group = previous_group
        if type(row.Dataset) != str:
            row.Dataset = previous_dataset
        previous_group = row.Group
        previous_dataset = row.Dataset
    return prediction_paths_df
