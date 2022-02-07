import requests
import pandas
from io import StringIO
import re


class Row:
    def __init__(self, df_row):
        self.df_row = df_row
        self._is_valid = True
        try:
            self.__get_useful_columns()
        except:
            self._is_valid = False

    def __get_column(self, column):
        c = self.df_row[column]
        if "x" in c:
            return [int(c["z"]), int(c["y"]), int(c["x"])]
        elif "x min" in c:
            return (
                [int(c["z min"]), int(c["y min"]), int(c["x min"])],
                [int(c["z max"]), int(c["y max"]), int(c["x max"])],
            )
        else:
            # Way to treat it when  contains eg unnamed 0_level_0
            val = c[c.keys()[0]]
            if not isinstance(val, str):
                return int(val)
            elif "Z:" in val:
                return val.replace("\\", "/").replace("Z:", "/groups/cosem/cosem")
            return val

    def __get_useful_columns(self):
        group_crop = self.__get_column("group")
        self.group = group_crop.split("_")[0]
        self.crop = group_crop.split("_")[1]
        self.raw_path = self.__get_column("raw data")
        self.gt_path = self.__get_column("crop pathway")
        self.original_coordinates = self.__get_column("original coordinates")
        self.gt_resolution = self.__get_column("groundtruth annotation resolution (nm)")
        self.correct_resolution = self.__get_column(
            "correct annotation resolution (nm)"
        )
        self.mins, self.maxs = self.__get_column("coordinates within crop")

    def is_valid(self):
        return self._is_valid


class MaskInformation:
    def __init__(self):
        self.__get_df_from_doc()
        self.__get_organelle_info()

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
        self.rows = [Row(r) for _, r in self.df.iterrows() if Row(r).is_valid()]

    def __get_organelle_info(self):
        self.all_organelle_names = []
        self.all_organelle_labels = []
        for c in self.df.columns:
            c = c[0]
            if re.search(r"\(\d\)", c) or re.search(r"\(\d\d\)", c):
                self.all_organelle_names.append(c.split(" (")[0])
                self.all_organelle_labels.append(int(c[c.find("(") + 1 : c.find(")")]))

