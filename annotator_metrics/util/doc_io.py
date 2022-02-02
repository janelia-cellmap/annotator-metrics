import requests
from io import StringIO
import os
import pandas


class Row:
    def __init__(self, row):
        self.row = row

    def get(self, column):
        c = self.row[column]
        if "x" in c:
            return [int(c["z"]), int(c["y"]), int(c["x"])]
        elif "x min" in c:
            return (
                [int(c["z min"]), int(c["y min"]), int(c["x min"])],
                [int(c["z max"]), int(c["y max"]), int(c["x max"])],
            )
        else:
            # Way to treat it when  contains eg unnamed 0_level_0
            return c[c.keys()[0]]


class RowIterator:
    def __init__(self, df):
        self._df = df
        self._index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._index < self._df.shape[0]:
            result = self._df.iloc[self._index]
            self._index += 1
            return self._index - 1, Row(result)

        raise StopIteration


class MaskInformation:
    def __init__(self):
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

    def iterrows(self):
        return RowIterator(self.df)
