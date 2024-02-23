import pandas as pd


class Loader:
    def __init__(self, csv_file_path):
        self.df = None
        self.csv_file_path = csv_file_path
        self.expected_headers = ['user', 'item', 'rating']

    def load(self):
        self.df = pd.read_csv(self.csv_file_path)

        if not list(self.df.columns) == self.expected_headers:
            raise Exception("Cannot load data")
