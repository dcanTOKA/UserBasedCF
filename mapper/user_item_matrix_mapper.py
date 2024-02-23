

class UserItemMatrixMapper:
    def __init__(self):
        self.pivot_matrix = None

    def fit(self, df):
        self.pivot_matrix = df.pivot(index='user', columns='item', values='rating')
        self.pivot_matrix.index.name = None
        self.pivot_matrix.columns.name = None
        return self

    def transform(self):
        return self.pivot_matrix

