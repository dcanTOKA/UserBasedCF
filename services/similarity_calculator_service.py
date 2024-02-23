from itertools import combinations
from typing import Dict, List, Tuple

import numpy as np

from models.user_k_neighbor_similarity import NeighborSimilarity


class SimilarityCalculatorService:
    def __init__(self, df, K, mode='user', limit=2):
        self.df = df
        self.mode = mode
        self.limit = limit
        self.K = K
        self.logs = {}
        self.user_neighbors: Dict[str, List[NeighborSimilarity]] = {}
        self.user_means: Dict[str, float] = {}
        self.top_k_filtered = None

    @staticmethod
    def pearson_similarity_with_nan_removal(x, y) -> Tuple[float, float, float]:
        combined = list(zip(x, y))
        filtered_pairs = [(x_i, y_i) for x_i, y_i in combined if not np.isnan(x_i) and not np.isnan(y_i)]

        if not filtered_pairs:
            raise ValueError("Cannot continue processing. No available data found")

        clean_x, clean_y = zip(*filtered_pairs)

        mean_x = float(np.mean(clean_x))
        mean_y = float(np.mean(clean_y))

        clean_x = np.array(clean_x) - mean_x
        clean_y = np.array(clean_y) - mean_y

        cosine_similarity = np.dot(clean_x, clean_y) / (
                np.sqrt(np.dot(clean_x, clean_x)) * np.sqrt(np.dot(clean_y, clean_y)))

        return cosine_similarity, mean_x, mean_y

    def add_neighbor(self, user: str, neighbor: str, similarity: float):
        if not user in self.user_neighbors:
            self.user_neighbors[user] = []
        self.user_neighbors[user].append(
            NeighborSimilarity(neighbor=neighbor, similarity=similarity))
        self.user_neighbors[user].sort(key=lambda x: x.similarity, reverse=True)

    def add_mean(self, user: str, mean: float):
        if user not in self.user_means:
            self.user_means[user] = mean

    def calculate_similarity(self):
        if self.mode == 'user':
            iter_df = self.df
        elif self.mode == 'item':
            iter_df = self.df
        else:
            raise ValueError("Mode must be 'user' or 'item'.")

        for elem1, elem2 in combinations(iter_df.index, 2):
            status = False

            if self.mode == "user":
                if iter_df.loc[elem1].notnull().sum() > self.limit and iter_df.loc[elem2].notnull().sum() > self.limit:
                    status = True
            if self.mode == "item":
                if iter_df[elem1].notnull().sum(axis=1) > self.limit and iter_df[elem2].notnull().sum(
                        axis=1) > self.limit:
                    status = True

            if status:
                similarity, mean_elem_1, mean_elem_2 = self.pearson_similarity_with_nan_removal(iter_df.loc[elem1],
                                                                                                iter_df.loc[elem2])

                self.add_mean(elem1, mean_elem_1)
                self.add_mean(elem2, mean_elem_2)

                self.add_neighbor(elem1, elem2, similarity)
                self.add_neighbor(elem2, elem1, similarity)

            else:
                self.logs[f'{elem1}-{elem2}'] = "Interaction count below limit."

        self.top_k_filtered = {key: value[:self.K] for key, value in self.user_neighbors.items()}
