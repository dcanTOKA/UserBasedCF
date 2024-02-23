from typing import Union, Dict, List

import numpy as np

from models.recommendation import Recommendation
from models.user_k_neighbor_similarity import NeighborSimilarity


class RecommendService:
    def __init__(self, df, means: Dict[str, float], top_k_neighbors: Dict[str, List[NeighborSimilarity]]):
        self.df = df
        self.means = means
        self.top_k_neighbors = top_k_neighbors
        self.recommendations = None

    def recommend_items_user_based(self, user, n_recommendations=5):
        user_ratings = self.df.loc[user]
        missing_items = user_ratings[user_ratings.isnull()].index.tolist()
        predictions = []

        for item in missing_items:
            predicted_rating = self.predict_rating(user, item)
            if predicted_rating is not None:
                predictions.append(Recommendation(item=item, rating=predicted_rating))

        self.recommendations = sorted(predictions, key=lambda x: x.rating, reverse=True)[:n_recommendations]

    def predict_rating(self, user, item) -> Union[float, None]:
        num = 0
        den = 0

        for neighbor_sim in self.top_k_neighbors[user]:
            print(f"{neighbor_sim.similarity} *  {(self.df.loc[neighbor_sim.neighbor][item] - self.means[neighbor_sim.neighbor])}")
            num += neighbor_sim.similarity * (self.df.loc[neighbor_sim.neighbor][item] - self.means[neighbor_sim.neighbor])
            den += abs(neighbor_sim.similarity)

        if den == 0:
            return None

        pred_rating = self.means[user] + (num / den)

        return max(1, min(5, pred_rating))
