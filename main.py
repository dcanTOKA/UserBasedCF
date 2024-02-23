from loader.rating_loader import Loader
import argparse

from mapper.user_item_matrix_mapper import UserItemMatrixMapper
from services.recommend_service import RecommendService
from services.similarity_calculator_service import SimilarityCalculatorService


def main(path: str):
    loader = Loader(path)
    loader.load()

    mapper = UserItemMatrixMapper()
    pivot_df = mapper.fit(loader.df).transform()

    sim_service = SimilarityCalculatorService(pivot_df)
    sim_service.calculate_similarity()

    recommend_service = RecommendService(pivot_df, sim_service.user_means, sim_service.top_k_filtered)
    recommend_service.recommend_items_user_based("Alice")

    print(recommend_service.recommendations)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="User based Collaborative Filtering")
    parser.add_argument("--csv_path", type=str, default="data/ratings_demo.csv",
                        help="Path to the csv file containing user, item and rating.")

    args = parser.parse_args()

    main(args.csv_path)
