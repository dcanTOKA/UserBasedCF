from pydantic import BaseModel


class NeighborSimilarity(BaseModel):
    neighbor: str
    similarity: float
