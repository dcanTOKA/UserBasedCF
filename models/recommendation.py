from pydantic import BaseModel
from typing import Dict, List, Tuple


class Recommendation(BaseModel):
    item: str
    rating: float
