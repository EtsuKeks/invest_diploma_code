from typing import Optional

from pydantic import BaseModel  # type: ignore

EPSILON = 1e-8


class PipelineSettings(BaseModel):
    input_csv: str = "binance_dump_with_valid_volumes_arbitrage_free.csv"
    spot_npy: str = "spot_full.npy"
    output_csv: Optional[str] = None
    risk_free_rate: float = 0.0
    epsilon: float = 1e-8
    total_hours: int = 2683
