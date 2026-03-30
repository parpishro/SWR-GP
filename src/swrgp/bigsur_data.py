"""Shared Big Sur data loading/splitting utilities for experiment scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


HYDROLOGICAL_SPLIT_YEAR = 30


def load_bigsur_dataframe(data_dir: Path) -> pd.DataFrame:
    data = pd.read_csv(data_dir / "bigsur.csv")
    data["date"] = pd.to_datetime(data["date"])
    data["hydr_year"] = np.where(
        data["date"].dt.month >= 10,
        data["date"].dt.year - 1978,
        data["date"].dt.year - 1979,
    )
    return data


def load_bigsur_train_test(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    data = load_bigsur_dataframe(data_dir)
    train = data[data["hydr_year"] < HYDROLOGICAL_SPLIT_YEAR].copy()
    test = data[data["hydr_year"] >= HYDROLOGICAL_SPLIT_YEAR].copy()
    return train, test
