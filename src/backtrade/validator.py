from typing import Any

import pandas as pd
from attrs import Attribute


def _na_validator(self: Any, attribute: "Attribute[Any]", value: Any) -> None:
    if pd.isna(value):
        raise ValueError(f"{attribute.name} cannot be {value}")
