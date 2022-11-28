from unittest import TestCase

import attrs
import numpy as np

from backtrade.validator import _na_validator


class TestNaValidator(TestCase):
    def test_na_validator(self):
        @attrs.define()
        class Test:
            a: int = attrs.field(validator=_na_validator)

        with self.assertRaises(ValueError):
            Test(a=np.nan)
        with self.assertRaises(ValueError):
            Test(a=None)  # type: ignore
