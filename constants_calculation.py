import os
from numpy import finfo, float64
from decimal import Decimal
import pandas as pd

constants = dict()

constants['lagphase_dividend'] = (Decimal('99')/Decimal('19')).ln()
constants['growthphase_dividend'] = Decimal('361').ln()
constants['growth-lag ratio'] = constants['growthphase_dividend'] / constants['lagphase_dividend']

root = os.path.dirname(__file__)
filepath = f'{root}/constants.csv'
pd.Series(constants).to_csv(filepath, header = False)

def get_constants():
    return pd.read_csv(filepath, header = None, index_col = 0).iloc[:, 0]


# NOTE: Represents the smallest nonzero float64 in NumPy.
smallest = finfo(float64).eps

zero_precise = Decimal('0')
ln_precise = {
    '1/100': Decimal('0.01').ln(),
    '1/99': (Decimal('1')/Decimal('99')).ln(),
    '1/20': Decimal('0.05').ln(),
    '1/19': (Decimal('1')/Decimal('19')).ln(),
    '1/10': Decimal('0.1').ln(),
    '1/9': (Decimal('1')/Decimal('9')).ln(),
    '1/2': Decimal('0.5').ln(),
    '0.9': Decimal('0.9').ln(),
    '0.95': Decimal('0.95').ln(),
    '0.99': Decimal('0.99').ln(),
    '1': zero_precise,
    '9': Decimal('9').ln(),
    '19': Decimal('19').ln(),
    '99': Decimal('99').ln() }

# NOTE: Represents ln(1/fraction - 1) for each fraction.
logistic_fraction_logs = {
    0.01: float64(ln_precise['99']),
    0.05: float64(ln_precise['19']),
    0.1: float64(ln_precise['9']),
    0.5: float64(zero_precise),
    0.9: float64(ln_precise['1/9']),
    0.95: float64(ln_precise['1/19']),
    0.99: float64(ln_precise['1/99']) }

# NOTE: Represents ln(1 - fraction) for each fraction.
exponential_fraction_logs = {
    0.01: float64(ln_precise['0.99']),
    0.05: float64(ln_precise['0.95']),
    0.1: float64(ln_precise['0.9']),
    0.5: float64(ln_precise['1/2']),
    0.9: float64(ln_precise['1/10']),
    0.95: float64(ln_precise['1/20']),
    0.99: float64(ln_precise['1/100']) }

ln99 = float64(ln_precise['99'])