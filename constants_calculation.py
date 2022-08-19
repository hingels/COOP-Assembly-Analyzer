import os
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

if __name__ == '__main__':
    retrieved = get_constants()
    print(retrieved)
    print('\n')
    print(retrieved['lagphase_dividend'])