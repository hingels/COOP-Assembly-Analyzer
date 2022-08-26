from optimizers import *

from KWW import *
from logistic import *
from exponential import *

special_samples = {
    'Averaged': {
        'lowercase': 'averaged'},
    'Combined': {
        'lowercase': 'combined'} }

end_default = 120

curves = onepercent_anchored_logistic, normalized_exponential
names_to_curves = {curve.__name__: curve for curve in curves}
modes = DE_leastsquares, NLS

experiment_optima = {'x': (None, None), 'y': (None, None)}
categories_per_experiment = {}
desks = {}
paths = {}
scatterplots = OD()
fits = OD()

abbreviations = {
    'RMSE_normalized': 'Normalized error (RMSE/mean)',
    'time': {'yy': r'%y', 'mm': r'%m', 'dd': r'%d', 'yyyy': r'%Y', 'hour': r'%I', 'HOUR': r'%H', 'min': r'%M', 'XM': r'%p'} }