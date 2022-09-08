from numpy import exp, log as ln
from styles import mark_styles
from constants_calculation import exponential_fraction_logs

def exponential(t, A, t0, k_E):
    return A * (1 - exp(-k_E * (t - t0)))
exponential.title = 'Non-normalized exponential'
exponential.title_lowercase = 'non-normalized exponential'
exponential.equation = 'A * (1 - exp(-k_E * (t - t0)))'
exponential.equation_notes = 'A is the vertical scale and\nt0 is the start time.'

def exponential_fraction_to_time(fraction, k_E, t0):
    if fraction in exponential_fraction_logs:
        fraction_log = exponential_fraction_logs[fraction]
    else:
        fraction_log = ln(1 - fraction)
    return (-1/k_E) * fraction_log + t0
def exponential_max_rate(A, *, k_E = None, k_NE = None):
    assert all((k_E is None, k_NE is None)) is False, 'Either k_E or k_NE must be specified.'
    assert all((k_E is not None, k_NE is not None)) is False, 'Either k_E or k_NE can be specified, not both.'
    if k_NE is None:
        return A * k_E
    return k_NE

def normalized_exponential(t, A, k_NE):
    t0 = 0
    k_E = k_NE / A
    return exponential(t, A, t0, k_E)
normalized_exponential.title = 'Exponential'
normalized_exponential.title_lowercase = 'exponential'
normalized_exponential.equation = 'A * (1 - exp(-(k_NE/A) * (t - t0)))'
normalized_exponential.equation_notes = 'k_NE describes the proportions\nof the curve independently from its scale (A).'
normalized_exponential.marks = {'t0', 't_1%', 't_5%', 't_10%', 't_50%', 't_90%', 't_95%', 't_99%'}
normalized_exponential.styles = mark_styles