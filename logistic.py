from numpy import exp, log as ln
from styles import mark_styles

def logistic(t, y0, A, t_half, k_L):
    return y0 + A/(1 + exp(-k_L*(t - t_half)))
logistic.title = 'Non-normalized logistic'
logistic.title_lowercase = 'non-normalized logistic'
logistic.equation = 'y0 + A/( 1 + exp(-k_L*(t - t_50%)) )'

def logistic_prime(t, y0, A, t_half, k_L):
    exponential = exp(-k_L*(t - t_half))
    return (A * k_L * exponential) / ((1 + exponential) ** 2)
def logistic_max_rate(A, *, k_L = None, k_NL = None):
    assert all((k_L is None, k_NL is None)) is False, 'Either k_L or k_NL must be specified.'
    assert all((k_L is not None, k_NL is not None)) is False, 'Either k_L or k_NL can be specified, not both.'
    if k_NL is None:
        return (A * k_L) / 4
    return k_NL / 4
def logistic_percent_to_time(percent, k_L, t_half):
    "Calculates the time at which a logistic function reaches a given percentage of its amplitude."
    return -ln(1/percent - 1)/k_L + t_half
def logistic_extrapolated_lagtime(k_L, t_half):
    """
    Calculates the time at which the tangent line to a logistic function's inflection point intercepts its lower asymptote.
    This is one method of calculating a lag time for the function.
    """
    return t_half - 2/k_L

def normalized_logistic(t, A, t_half, k_NL):
    k_L = k_NL / A
    return logistic(t, 0, A, t_half, k_L)
normalized_logistic.title = 'Logistic'
normalized_logistic.title_lowercase = 'logistic'
normalized_logistic.equation = 'A/( 1 + exp(-(k_NL/A)*(t - t_50%)) )'
normalized_logistic.equation_notes = 'k_NL describes the proportions\nof the curve independently from its scale (A).'
normalized_logistic.marks = {'t_5%', 't_10%', 't_90%', 't_95%', 'lagtime', 't_50%'}
normalized_logistic.styles = mark_styles

def onepercent_anchored_logistic(t, A, t_1percent, k_NL):
    k_L = k_NL / A
    offset = -ln(99) / k_L
    t_half = t_1percent - offset
    return logistic(t, 0, A, t_half, k_L)
onepercent_anchored_logistic.title = 'Logistic'
onepercent_anchored_logistic.title_lowercase = 'logistic'
onepercent_anchored_logistic.equation = 'A/( 1 + exp(-(t - t_1%)k_NL/A + ln(99)) )'
onepercent_anchored_logistic.equation_notes = 'k_NL describes the proportions\nof the curve independently from its scale (A).'
onepercent_anchored_logistic.marks = {'t_1%', 't_5%', 't_10%', 't_50%', 't_90%', 't_95%', 't_99%'}
onepercent_anchored_logistic.styles = mark_styles