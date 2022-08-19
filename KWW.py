from collections import OrderedDict as OD
from numpy import exp, log as ln
from styles import mark_styles

def KWW(t, tau, beta):
    "Kohlrausch-Williams-Watts compressed (beta>1) exponential function."
    return 1 - exp(-(t/tau)**beta)
KWW.title = 'Non-normalized Kohlrausch-Williams-Watts compressed (beta>1) exponential function'
KWW.title_lowercase = 'non-normalized Kohlrausch-Williams-Watts compressed (beta>1) exponential function'
KWW.equation = '1 - exp(-(t/tau)^beta)'

def KWW_inverse(KWW_output, tau, beta):
    return tau * (-ln(1 - KWW_output))**(1/beta)
def KWW_prime(t, tau, beta):
    return (beta/tau) * (t/tau)**(beta-1) * exp(-(t/tau)**beta)
def KWW_doubleprime(t, tau, beta):
    return -(beta/tau**2) * (t/tau)**(beta-2) * exp(-(t/tau)**beta) * (1 + beta*((t/tau)**beta - 1))
def KWW_inflection(tau, beta):
    "Returns inflection time."
    return tau * ((beta-1)/beta)**(1/beta)
def KWW_lagtime(tau, beta, t_inflection, leadtime = False):
    _KWW = KWW(t_inflection, tau, beta)
    _KWW_prime = KWW_prime(t_inflection, tau, beta)
    return ((leadtime - _KWW)/_KWW_prime) + t_inflection

def KWW_scaled(t, tau, slowbeta, scale, t0):
    fastbeta = slowbeta**4
    _KWW = lambda t: KWW(t, tau, fastbeta)
    return ( scale * _KWW((t-t0)/scale) )
KWW_scaled.title = 'Scaled KWW'
KWW_scaled.title_lowercase = 'scaled KWW'
KWW_scaled.equation = '( scale * KWW((t-t0)/scale, tau, slowbeta^4) )'
KWW_scaled.marks = {'t_5%', 't_10%', 't_90%', 't_95%', 'lagtime'}
KWW_scaled.styles = mark_styles

def KWW_descaler(tau, slowbeta, scale = None, t0 = None):
    "Converts KWW_scaled parameters to KWW parameters."
    fastbeta = slowbeta**4
    return tau, fastbeta
def KWW_timeinfo(tau, beta, t_inflection):
    t_percent = lambda percent: KWW_inverse(percent, tau, beta)
    return OD({
        't_5%': t_percent(0.05),
        't_10%': t_percent(0.1),
        't_90%': t_percent(0.9),
        't_95%': t_percent(0.95),
        'lagtime': KWW_lagtime(tau, beta, t_inflection) })
def KWW_timescaler(t, scale, t0):
    "Takes a time on an unscaled KWW curve and returns the corresponding time on the scaled curve."
    return scale * t + t0

if __name__ == '__main__':
    scaled_args = { 'tau': 1.2, 'slowbeta': 1.2, 'scale': 2.9, 't0': 0.4 }
    descaled_args = KWW_descaler(**scaled_args)

    t_inflection = KWW_inflection(*descaled_args)
    rate_max = KWW_prime(t_inflection, *descaled_args)

    print(*(f"{value['title']}: {value['value']}" for key, value in KWW_timeinfo(*descaled_args, t_inflection).items()), sep = '\n')