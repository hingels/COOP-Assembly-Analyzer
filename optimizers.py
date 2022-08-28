from collections import OrderedDict as OD
from scipy.optimize import differential_evolution as diff_ev, curve_fit as nonlin_least_sq

diff_ev.function = 'scipy.optimize.differential_evolution'
diff_ev.title = 'Differential Evolution Least-Squares Fit'
diff_ev.redundant_info = {
    'maxiter': None }
diff_ev.move_to_top = {
    'optimizer': OD({
        'func': 'See objective function.',
        'args': 'See objective function variable: objective_variables.',
        'maxiter': lambda *_: diff_ev.redundant_info['maxiter'],
        'bounds': 'See curve bounds.' }),
    'output': OD({
        'x': 'See curve variables.' }) }
diff_ev.notes = {
    'optimizer': {
        'func': 'func (objective function)',
        'args': 'args (objective function variables, or "arguments")',
        'maxiter': 'maxiter (maximum iterations allowed; for iterations performed, see optimizer output: nit)' },
    'output': {
        'x': 'x (estimation of optimal curve variables)',
        'jac': "jac (Jacobian matrix of diff_ev's objective function)",
        'fun': "fun (value of diff_ev's objective function)" } }
diff_ev.conversions = {
    'optimizer': dict(),
    'output': dict() }

nonlin_least_sq.function = 'scipy.optimize.curve_fit'
nonlin_least_sq.title = 'Nonlinear Least-Squares Regression'
nonlin_least_sq.move_to_top = {
    'optimizer': OD(),
    'output': OD() }
nonlin_least_sq.variable_notes = {
    'optimizer': {
        'f': 'f (function of curve)' },
    'output': {
        'nfev': 'nfev (number of objective function evaluations performed)',
        'nit': 'nit (number of iterations performed by the optimizer)' } }
nonlin_least_sq.conversions = {
    'optimizer': {
        'f': lambda func: func.__name__ },
    'output': dict() }


def SSR(curve_variables, *objective_variables):
    "Sum of squared residuals (SSR). A differential evolution curve-fitting objective function."
    x_data, y_data, f = objective_variables
    return ( (f(x_data, *curve_variables) - y_data)**2 ).sum()
SSR.title = 'Least Squares; minimizes sum of square errors (SSR)'

class Mode():
    def __init__(self, optimizer, title, objective_function = None):
        self.optimizer, self.title = optimizer, title
        if objective_function is not None: self.objective_function = objective_function

DE_leastsquares = Mode(diff_ev, 'DE, least-squares objective', SSR)
NLS = Mode(nonlin_least_sq, 'NLS')