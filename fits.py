import numpy as np
import pandas as pd
from collections import OrderedDict as OD
from inspect import signature, Signature

from optimizers import *
from KWW import *
from logistic import *
from exponential import *
from global_info import *
from styles import *

class Fits(list):
    class Fit():
        class CurveParameters():
            def __init__(self, curve, bounds, values, ignore = ('t',)):
                params = OD(signature(curve).parameters)
                for param in ignore: params.pop(param)
                sig = Signature(params.values())
                self.bounds = OD(sig.bind(*bounds).arguments)
                self.formatted_bounds = ', '.join( f'{variable}: {bound[0]} to {bound[1]}' for variable, bound in self.bounds.items() )
                self.values = OD(sig.bind(*tuple(values)).arguments)
        
        def __init__(self, group, category, mode, curve, sample, fit_input, fit_output, line, scatterplot = None, color = None, ax = None):
            self.curve, self.fit_output, self.mode, self.fit_input, self.line = curve, fit_output, mode, fit_input, line
            self.group, self.category, self.sample = group, category, sample
            self.ax = ax
            self.scatterplot = scatterplot
            
            line2d = line['line2d']
            self.color = line2d.get_color() if color is None else color
            self.initial_linestyle = line2d.get_linestyle()

            optimizer = mode.optimizer
            self.optimizer = optimizer
            self.objective_function = fit_input['func']

            if optimizer == diff_ev and self.objective_function == SSE:
                x_arg, y_arg, function_arg = fit_input['args']
                x_arg, y_arg = np.array(x_arg), np.array(y_arg)
                final_SSE = fit_output['fun']
                N = len(x_arg)
                MSE = final_SSE / N
                RMSE = np.sqrt(MSE)
                self.RMSE = RMSE
                self.RMSE_normalized = RMSE / y_arg.mean()
        
        def winner_setup(self):
            curve, fit_output, optimizer, fit_input, line, color = self.curve, self.fit_output, self.optimizer, self.fit_input, self.line, self.color

            curve_bounds = fit_input['bounds']
            if hasattr(optimizer, 'redundant_info'):
                redundant_info = optimizer.redundant_info
                unset = (key for key in redundant_info if redundant_info[key] is None)
                redundant_info.update({key: fit_input[key] for key in unset})
            optimizer_skip, output_skip = (set(optimizer.move_to_top[key].keys()) for key in ('optimizer', 'output'))
            optimizer_notes, output_notes = (optimizer.notes[key] for key in ('optimizer', 'output'))
            optimizer_conversions, output_conversions = (optimizer.conversions[key] for key in ('optimizer', 'output'))
            
            objective_columns, optimizer_columns, output_columns, curve_columns = OD(), OD(), OD(), OD()
            if optimizer == diff_ev:
                self.curve_parameters = self.CurveParameters(curve, curve_bounds, fit_output.x)

                x_arg, y_arg, function_arg = fit_input['args']
                x_arg, y_arg = np.array(x_arg), np.array(y_arg)
                objective_variables = '(' + ', '.join(('x', 'y', function_arg.__name__)) + ')'

                objective_columns.update({
                    'Objective function variable: objective_variables': objective_variables,
                    'objective_variables: x': x_arg,
                    'objective_variables: y': y_arg })

                output_skip.add('message')
                additional_info = OD()
                objective_function = self.objective_function
                if objective_function == SSE:
                    additional_info['Error (root-mean-square, RMSE)'] = self.RMSE
                    additional_info['Normalized error (RMSE/mean)'] = self.RMSE_normalized
                    output_skip.add('fun')
                output_columns.update(
                    **{
                        f'''Optimizer output: {(
                            key if key not in output_notes
                            else output_notes[key]
                        )}''': (
                            value if key not in output_conversions
                            else output_conversions[key](value) )
                        for key, value in fit_output.items() if key not in output_skip },
                    **additional_info )
            else:
                assert optimizer == nonlin_least_sq
                self.curve_parameters = self.CurveParameters(curve, zip(*curve_bounds), fit_output[0])
                
                covariance = fit_output[1]
                variance = np.diag(covariance)
                standard_deviation = np.sqrt(variance)
                output_columns.update({
                    'Optimizer output: optimal curve variables estimation': 'See curve variables.',
                    'Optimizer output: estimation covariance matrix': covariance,
                    'Estimation variance (diagonal of covariance; numpy.diag(covariance))': variance,
                    'Estimation error standard deviation (square root of variance; numpy.sqrt(variance))': standard_deviation })
                self.covariance, self.variance, self.standard_deviation = covariance, variance, standard_deviation
            optimizer_columns.update(
                **{ f'''Optimizer variable: {(
                        key if key not in optimizer_notes
                        else optimizer_notes[key]
                    )}''': (
                        value if key not in optimizer_conversions
                        else optimizer_conversions[key](value) )
                    for key, value in fit_input.items() if key not in optimizer_skip })
            curve_kwargs = self.curve_parameters.values
            styles = curve.styles
            def plot_mark(x, y, name, color):
                style = styles[name]['style']
                fill = style['fill']
                matchfill = 'match' in fill
                colors = { 'facecolors': fill if matchfill is False else color }
                if (outline := style['outline']) != 'match':
                    colors.update({ 'edgecolors': outline })
                scatter_kwargs = {
                    'marker': style['shape'],
                    **colors,
                    's': style['box_area'], 'color': color,
                    'zorder': 3, 'clip_on': False }
                if 'hatch' in style:
                    scatter_kwargs['hatch'] = style['hatch']
                scatter_data = { 'x': x, 'y': y }
                line['marks'].update({name: self.ax.scatter(**scatter_data, **scatter_kwargs)})
            curve_calculations = OD()
            def add_calculations(calculations, plot_curve, plot_kwargs):
                curve_calculations.update({
                    (styles[name]['title'] if name in styles
                        else (
                            abbreviations[name] if name in abbreviations
                            else name )): value
                    for entry in calculations.values()
                    for name, value in entry.items() })
                plot_calculations = calculations['on figure']
                for name in plot_calculations:
                    x = plot_calculations[name]
                    y = plot_curve(x, **plot_kwargs)
                    plot_mark(x, y, name, color)
            last_point = self.scatterplot['pathcollections']['use'].get_offsets()[-1]
            if curve == normalized_logistic:
                k_NL = curve_kwargs['k_NL']
                A = curve_kwargs['A']
                t_half = curve_kwargs['t_half']
                k_L = k_NL / A
                calculations = OD({
                    't_1%': logistic_percent_to_time(0.01, k_L, t_half),
                    't_5%': logistic_percent_to_time(0.05, k_L, t_half),
                    't_10%': logistic_percent_to_time(0.1, k_L, t_half),
                    't_50%': t_half,
                    't_90%': logistic_percent_to_time(0.9, k_L, t_half),
                    't_95%': logistic_percent_to_time(0.95, k_L, t_half),
                    't_99%': logistic_percent_to_time(0.99, k_L, t_half),
                    'lagtime': logistic_extrapolated_lagtime(k_L, t_half) })
                logistic_kwargs = {'y0': 0, 'A': A, 't_half': t_half, 'k_L': k_L}
                add_calculations(
                    OD({
                        'on figure': OD({key: calculations[key] for key in ('t_5%', 't_10%', 't_50%', 't_90%', 't_95%', 'lagtime')}),
                        'other': OD({
                            'Maximum rate (at 50% time)': logistic_max_rate(A, k_NL = k_NL),
                            'Value at 1% time': logistic(calculations['t_1%'], **logistic_kwargs),
                            'Value at 5% time': logistic(calculations['t_5%'], **logistic_kwargs),
                            'Value at 10% time': logistic(calculations['t_10%'], **logistic_kwargs),
                            'Value at 50% time': logistic(calculations['t_50%'], **logistic_kwargs),
                            'Value at 90% time': logistic(calculations['t_90%'], **logistic_kwargs),
                            'Value at 95% time': logistic(calculations['t_95%'], **logistic_kwargs),
                            'Value at 99% time': logistic(calculations['t_99%'], **logistic_kwargs),
                            'Value at lag time': logistic(calculations['lagtime%'], **logistic_kwargs),
                            'Value at time of last data point': logistic(last_point[0], **logistic_kwargs),
                            'Last data point value': last_point[1] }) }),
                    logistic, logistic_kwargs )
            elif curve == onepercent_anchored_logistic:
                k_NL = curve_kwargs['k_NL']
                A = curve_kwargs['A']
                k_L = k_NL / A
                t_1percent = curve_kwargs['t_1percent']
                t_half = t_1percent + ln(99) / k_L
                calculations = OD({
                    't_1%': t_1percent,
                    't_5%': logistic_percent_to_time(0.05, k_L, t_half),
                    't_10%': logistic_percent_to_time(0.1, k_L, t_half),
                    't_50%': t_half,
                    't_90%': logistic_percent_to_time(0.9, k_L, t_half),
                    't_95%': logistic_percent_to_time(0.95, k_L, t_half),
                    't_99%': logistic_percent_to_time(0.99, k_L, t_half) })
                logistic_kwargs = {'y0': 0, 'A': A, 't_half': t_half, 'k_L': k_L}
                add_calculations(
                    OD({
                        'on figure': calculations,
                        'other': OD({
                            'Maximum rate (at 50% time)': logistic_max_rate(A, k_NL = k_NL),
                            'Duration of lag phase (1% to 5%)': calculations['t_5%'] - calculations['t_1%'],
                            'Value at 1% time': logistic(calculations['t_1%'], **logistic_kwargs),
                            'Value at 5% time': logistic(calculations['t_5%'], **logistic_kwargs),
                            'Value at 10% time': logistic(calculations['t_10%'], **logistic_kwargs),
                            'Value at 50% time': logistic(calculations['t_50%'], **logistic_kwargs),
                            'Value at 90% time': logistic(calculations['t_90%'], **logistic_kwargs),
                            'Value at 95% time': logistic(calculations['t_95%'], **logistic_kwargs),
                            'Value at 99% time': logistic(calculations['t_99%'], **logistic_kwargs),
                            'Value at time of last data point': logistic(last_point[0], **logistic_kwargs),
                            'Last data point value': last_point[1] }) }),
                    logistic, logistic_kwargs)
            elif curve == normalized_exponential:
                t0 = curve_kwargs['t0']
                A = curve_kwargs['A']
                k_NE = curve_kwargs['k_NE']
                k_E = k_NE / A
                calculations = OD({
                    't0': t0,
                    't_1%': exponential_percent_to_time(0.01, k_E, t0),
                    't_5%': exponential_percent_to_time(0.05, k_E, t0),
                    't_10%': exponential_percent_to_time(0.1, k_E, t0),
                    't_50%': exponential_percent_to_time(0.5, k_E, t0),
                    't_90%': exponential_percent_to_time(0.9, k_E, t0),
                    't_95%': exponential_percent_to_time(0.95, k_E, t0),
                    't_99%': exponential_percent_to_time(0.99, k_E, t0) })
                exponential_kwargs = {'t0': t0, 'A': A, 'k_E': k_E}
                add_calculations(
                    OD({
                        'on figure': calculations,
                        'other': OD({
                            'Maximum rate (at start time)': exponential_max_rate(A, k_NE = k_NE),
                            'Value at start time': exponential(calculations['t0'], **exponential_kwargs),
                            'Value at 1% time': exponential(calculations['t_1%'], **exponential_kwargs),
                            'Value at 5% time': exponential(calculations['t_5%'], **exponential_kwargs),
                            'Value at 10% time': exponential(calculations['t_10%'], **exponential_kwargs),
                            'Value at 50% time': exponential(calculations['t_50%'], **exponential_kwargs),
                            'Value at 90% time': exponential(calculations['t_90%'], **exponential_kwargs),
                            'Value at 95% time': exponential(calculations['t_95%'], **exponential_kwargs),
                            'Value at 99% time': exponential(calculations['t_99%'], **exponential_kwargs),
                            'Value at time of last data point': exponential(last_point[0], **exponential_kwargs),
                            'Last data point value': last_point[1] }) }),
                    exponential, exponential_kwargs)
            elif curve == KWW_scaled:
                descaled_args = KWW_descaler(**curve_kwargs)
                descaled_t_inflection = KWW_inflection(*descaled_args)
                descaled_timeinfo = KWW_timeinfo(*descaled_args, descaled_t_inflection)
                t0, scale = curve_kwargs['t0'], curve_kwargs['scale']
                scaled_timeinfo = OD({ key: KWW_timescaler(value, scale, t0) for key, value in descaled_timeinfo.items() })
                calculations = scaled_timeinfo
                add_calculations(
                    OD({
                        'on figure': calculations,
                        'other': OD() }),
                    KWW_scaled, curve_kwargs)
            curve_columns.update(
                **{ 'Curve bounds': str(self.curve_parameters.formatted_bounds) },
                **{ f'Curve variable: {key}': str(value) for key, value in curve_kwargs.items() },
                **curve_calculations )
            self.full_columns = OD({
                **objective_columns,
                **optimizer_columns,
                **output_columns,
                **curve_columns })
            exclude = { 'Curve bounds' }
            self.minimal = OD({
                **{key: value for key, value in curve_columns.items() if key not in exclude} })
            
            self.fit_input = fit_input
            self.curve_bounds = curve_bounds
            self.curve_kwargs = curve_kwargs
            self.curve_calculations = curve_calculations

        def curve_report(self, *variable_names):
            "Reports the values of variables in self whose names are given."
            attributes = self.__dict__
            curve_calculations = self.curve_calculations
            def value_generator():
                for variable_name in variable_names:
                    if variable_name in curve_calculations:
                        yield curve_calculations[variable_name]
                        continue
                    if variable_name in attributes:
                        yield attributes[variable_name]
                        continue
                    found = False
                    for attribute in attributes.values():
                        if issubclass(type(attribute), dict) is False: continue
                        if variable_name in attribute:
                            yield attribute[variable_name]
                            found = True; break
                    assert found, f'Could not find {variable_name} in attributes {attributes.keys()} or their contents {tuple(item.keys() for item in attributes.values() if issubclass(type(item), dict))}'
            return tuple(value_generator())
        def __call__(self, minimal = False):
            datasource = self.full_columns
            if minimal: datasource = self.minimal
            return pd.DataFrame(
                data = tuple(datasource.values()),
                index = tuple(datasource.keys()),
                columns = ['Value'] )
    def __init__(self, group = None, category = None, mode = None, curve = None, sample = None, color = None, scatterplot = None, fits = None, ax = None):
        self.fit_args = (group, category, mode, curve, sample)
        self.group, self.category, self.mode, self.curve, self.sample = group, category, mode, curve, sample
        self.ax = ax
        if color is not None: self.color = color
        if scatterplot is None and all(_ is not None for _ in (group, category, sample)):
            scatterplot = self.scatterplots[group][category][sample]
        self.scatterplot = scatterplot
        self.fits = fits
        if fits is not None: super().__init__(fits)
    def add_fit(self, fit_input, fit_output, line):
        fit_args = self.fit_args
        fit = self.Fit(*fit_args, fit_input, fit_output, line, self.scatterplot, self.color, self.ax)
        self.append(fit)
        return fit
    def set_winner(self, index):
        winner = self[index]
        losers = self[:index] + self[index + 1:]
        self.winner = winner
        self.losers = losers
        return winner
    def separate_winner(self):
        'Returns a tuple of two new Fits objects: winner, losers.'
        assert hasattr(self, 'winner') and hasattr(self, 'losers'), 'set_winner must be called first.'
        winner, losers = self.winner, self.losers
        fits_args = (*self.fit_args, self.color)
        return (
            Fits(*fits_args, fits = (winner,)),
            Fits(*fits_args, fits = losers) )