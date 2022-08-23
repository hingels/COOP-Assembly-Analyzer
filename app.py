is_main = True if __name__ == '__main__' else False
from time import monotonic, perf_counter
initial_time = monotonic(), perf_counter()

from collections import OrderedDict as OD
import datetime
from functools import partial, reduce
from inspect import Signature, signature
import sys
import os
import shutil
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import warnings
import typing

from optimizers import *
from prepare_sheets import ConfigReader
from KWW import *
from logistic import *
from exponential import *
from styles import *
from constants_calculation import get_constants



end_default = 120

curves = onepercent_anchored_logistic, normalized_exponential
modes = DE_leastsquares, NLS

abbreviations = {
    'RMSE_normalized': 'Normalized error (RMSE/mean)',
    'time': {'yy': r'%y', 'mm': r'%m', 'dd': r'%d', 'yyyy': r'%Y', 'hour': r'%I', 'HOUR': r'%H', 'min': r'%M', 'XM': r'%p'} }

def get_report_info(key, paths):
    if key == 'Growth-lag ratio':
        return {key: get_constants()['growth-lag ratio']}
    elif key == 'Input files':
        return {
            'Data file': os.path.basename(paths['data_path']),
            'Configuration file': os.path.basename(paths['config_path']) }
    elif key == 'Time of report generation':
        return {key: datetime.datetime.now()}
    if key == '': return {}
    raise Exception(f'Could not recognize {key} in report config settings.')

special_samples = {
    'Averaged': {
        'lowercase': 'averaged'},
    'Combined': {
        'lowercase': 'combined'} }

experiment_optima = {'x': (None, None), 'y': (None, None)}
categories_per_experiment = {}
desks = {}
paths = {}
scatterplots = OD()
fits = OD()


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
        
        def __init__(self, group, category, mode, curve, sample, fit_input, fit_output, line, scatterplot = None, color = None):
            self.curve, self.fit_output, self.mode, self.fit_input, self.line = curve, fit_output, mode, fit_input, line
            self.group, self.category, self.sample = group, category, sample
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
                line['marks'].update({name: plt.scatter(**scatter_data, **scatter_kwargs)})
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
    def __init__(self, group = None, category = None, mode = None, curve = None, sample = None, color = None, scatterplot = None, fits = None):
        self.fit_args = (group, category, mode, curve, sample)
        self.group, self.category, self.mode, self.curve, self.sample = group, category, mode, curve, sample
        if color is not None: self.color = color
        if scatterplot is None and all(_ is not None for _ in (group, category, sample)):
            scatterplot = scatterplots[group][category][sample]
        self.scatterplot = scatterplot
        self.fits = fits
        if fits is not None: super().__init__(fits)
    def add_fit(self, fit_input, fit_output, line):
        fit_args = self.fit_args
        fit = self.Fit(*fit_args, fit_input, fit_output, line, self.scatterplot, self.color)
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


class Desk():
    figure_box = None
    def __init__(self, group):
        self.group = group
        self.legend_handles_labels = OD()
        self.legend_categories = OD()
        self.figure = {}
        self.averaged_samples = {}
        self.standard_deviations = {}
        self.DE_leastsquares_averaged_lines = {}
        self.group_optima = {'x': (None, None), 'y': (None, None)}
        self.category_optima = {}
        self.sample_optima = {}
    def setup(self):
        group = self.group

        paths = self.paths[group]
        self.figures_path = paths['figures_path']
        self.groupfolder_paths = paths['groupfolder_path']
        if 'candidates' in paths:
            candidates_paths = paths['candidates']
            self.candidates_individual = candidates_paths['individual']
            self.candidates_special = candidates_paths['special']
        if 'winners' in paths:
            winners_paths = paths['winners']
            self.winners_individual = winners_paths['individual']
            self.winners_special = winners_paths['special']
        
        figure_info = self.figure
        self.fig, self.ax, self.fig_number = figure_info['figure'], figure_info['axes'], figure_info['number']
    def show_all(self, fits, fits_visible = True, marks_visible = True, legend_visible = True, errorbars_visible = False, show_categories = True):
        'Shows or hides all lines and scatterplots in the given Fits object.'
        show_marks_on_legend = self.show_marks_on_legend
        scatterplot = fits.scatterplot
        use_fits_scatterplot = (scatterplot is not None)
        show_ignored = False
        if use_fits_scatterplot:
            if scatterplot['pathcollections']['ignore'] is not None: show_ignored = True
            try: print('Showing', fits.group, fits.category, fits.sample, 'and setting errorbar visibility to', errorbars_visible, 'affecting', errorbars_text)
            except: pass
            show_scatterplot(scatterplot, fits_visible)
        if not marks_visible or not fits_visible: show_marks_on_legend(marks_visible = False, legend_visible = legend_visible, errorbars_visible = errorbars_visible, show_categories = show_categories, show_ignored = show_ignored)
        if not fits_visible:
            for fit in fits:
                show(fit.line, False)
                if not use_fits_scatterplot:
                    fit_scatterplot = fit.scatterplot
                    if fit_scatterplot['pathcollections']['ignore'] is not None: show_ignored = True
                    show_scatterplot(fit_scatterplot, False)
            return
        if hasattr(fits, 'color'):
            for fit in fits:
                show(fit.line, linestyle = fit.initial_linestyle, marks_visible = marks_visible)
                if not use_fits_scatterplot:
                    fit_scatterplot = fit.scatterplot
                    if fit_scatterplot['pathcollections']['ignore'] is not None: show_ignored = True
                    show_scatterplot(fit_scatterplot)
            if marks_visible: show_marks_on_legend(fits.curve, legend_visible = legend_visible, errorbars_visible = errorbars_visible, show_categories = show_categories, show_ignored = show_ignored, color = fits.color)
            return
        curve = fits.curve
        for fit in fits:
            show(fit.line, linestyle = fit.initial_linestyle, marks_visible = marks_visible)
            if not use_fits_scatterplot:
                fit_scatterplot = fit.scatterplot
                if fit_scatterplot['pathcollections']['ignore'] is not None: show_ignored = True
                show_scatterplot(fit_scatterplot)
        if marks_visible: show_marks_on_legend(curve, legend_visible = legend_visible, errorbars_visible = errorbars_visible, show_categories = show_categories, show_ignored = show_ignored)
    
    def get_capture_filename(self, curve, category = None, sample = None, special = False, autozoom = False, lens = 1, all_fits = False, all_categories = False):
        out = []
        if special:
            if category is not None: out.append(f'{curve.title}_{category}')
            else: out.append(curve.title)
        if sample is not None: out.append(sample)
        if all_fits: out.append('All fits')
        if autozoom: out.append('Autozoom')
        else: out.append(f'ZoomX{lens}')
        if all_categories: out.append('All')
        return '_'.join(out) + '.png'
    def capture(self, fits: Fits, folder = None, filename = None, lens = 1, autozoom = False, marks_visible = True, legend_visible = True, errorbars_visible = False, all_fits = False, all_categories = False, title_info = None):
        ax, names, show_all, zoom, offscreen_marks_transparent = self.ax, self.names, self.show_all, self.zoom, self.offscreen_marks_transparent
        show_all_args = {'marks_visible': marks_visible, 'legend_visible': legend_visible, 'errorbars_visible': errorbars_visible, 'show_categories': all_categories}

        plt.sca(ax)
        if autozoom: assert lens == 1, 'Lens must be 1 if autozoom is enabled.'
        else: folder += f'/Zoom x{lens}'
        os.makedirs(folder, exist_ok = True)
        fits_title = f'{fits.curve.title_lowercase} fit' if not all_fits else f'all {fits.curve.title_lowercase} fits'
        categories_title = f'{fits.category}' if not all_categories else f"all {names['category']['plural']}"
        if title_info is None:
            ax.set_title(f"{names['figure title base']}, {self.group},\n{categories_title} ({fits_title})")
        else:
            ax.set_title(f"{names['figure title base']}, {self.group},\n{categories_title}, {title_info} ({fits_title})")

        show_all(fits, **show_all_args)
        if autozoom:
            if fits.category is not None:
                category = fits.category
                optima = self.category_optima[category]
            else:
                optima = self.group_optima
            bottom, top = optima['y']
            left, right = optima['x']
            zoom(bottom, top, left, right)
        else:
            zoom(lens = lens, margins = Margins(bottom = 0), bottom = 0, zoom_axis = 0)
        offscreen_marks_transparent(fits)
        plt.savefig(f'{folder}/{filename}', dpi = 300)
        show_all(fits, False, **show_all_args)
    def capture_all(self, zoom_settings, capture_args, filename_args, presetzoom_folder, autozoom_folder):
        capture, get_capture_filename = self.capture, self.get_capture_filename
        for setting in zoom_settings:
            if setting == 'autozoom':
                capture(autozoom = True, folder = autozoom_folder, filename = get_capture_filename(autozoom = True, **filename_args), **capture_args)
            else:
                assert setting == 'zoom', f'{setting} in {zoom_settings=} is unrecognized.'
                for lens in zoom_settings[setting]:
                    lens = float(lens)
                    capture(lens = lens, folder = presetzoom_folder, filename = get_capture_filename(lens = lens, **filename_args), **capture_args)
    
    def fit_diff_ev_least_sq(self, curve, bounds, x, y, category, sample, other_args, color = 'black', iterations = None):
        iterations, save_candidates, zoom_settings, group, fits, lines_xdata = self.iterations, self.save_candidates, self.zoom_settings, self.group, self.fits, self.lines_xdata
        candidates_individual, winners_individual = self.candidates_individual, self.winners_individual
        capture_all, show_all = self.capture_all, self.show_all
        if iterations is None:
            iterations = self.iterations
        
        
        fit_input = OD({ 'func': SSE, 'args': (x, y, curve), 'bounds': bounds, **other_args })
        
        special = sample in special_samples
        if special:
            candidates_special, winners_special = self.candidates_special[sample], self.winners_special[sample]
        
        curve_name = curve.title
        curve_category = f'{curve_name}_{category}'
        
        candidates = Fits(group, category, DE_leastsquares, curve, sample, color)
        for iteration in range(iterations):
            fit_output = diff_ev(**fit_input)
            y_model = curve(lines_xdata, *fit_output.x)
            fitline_candidate = {
                'line2d': plt.plot(lines_xdata, y_model, color = color)[0],
                'marks': {} }
            candidates.add_fit(fit_input, fit_output, fitline_candidate)
        best = None
        for index, fit in enumerate(candidates):
            if best is None or fit.RMSE < candidates[best].RMSE:
                best = index
        candidates.set_winner(best).winner_setup()

        winner_fits, losers_fits = candidates.separate_winner()

        fit_output = candidates.winner
        DE_leastsquares_fits = fits[group][category][DE_leastsquares]
        DE_leastsquares_fits[curve][sample] = fit_output

        def delete_losers():
            for fit in losers_fits:
                line = fit.line
                line['line2d'].remove()
                for mark in line['marks'].values(): mark.remove()
                del line
        
        if ((not self.save_averaged and sample == 'Averaged') or
            (not self.save_combined and sample == 'Combined') ):
            show_all(candidates, False)
            delete_losers()
            return fit_output
        
        filename_args = {'curve': curve, 'category': category, 'sample': sample, 'special': special}
        
        if save_candidates:
            capture_args = {'fits': candidates, 'marks_visible': False, 'legend_visible': False}
            if special: capture_args['title_info'] = special_samples[sample]['lowercase']
            capture_args['errorbars_visible'] = (sample == 'Averaged')
            presetzoom_folder = capture_args['folder'] = f'{candidates_individual["Preset"]}/{curve_category}' if not special else f'{candidates_special["Preset"]}'
            autozoom_folder = capture_args['folder'] = f'{candidates_individual["Autozoom"]}/{curve_category}' if not special else f'{candidates_special["Autozoom"]}'
            capture_all(zoom_settings, capture_args, filename_args, presetzoom_folder, autozoom_folder)
        
        delete_losers()

        capture_args = {'fits': winner_fits}
        if special: capture_args['title_info'] = special_samples[sample]['lowercase']
        capture_args['errorbars_visible'] = (sample == 'Averaged')
        presetzoom_folder = f'{winners_individual["Preset"]}/{curve_category}' if not special else f'{winners_special["Preset"]}'
        autozoom_folder = f'{winners_individual["Autozoom"]}/{curve_category}' if not special else f'{winners_special["Autozoom"]}'
        capture_all(zoom_settings, capture_args, filename_args, presetzoom_folder, autozoom_folder)
        
        return fit_output
        
    def offscreen_marks_transparent(self, fits: Fits):
        """
        Makes any mark in "fits" transparent if it's out of bounds.
        """
        ax = self.ax
        for fit in fits:
            line = fit.line
            marks = line['marks']
            for mark in marks.values():
                x, y = tuple(mark.get_offsets()[0])
                left, right = ax.get_xlim()
                bottom, top = ax.get_ylim()
                if any((x < left, x > right, y < bottom, y > top)):
                    mark.set_alpha(0.5)
                else:
                    mark.set_alpha(1)
    
    def apply_margins(self, bottom, top, left, right, margins):
        margins_type = type(margins)
        if margins_type is Margins:
            bottom_margin, top_margin, left_margin, right_margin = margins.bottom, margins.top, margins.left, margins.right
        elif issubclass(margins_type, tuple):
            bottom_margin, top_margin, left_margin, right_margin = margins
        else:
            assert issubclass(margins_type, dict), f'Cannot recognize type {margins_type} of {margins=}.'
            bottom_margin, top_margin, left_margin, right_margin = margins['bottom'], margins['top'], margins['left'], margins['right']
        height, width = (top - bottom), (right - left)
        bottom -= height * bottom_margin
        top += height * top_margin
        left -= width * left_margin
        right += width * right_margin
        return bottom, top, left, right
    
    def zoom(self, bottom = None, top = None, left = None, right = None, lens = 1, margins = None, zoom_axis = 0.5):
        ax = self.ax
        if margins is None: margins = default_margins
        
        x_min, x_max = experiment_optima['x']
        if left is None: left = x_min
        if right is None: right = x_max
        y_min, y_max = experiment_optima['y']
        if bottom is None: bottom = y_min
        if top is None: top = y_max
        
        old_height = top - bottom
        new_height = old_height / lens
        delta = old_height - new_height
        bottom += delta * zoom_axis
        top -= delta * (1 - zoom_axis)
        bottom, top, left, right = self.apply_margins(
            bottom, top, left, right,
            margins = margins )
        
        ax.set_ylim(bottom, top)
        if ax.get_xlim() != (left, right):
            ax.set_xlim(left, right)
        
        return bottom, top, left, right

    def set_legend(self, hidden = None, visible = True, errorbars_visible = True, show_categories = True, color = None):
        ax, legend_handles_labels, legend_categories = self.ax, self.legend_handles_labels, self.legend_categories
        figure_box = self.figure_box
        if figure_box is None:
            self.figure_box_initial = ax.get_position()
            figure_box, legend_box = legend_sizing(ax)
            default_legend_kwargs['bbox_to_anchor'] = legend_box
            self.figure_box = figure_box
        
        errorbars_visible = errorbars_visible and visible

        if hidden is not None:
            new_legend_handles_labels = OD()
            no_data = OD({ 'x': tuple(), 'y': tuple() })
            no_line = { 'linestyle': 'None', 'color': 'black' }
            for style_dict in (*mark_styles.values(), scatter_styles['default']['ignore']):
                label, style = style_dict['title'], style_dict['style']

                invisible_label = label if label.startswith('_') else f'_{label}'
                visible_label = label[1:] if label.startswith('_') else label
                newlabel = invisible_label if label in hidden else visible_label
                
                is_new = False
                if visible_label in legend_handles_labels:
                    handle = legend_handles_labels[visible_label]
                elif invisible_label in legend_handles_labels:
                    handle = legend_handles_labels[invisible_label]
                else:
                    handle = None
                    is_new = True

                fill, outline_color = style['fill'], style['outline']
                if outline_color == 'match': outline_color = 'black'
                matchfill = 'match' in fill
                hatch = 'hatch' in style
                if matchfill: assert hatch is False, 'Marks cannot use matched fill and hatching at the same time.'
                
                if hatch:
                    if is_new:
                        marker = { 'marker': style['shape'], 's': style['box_area'], 'c': 'white', 'edgecolors': outline_color, 'hatch': style['hatch'] }
                        handle = plt.scatter(**no_data, **marker)
                    new_legend_handles_labels[newlabel] = handle
                    continue
                
                marker = { 'marker': style['shape'], 'markersize': np.sqrt(style['box_area']), 'markeredgecolor': outline_color, 'fillstyle': 'full' }
                if matchfill:
                    if color is None: marker.update(legend_match_combos[fill])
                    else: marker.update({ 'markerfacecolor': color })
                else: marker.update({ 'markerfacecolor': fill })
                
                if is_new:
                    new_legend_handles_labels[newlabel], = plt.plot(*no_data.values(), **no_line, **marker)
                    continue
                assert type(handle) is Line2D
                handle.set(**marker)
                new_legend_handles_labels[newlabel] = handle
            if show_categories: new_legend_handles_labels.update(legend_categories)
            legend_handles_labels.clear()
            legend_handles_labels.update(new_legend_handles_labels)
        with warnings.catch_warnings(record=True):
            labels_handles = tuple(map(list, zip(*legend_handles_labels.items())))
            if len(labels_handles) != 0:
                labels, handles = labels_handles
                legend = ax.legend(handles, labels, **default_legend_kwargs)
            else:
                legend = ax.legend(**default_legend_kwargs)
            legend.set_visible(visible)
            if hasattr(self, 'errorbars_text'): self.errorbars_text.set_visible(errorbars_visible)
            if visible: ax.set_position(figure_box)
            else: ax.set_position(self.figure_box_initial)
    def show_marks_on_legend(self, curve = None, marks_visible = True, legend_visible = True, errorbars_visible = False, show_categories = True, show_ignored = True, color = None):
        if curve is not None:
            if hasattr(curve, 'marks') is False: marks_visible = False
            styles = curve.styles
            show = curve.marks if marks_visible else []
        else:
            styles = mark_styles
            show = styles.keys() if marks_visible else []
        if not show_ignored: styles = styles | {'ignore': scatter_styles['default']['ignore']}
        hidden = tuple(style['title'] for mark, style in styles.items() if mark not in show)
        self.set_legend(hidden = hidden, visible = legend_visible, errorbars_visible = errorbars_visible, show_categories = show_categories, color = color)
    def get_winner_lines(self, mode, sample = None):
        group = self.group
        groupfits = self.fits[group]
        return OD({
            curve: OD({
                category: ((
                    (curve_dict := category_dict[mode][curve])[(
                        sample if sample is not None
                        else 'Averaged' if 'Averaged' in curve_dict
                        else 'Combined'
                    )].line) )
                for category, category_dict in groupfits.items() })
            for curve in curves })

def update_optima(optima, new_values):
    for dimension in new_values:
        old_min, old_max = optima[dimension]
        new_min, new_max = new_values[dimension]
        replace_min, replace_max = False, False
        if old_min is None or new_min < old_min: replace_min = True
        if old_max is None or new_max > old_max: replace_max = True
        optima[dimension] = (new_min if replace_min else old_min, new_max if replace_max else old_max)

def prepare_groups(experiment, groups, reader):
    global experiment_optima, scatterplots, fits
    categories = categories_per_experiment[experiment]
    for group in groups:
        desk = Desk(group)
        desks[group] = desk
        groupcategories = categories[group]
        time_dataframe = groupcategories.pop('independent_var_column')
        desk.time_dataframe = time_dataframe
        desk.categories = groupcategories
        group_optima = desk.group_optima

        data = groupcategories
        time_values = time_dataframe.values
        averaged_samples = { category: data[category]['data'].mean(axis='columns').values for category in data }
        desk.averaged_samples.update(averaged_samples)

        desk.samples = {
            category: OD({
                f'Sample {sample_index + 1}': data[category]['data'][sample].values
                for sample_index, sample in enumerate(data[category]['data'].columns) })
            for category in data }
        
        desk.config_per_category = { category: CategoryConfig(group, category, data[category]['category_config'], reader) for category in data }

        desk.x = {}
        desk.lines_xdata = None

        desk.y = {}
        desk.errorbars = {}

        fits[group] = {}
        scatterplots[group] = {}
        for category in groupcategories:
            fits[group][category] = {
                mode: {
                    curve: {}
                    for curve in curves }
                for mode in modes }
            scatterplots[group][category] = {}

            config = desk.config_per_category[category]
            end = config.get_setting('end')
            if end is None: end = end_default
            subtract_initial = config.get_setting('subtract_initial')
            subtract_min = config.get_setting('subtract_min')
            if subtract_min is None: subtract_min = True
            combine_samples = config.get_setting('combine_samples')
            
            x_data = time_values

            samples = desk.samples[category]
            if combine_samples is None: combine_samples = True
            if combine_samples:
                population = len(samples)
                combined_x_data = np.array(
                    reduce(
                        lambda a, b: (*a, *b),
                        ((value,)*population for value in x_data) ))
                samples['Combined'] = np.array(tuple(value for time in zip(*samples.values()) for value in time))
            samples['Averaged'] = averaged_samples[category]

            def find_end(data):
                end_index = -1
                if end is not None:
                    for index, value in enumerate(data):
                        end_index = index
                        if value > end: break
                return end_index
            end_index = find_end(x_data)
            def get_x(x_data, end_index):
                x_use = x_data[:end_index]
                x_ignore = x_data[end_index:] if end is not None else []
                x_min, x_max = x_use[0], x_use[-1]
                x_standard_deviation = np.std(x_use)
                x_window = x_min - x_standard_deviation, x_max + x_standard_deviation
                return {
                    'all': x_data,
                    'use': x_use,
                    'ignore': x_ignore,
                    'min': x_min,
                    'max': x_max,
                    'standard_deviation': x_standard_deviation,
                    'window': x_window }
            default_x = get_x(x_data, end_index)
            category_x_min, category_x_max = default_x['min'], default_x['max']

            x = {}
            desk.x[category] = x
            
            y = {}
            desk.y[category] = y

            errorbars = {}
            desk.errorbars[category] = errorbars
            
            category_optima = {'x': (category_x_min, category_x_max), 'y': (None, None)}
            desk.category_optima[category] = category_optima

            all_sample_optima = {}
            desk.sample_optima[category] = all_sample_optima
            
            standard_deviations = data[category]['data'].std(axis=1)
            desk.standard_deviations[category] = standard_deviations
            
            for sample, y_data in samples.items():
                if subtract_initial:
                    y_data -= y_data[0]
                elif subtract_min:
                    y_data -= min(y_data)
                
                index = end_index
                if combine_samples and sample == 'Combined':
                    index *= population
                    x[sample] = get_x(combined_x_data, index)
                else:
                    x[sample] = default_x
                
                y_use = y_data[:index]
                y_ignore = y_data[index:] if end is not None else []
                
                sample_y_min, sample_y_max = min(y_use), max(y_use)
                sample_y_standard_deviation = np.std(y_use)
                y_window = sample_y_min - sample_y_standard_deviation, sample_y_max + sample_y_standard_deviation

                y[sample] = {
                    'all': y_data,
                    'use': y_use,
                    'ignore': y_ignore,
                    'min': sample_y_min,
                    'max': sample_y_max,
                    'standard_deviation': sample_y_standard_deviation,
                    'window': y_window }
                
                errorbars_use = standard_deviations[:index]
                errorbars_ignore = standard_deviations[index:] if end is not None else []
                errorbars[sample] = {
                    'use': errorbars_use,
                    'ignore': errorbars_ignore }

                sample_optima = {'x': (category_x_min, category_x_max), 'y': (sample_y_min, sample_y_max)}
                all_sample_optima[sample] = sample_optima
                
                update_optima(category_optima, sample_optima)
            update_optima(group_optima, category_optima)
        update_optima(experiment_optima, group_optima)
    for group in groups:
        desk = desks[group]
        left, right = experiment_optima['x']
        bottom, top = experiment_optima['y']
        bottom, top, left, right = desk.apply_margins(
            bottom, top, left, right,
            margins = default_margins )
        desk.lines_xdata = np.linspace(left, right, round(right - left))
    
def show(line, visible = True, /, linestyle = None, marks_visible = True):
    if not visible: marks_visible = False
    line2d, marks = line['line2d'], line['marks']
    active_linestyle = '-' if linestyle is None else linestyle
    line2d.set_linestyle(active_linestyle if visible else 'None')
    for mark in marks.values():
        mark.set_visible(marks_visible)
def show_scatterplot(scatterplot, visible = True):
    for pathcollection in scatterplot['pathcollections'].values():
        if pathcollection is None: continue
        pathcollection.set_visible(visible)
    if 'errorbarcontainers' not in scatterplot: return
    for errorbarcontainer in scatterplot['errorbarcontainers']:
        for component_tuple in errorbarcontainer.lines:
            if component_tuple is None: continue
            for component in component_tuple:
                component.set_visible(visible)

def get_files(path, extensions, filename = r'.*'):
    if type(extensions) is not tuple: extensions = (extensions,)
    found = False
    output_filepath = None
    for extension in extensions:
        if extension[0] == '.': extension = extension[1:]
        files = (
            element
            for element in os.listdir(path)
            if os.path.isfile(os.path.join(path, element)) )
        search_obj = re.compile(rf'{filename}\.{extension}$')
        for file in files:
            if search_obj.search(file) is None: continue
            assert found is False, f'More than one file with extension{ f" {extension}" if len(extensions) == 1 else f"s {extensions}" } was found in {path}. Please remove all but one.'
            output_filepath = f'{path}/{file}'
            found = True
    assert type(output_filepath) is str, f'No .{" or .".join(extensions)} files were found in {path}.'
    return output_filepath

def legend_sizing(ax):
    # https://stackoverflow.com/questions/4700614/how-to-put-the-legend-outside-the-plot/43439132#43439132
    window_origin, window_space = {'x': 0, 'y': 0}, {'x': 1, 'y': 1}
    ax_box = ax.get_position()
    ax_origin = OD({'x': ax_box.x0, 'y': ax_box.y0})
    ax_space_initial = OD({'x': ax_box.width, 'y': ax_box.height})
    window_margins_initial = {'right': (window_space['x'] - (ax_origin['x'] + ax_space_initial['x']))}
    scaling_factors = {'x': 0.8, 'y': 1}
    ax_space_scaled = {'x': (scaling_factors['x'] * ax_space_initial['x']), 'y': (scaling_factors['y'] * ax_space_initial['y'])}
    window_margins_final = {'right': (window_margins_initial['right'] + (ax_space_initial['x'] - ax_space_scaled['x']))}
    windowspace_to_axspace = lambda width, height: {'width': width/ax_space_scaled['x'], 'height': height/ax_space_scaled['y']}
    new_legend_dimensions = windowspace_to_axspace( window_margins_final['right'], 1 )
    figure_box = [*ax_origin.values(), *ax_space_scaled.values()]
    legend_box = (1, 0.2, new_legend_dimensions['width'], new_legend_dimensions['height'])
    return figure_box, legend_box


class CategoryConfig():
    category_config_types = {
        'end': float,
        'max_as_max': bool,
        'subtract_initial': bool,
        'subtract_min': bool,
        'combine_samples': bool }
    def __init__(self, group, category, config, reader):
        self.group, self.category, self.config, self.reader = group, category, config, reader
        self.sheet_names_as_groups = reader.sheet_names_as_groups
        
        parse_value = self.parse_value
        self.defaults = {
            setting: parse_value(setting, value)
            for setting, value in reader.category_config_defaults.items() }
    def parse_value(self, setting, value):
        config_type = self.category_config_types[setting]
        if config_type is bool:
            assert value in ('True', 'False'), f'Value {value} is not recognized as a boolean value. Must be True or False.'
            value = (value == 'True')
        else:
            try:
                value = config_type(value)
            except ValueError:
                raise ValueError(f'"{setting}" could not be converted to type {config_type}. Given {value=}.')
        return value
    def get_setting(self, setting):
        config, group, sheet_names_as_groups = self.config, self.group, self.sheet_names_as_groups
        parse_value = self.parse_value
        if setting not in config: return self.defaults[setting]
        value = config[setting]
        if not sheet_names_as_groups:
            assert issubclass(type(value), dict) is False, f'Invalid syntax: when "Use sheet names as groups" is disabled, "{setting}" must be specified as a number. Given {value=}.'
            return parse_value(setting, value)
        assert issubclass(type(value), dict), f'Invalid syntax: when "Use sheet names as groups" is enabled, "{setting}" must be specified in the following format: {setting}=[group1=value1; group2=value2 ... ]. Given {value=}.'
        if group not in value: return self.defaults[setting]
        return parse_value(setting, value[group])



def main():
    def setup_files(root):
        input_path = f'{root}/Input'
        return input_path, get_files(input_path, ('csv', 'xlsx')), get_files(input_path, 'md')
    try:
        root = os.path.dirname(sys.executable)
        input_path, data_path, config_path = setup_files(root)
    except:
        root = os.path.dirname(os.path.abspath(__file__))
        input_path, data_path, config_path = setup_files(root)
    reader = ConfigReader(config_path)
    assert reader.iterations.isdigit()
    iterations = int(reader.iterations)
    names = {
        'figure title base': reader.figure_name,
        'group': {
            'singular': reader.group_name,
            'plural': reader.groups_name },
        'category': {
            'singular': reader.category_name,
            'plural': reader.categories_name } }
    save_candidates = reader.save_candidates
    save_all_fits, save_averaged, save_combined = reader.save_all_fits, reader.save_averaged, reader.save_combined
    experiment = reader.selected_experiment
    plt.rcParams['font.family'] = reader.font if hasattr(reader, 'font') else 'Arial'
    zoom_settings = reader.zoom_settings

    output_filename_unformatted = partial(reader.output_filename.format)
    current_time = datetime.datetime.now()
    for abbreviation, meaning in abbreviations['time'].items():
        output_filename_unformatted = partial(output_filename_unformatted, **{abbreviation: current_time.strftime(meaning)})
    output_filename_base = output_filename_unformatted()
    
    if os.path.isdir(f'{root}/Output') is False:
        os.mkdir(f'{root}/Output')
    
    path_base = '{root}/Output/{name}{i}'
    format_kwargs = {'root': root, 'name': output_filename_base, 'i': ''}
    while os.path.isdir(path_base.format(**format_kwargs)):
        format_kwargs['i'] = 2 if (i := format_kwargs['i']) == '' else (i + 1)
    path_base = path_base.format(**format_kwargs)
    os.mkdir(path_base)

    input_copy = f'{path_base}/Input'
    os.mkdir(input_copy)
    shutil.copy2(data_path, input_copy)
    shutil.copy2(config_path, input_copy)
    
    output_path_base = f'{path_base}/Output'
    os.mkdir(output_path_base)
    
    paths.update({'root': root, 'input_path': input_path, 'data_path': data_path, 'config_path': config_path, 'output_filename_base': output_filename_base, 'path_base': path_base, 'input_copy': input_copy, 'output_path_base': output_path_base })

    _globals = globals()
    def get_info_rows(report):
        keys = list(key.strip() for key in report['info'].split(';'))
        defaults = ('Input files', 'Time of report generation')
        for default in defaults:
            if default not in keys: keys.append(default)
        for key in keys:
            for name, value in get_report_info(key, paths).items():
                yield name, value
    def get_curve_reports():
        for report in reader.reports.values():
            yield OD({
                'curve': _globals[report['curve']],
                'variable_names': tuple(
                    variable.strip()
                    for variable in report['variable_names'].split(';') ),
                **({
                    'info': OD(get_info_rows(report))}
                    if 'info' in report else {} ) })
    curve_reports = get_curve_reports()

    categories_per_experiment.update(reader.read_data(data_path))
    groups = categories_per_experiment[experiment].keys()

    for key, value in { 'iterations': iterations, 'save_candidates': save_candidates, 'save_all_fits': save_all_fits, 'save_averaged': save_averaged, 'save_combined': save_combined, 'zoom_settings': zoom_settings, 'groups': groups, 'fits': fits, 'names': names, 'scatterplots': scatterplots, 'paths': paths, 'desks': desks }.items():
        setattr(Desk, key, value)
    prepare_groups(experiment, groups, reader)
    
    for group in groups:
        print(f'\n\nGROUP {group}\n')

        def figure_paths(path):
            autozoom_path = f'{path}/Automatic zoom'
            presetzoom_path = f'{path}/Preset zoom'
            return {'Autozoom': autozoom_path, 'Preset': presetzoom_path}
        def special_paths(path):
            averaged_paths = figure_paths(f'{path}/Averaged samples')
            combined_paths = figure_paths(f'{path}/Combined samples')
            return {'Averaged': averaged_paths, 'Combined': combined_paths}
        
        groupfolder_path = f'{output_path_base}/{group}'
        figures_path = f'{groupfolder_path}/Figures'

        candidates_path = f'{figures_path}/Candidates'
        candidates_individual = figure_paths(f'{candidates_path}/Individual samples')
        candidates_special = special_paths(f'{candidates_path}/All samples')

        winners_path = f'{figures_path}/Winners' if save_candidates else figures_path
        winners_individual = figure_paths(f'{winners_path}/Individual samples')
        winners_special = special_paths(f'{winners_path}/All samples')
        winners_all = figure_paths(f'{winners_path}/All samples/All fits')

        paths[group] = {
            'groupfolder_path': groupfolder_path,
            'figures_path': figures_path,
            'candidates': {
                'individual': candidates_individual,
                'special': candidates_special },
            'winners': {
                'individual': winners_individual,
                'special': winners_special,
                'all': winners_all } }

        group_report_filename = f'{group} report'
        
        fig, ax = plt.subplots()
        plt.xlabel(reader.independent_variable_axis)
        plt.ylabel(reader.dependent_variable_axis)
        ax.set_title(f'{names["figure title base"]}, {group}')
        
        desk = desks[group]
        desk.figure.update({'figure': fig, 'axes': ax, 'number': fig.number})
        desk.setup()
        desk.zoom(lens = 1)

        fit_diff_ev_least_sq, set_legend, capture_all = desk.fit_diff_ev_least_sq, desk.set_legend, desk.capture_all
        data = desk.categories
        config_per_category = desk.config_per_category
        samples = desk.samples
        averaged_samples = desk.averaged_samples

        ascending_order = [ (averaged_samples[category].max(), category) for category in data ]; ascending_order.sort()
        colors = { category: f'C{index}' for index, (_, category) in enumerate(ascending_order) }
        
        desk.errorbars_text = ax.text(1.01, 0.15, 'Error bars: standard deviation', fontsize = 'x-small', transform = ax.transAxes)
        set_legend()

        styles_use, styles_ignore = {}, {}
        no_data = OD({ 'x': tuple(), 'y': tuple() })
        style = scatter_styles['default']
        for category in data:
            for substyle, instructions in ((styles_use, style['use']['style']), (styles_ignore, style['ignore']['style'])):
                fill, outline = instructions['fill'], instructions['outline']
                translated = { 'marker': instructions['shape'], 's': instructions['box_area'], 'color': colors[category] }
                if 'hatch' in instructions: translated['hatch'] = instructions['hatch']
                if 'match' not in fill: translated['facecolors'] = fill
                if 'match' != outline: translated['edgecolors'] = outline
                substyle[category] = translated
            if category not in desk.legend_categories:
                desk.legend_categories[category] = plt.scatter(**no_data, **styles_use[category])

        for index, category in enumerate(data):
            print(f'\n\nCATEGORY {category}, INDEX {index}\n')

            color = colors[category]
            config = config_per_category[category]
            max_as_max = config.get_setting('max_as_max')

            category_x = desk.x[category]
            category_y = desk.y[category]
            
            category_samples = samples[category]
            category_scatterplots = scatterplots[group][category]
            
            for sample in category_samples:
                x = category_x[sample]
                x_window = x['window']
                x_use, x_ignore = x['use'], x['ignore']
                
                y = category_y[sample]
                y_use, y_ignore = y['use'], y['ignore']
                y_window = y['window']
                y_window_height = y_window[1] - y_window[0]
                y_max = y['max']

                data_use, data_ignore = { 'x': x_use, 'y': y_use }, { 'x': x_ignore, 'y': y_ignore }
                pathcollections_use = plt.scatter(**data_use, zorder = 2.5, **styles_use[category])
                pathcollections_ignore = (
                    plt.scatter(**data_ignore, zorder = 2.5, **styles_ignore[category])
                    if len(x_ignore) != 0 else None )
                pathcollections = {'use': pathcollections_use, 'ignore': pathcollections_ignore}

                errorbars = desk.errorbars[category][sample]
                errorbars_use, errorbars_ignore = errorbars['use'], errorbars['ignore']
                
                scatterplot = {
                    'pathcollections': pathcollections }
                category_scatterplots[sample] = scatterplot
                if sample == 'Averaged':
                    errorbarcontainers_use = plt.errorbar(x_use, y_use, errorbars_use, fmt = 'None', color = color, capsize = 2, zorder = 0)
                    errorbarcontainers_ignore = (
                        plt.errorbar(x_ignore, y_ignore, errorbars_ignore, fmt = 'None', color = color, capsize = 2, zorder = 0)
                        if len(errorbars_ignore) != 0 else None )
                    errorbarcontainers = (errorbarcontainers_use, errorbarcontainers_ignore) if errorbarcontainers_ignore is not None else (errorbarcontainers_use,)
                    scatterplot['errorbarcontainers'] = errorbarcontainers
                show_scatterplot(scatterplot, False)
                
                fitting_info = { 'x': x_use, 'y': y_use, 'category': category, 'sample': sample, 'color': color }
                upperbound = y_max if max_as_max else 10 * y_window_height
                
                fit_diff_ev_least_sq(curve = normalized_exponential, bounds = ((-10, x_window[1]), (0, upperbound), (0, y_max)), other_args = {'maxiter': 1000}, **fitting_info)
                fit_diff_ev_least_sq(curve = onepercent_anchored_logistic, bounds = ((0, upperbound), (-10, x_window[1]), (0, 10000)), other_args = {'maxiter': 1000}, **fitting_info)

            if save_all_fits:
                DE_leastsquares_fits = fits[group][category][DE_leastsquares]
                for curve in curves:
                    curve_fits = DE_leastsquares_fits[curve]
                    all_winners = Fits(
                        group, category, DE_leastsquares, curve,
                        fits = tuple(curve_fits[sample] for sample in category_samples if sample not in special_samples) )
                    
                    filename_args = {'curve': curve, 'category': category, 'special': True, 'all_fits': True}
                    capture_args = {'fits': all_winners, 'marks_visible': False, 'legend_visible': False, 'errorbars_visible': False, 'all_fits': True}
                    
                    presetzoom_folder = winners_all['Preset']
                    autozoom_folder = winners_all['Autozoom']
                    
                    capture_all(zoom_settings, capture_args, filename_args, presetzoom_folder, autozoom_folder)
            
        for curve in curves:
            all_averaged_samples = Fits(
                group, mode = DE_leastsquares, curve = curve,
                fits = tuple(fits[group][category][DE_leastsquares][curve]['Averaged'] for category in data) )
            
            filename_args = {'curve': curve, 'sample': 'Averaged', 'special': True, 'all_categories': True}
            capture_args = {'fits': all_averaged_samples, 'marks_visible': False, 'legend_visible': True, 'errorbars_visible': True, 'all_categories': True, 'title_info': 'averaged'}
            
            presetzoom_folder = winners_special['Averaged']['Preset']
            autozoom_folder = winners_special['Averaged']['Autozoom']

            capture_all(zoom_settings, capture_args, filename_args, presetzoom_folder, autozoom_folder)

        
        desk.DE_leastsquares_averaged_lines = desk.get_winner_lines(DE_leastsquares)
        
        def make_output(minimal = False):
            def make_body(group, mode, minimal = False):
                for curve in curves:
                    group_fits = fits[group]
                    categories = (
                        pd.concat(
                            {f'{category}: {sample}': fit_output(minimal = minimal)},
                            names = ['Category', 'Variable or Output'] ) 
                        for category in desk.categories
                        for sample, fit_output in group_fits[category][mode][curve].items() )
                    combined_categories = pd.concat(categories)
                    curve_label = f'{curve.title}\n\nFunction: {curve.__name__}{signature(curve)}:\n\nEquation: {curve.equation}'
                    yield pd.concat({ curve_label: combined_categories }, names = ['Curve'])
            
            for mode in modes:
                top_info = OD()
                if all(len(samples) == 0 for category in fits[group] for samples in fits[group][category][mode].values()):
                    yield None, None
                    continue
                if mode is DE_leastsquares and hasattr(mode, 'objective_function'):
                    objective_function = mode.objective_function
                    optimizer_notes, output_notes = (diff_ev.notes[key] for key in ('optimizer', 'output'))

                    top_info.update({
                        'Optimizer': f'{diff_ev.function}{signature(diff_ev)}',
                        **{ f'''Optimizer variable: {(
                                key if key not in optimizer_notes
                                else (
                                    note() if callable(note := optimizer_notes[key]) else note )
                            )}''': (
                                value() if callable(value) else value )
                            for key, value in diff_ev.move_to_top['optimizer'].items() },
                        **{ f'''Optimizer output: {(
                                key if key not in output_notes
                                else (
                                    note() if callable(note := output_notes[key]) else note )
                            )}''': (
                                value() if callable(value) else value )
                            for key, value in diff_ev.move_to_top['output'].items() },
                        'Objective function': f'{objective_function.__name__}{signature(objective_function)}',
                        'Objective function variable: curve_variables': 'See curve variables.',
                        '': '' })
                top = pd.DataFrame(
                    np.array(
                        tuple((variable, value)
                        for variable, value in top_info.items()) ))
                body = pd.concat(make_body(group, mode, minimal = minimal))
                yield top, body

        with pd.ExcelWriter(f'{groupfolder_path}/{group_report_filename}.xlsx', engine = 'xlsxwriter') as writer:
            for name, (top, body) in zip(('Differential evolution (DE)', 'Nonlinear least squares (NLS)', 'Minimal DE', 'Minimal NLS'), (*make_output(), *make_output(minimal = True))):
                if body is None: continue
                non_index_copy = body.reset_index()
                former_index = non_index_copy.iloc[:, 0: body.index.nlevels]
                for column in former_index:
                    last_row_value = None
                    for row_index, row in enumerate(former_index[column]):
                        if row != last_row_value: last_row_value = row
                        else: former_index[column].iloc[row_index] = ''
                body.reset_index(inplace = True, drop = True)
                body = pd.concat((former_index, body), axis = 1)
                body.to_excel(writer, sheet_name = name, index = False, startrow = len(top.index))
                workbook = writer.book
                worksheet = writer.sheets[name]
                main_format = workbook.add_format({'text_wrap': True, 'bold': True, 'left': True, 'right': True})
                top_format = workbook.add_format({'text_wrap': False, 'bold': False, 'left': False, 'right': False})
                def reformat_column(column_index, column_name, format):
                    # https://stackoverflow.com/questions/17326973/is-there-a-way-to-auto-adjust-excel-column-widths-with-pandas-excelwriter
                    if column_index == (value_column_index := len(body.columns) - 1): return False
                    column = body[column_name]
                    if column_index == 0:
                        try: column = (*top.iloc[:, column_index], *body[column_name])
                        except IndexError: pass
                    max_entry_width = max(
                        max(
                            map(len, row.split('\n')) )
                        for row in column )
                    header_width = len(column_name)
                    width = np.ceil( max(header_width, max_entry_width) * (10 / 12) )
                    worksheet.set_column(
                        first_col = column_index,
                        last_col = column_index,
                        width = width,
                        cell_format = format )
                    return True
                
                for column_index, column_name in enumerate(body):
                    reformatting = reformat_column(column_index, column_name, main_format)
                    if reformatting is False: break
                
                for row_index in top.index:
                    worksheet.set_row(row_index, cell_format = top_format)
                    for column_index, column in enumerate(top.columns):
                        worksheet.write(
                            row_index,
                            column_index,
                            str(top.iloc[row_index, column_index]),
                            main_format if column_index == 0 else top_format )

    beginning_curve_reports = perf_counter() - initial_time[1]
    formatted_time = divmod(beginning_curve_reports, 60)
    print(f'\nBeginning curve report generation at {int(formatted_time[0])} minute(s) and {formatted_time[1]} seconds. (time.perf_counter())')
    for report_args in curve_reports:
        variable_names, curve = report_args['variable_names'], report_args['curve']
        report_args['variable_names'] = ('RMSE_normalized', *variable_names)
        for if_averaging in (False, True):
            report_name = f'{curve.title} report' + ', averaged'*if_averaging
            with pd.ExcelWriter(f'{output_path_base}/{report_name}.xlsx', engine = 'xlsxwriter') as writer:
                sheets = OD()
                for mode in modes:
                    if mode not in sheets:
                        sheets[mode] = list()
                    sheet = sheets[mode]
                    for group in groups:
                        def group_section(curve, variable_names, average = True, info = None):
                            styles = curve.styles
                            variable_titles = tuple(
                                styles[variable_name]['title'] if variable_name in styles
                                else (
                                    abbreviations[variable_name] if variable_name in abbreviations
                                    else variable_name )
                                for variable_name in variable_names)
                            def categories():
                                for category in fits[group]:
                                    samples = fits[group][category][mode][curve]
                                    if len(samples) == 0: continue
                                    if average:
                                        samples_values = tuple(
                                            fit_output.curve_report(*variable_names)
                                            for sample, fit_output in samples.items() if sample not in special_samples )
                                        if len(samples_values) == 0: continue
                                        category_index = pd.Index((category,), name = 'Category')
                                        calculation_titles = 'Average', 'Standard deviation'
                                        calculation_values = {
                                            title: (values.mean(), values.std())
                                            for title, values in zip(variable_titles, np.array(tuple(zip(*samples_values)))) }
                                        population = len(samples_values)
                                        variables_index = (
                                            pd.MultiIndex.from_tuples((('', 'Population size'),)).append(
                                            pd.MultiIndex.from_product((variable_titles, calculation_titles)) ) )
                                        yield pd.DataFrame(
                                            data = ((population, *(value for title in variable_titles for value in calculation_values[title])),),
                                            index = category_index,
                                            columns = variables_index )
                                        continue
                                    samples_values = tuple(
                                        fit_output.curve_report(*variable_names)
                                        for fit_output in samples.values() )
                                    samples_index = pd.Index((f'{category}_{sample}' for sample in samples), name = 'Category')
                                    yield pd.DataFrame(
                                        data = samples_values,
                                        index = samples_index,
                                        columns = variable_titles )
                            uncombined = tuple(categories())
                            if len(uncombined) == 0: return
                            combined_categories = pd.concat(uncombined)
                            return pd.concat({ group: combined_categories }, names = ['Group'])
                        gs = group_section(average = if_averaging, **report_args)
                        if gs is None: continue
                        sheet.append(gs)
                    sheet_name = mode.title
                    if len(sheet) == 0: continue
                    report = pd.concat(sheet)
                    info, startrow = None, 0
                    if 'info' in report_args:
                        info = pd.Series(report_args['info'])
                        startrow = len(info) + 1
                    report.to_excel(writer, sheet_name = sheet_name, startrow = startrow)
                    workbook = writer.book
                    worksheet = writer.sheets[sheet_name]
                    wrap_format = workbook.add_format({'text_wrap': True})
                    wrap_bold_format = workbook.add_format({'text_wrap': True, 'bold': True})
                    top_format = workbook.add_format({'text_wrap': False})
                    all_columns = (
                        *((None, pd.Series(column)) for column_number, column in enumerate(zip(*report.index))),
                        *((column_name, report[column_name]) for column_name in report ))
                    for column_index, (column_name, column) in enumerate(all_columns):
                        max_entry_width = max(
                            max(
                                map(len, str(row).split('\n')) )
                            for row in column )
                        column_name_type = type(column_name)
                        if column_name_type is str: header_width = len(column_name)
                        elif column_name_type is tuple: header_width = max(len(name) for name in column_name)
                        else: header_width = 0
                        width = np.ceil( max(header_width, max_entry_width) * (10 / 12) )
                        worksheet.set_column(
                            first_col = column_index,
                            last_col = column_index,
                            width = width,
                            cell_format = wrap_format )
                    if info is None: continue
                    for row_index, row in enumerate(info):
                        worksheet.set_row(row_index, cell_format = top_format)
                        worksheet.write(
                            row_index, 0,
                            str(info.index[row_index]),
                            wrap_bold_format )
                        worksheet.write(
                            row_index, 1,
                            str(info[row_index]),
                            top_format )

        
    finishing = perf_counter() - initial_time[1]
    duration = divmod(finishing - beginning_curve_reports, 60)
    print(f'Curve report generation took {int(duration[0])} minute(s) and {duration[1]} seconds. (time.perf_counter())')
    elapsed_perf_counter = divmod(finishing, 60)
    elapsed_monotonic = divmod(monotonic() - initial_time[0], 60)
    with open(f'{output_path_base}/Notes.md', mode = 'w') as readme:
        readme.write('\n'.join((
            "Notes:",
            f"- {iterations} iterations were used to generate this output.",
            "- Run time:",
            f"\t- Measured by time.monotonic(): {int(elapsed_monotonic[0])} minute(s) and {elapsed_monotonic[1]} seconds.",
            f"\t- Measured by time.perf_counter(): {int(elapsed_perf_counter[0])} minute(s) and {elapsed_perf_counter[1]} seconds.") ))
    print(f'\nFinished.\nTime elapsed (time.monotonic()): {int(elapsed_monotonic[0])} minute(s) and {elapsed_monotonic[1]} seconds.\nTime elapsed (time.perf_counter()): {int(elapsed_perf_counter[0])} minute(s) and {elapsed_perf_counter[1]} seconds.')

if is_main: main()