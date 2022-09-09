is_main = True if __name__ == '__main__' else False
from timer import Timer
timer = Timer()

from matplotlib import pyplot as plt

from collections import OrderedDict as OD
import datetime
from functools import partial
import sys
import os
import shutil
import typing

from optimizers import *
from reports import GroupReport, CurveReports
from fitter import Fitter
from config_reader import ConfigReader
from constants_calculation import smallest
from styles import *

from Curves.KWW import *
from Curves.logistic import *
from Curves.exponential import *



def main():
    get_files = Fitter.get_files
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
    category_collections = reader.category_collections if hasattr(reader, 'category_collections') else {}
    plt.rcParams['font.family'] = reader.font if hasattr(reader, 'font') else 'Arial'
    zoom_settings = reader.zoom_settings

    output_filename_unformatted = partial(reader.output_filename.format)
    current_time = datetime.datetime.now()
    for abbreviation, meaning in Fitter.abbreviations['time'].items():
        output_filename_unformatted = partial(output_filename_unformatted, **{abbreviation: current_time.strftime(meaning)})
    output_filename_base = output_filename_unformatted()
    
    path_base = '{root}/Output/{name}{i}'
    format_kwargs = {'root': root, 'name': output_filename_base, 'i': ''}
    while os.path.isdir(path_base.format(**format_kwargs)):
        format_kwargs['i'] = 2 if (i := format_kwargs['i']) == '' else (i + 1)
    path_base = path_base.format(**format_kwargs)

    input_copy = f'{path_base}/Input'
    os.makedirs(input_copy)
    shutil.copy2(data_path, input_copy)
    shutil.copy2(config_path, input_copy)
    
    output_path_base = f'{path_base}/Output'
    os.makedirs(output_path_base)
    
    paths, categories_per_experiment, scatterplots, fitters = Fitter.paths, Fitter.categories_per_experiment, Fitter.scatterplots, Fitter.fitters
    prepare_fitters = Fitter.prepare_fitters
    
    paths.update({'root': root, 'input_path': input_path, 'data_path': data_path, 'config_path': config_path, 'output_filename_base': output_filename_base, 'path_base': path_base, 'input_copy': input_copy, 'output_path_base': output_path_base })
    paths['groups'] = dict()

    categories_per_experiment.update(reader.read_data(data_path))
    groups = categories_per_experiment[experiment].keys()

    for key, value in { 'iterations': iterations, 'save_candidates': save_candidates, 'save_averaged': save_averaged, 'save_combined': save_combined, 'zoom_settings': zoom_settings, 'names': names }.items():
        setattr(Fitter, key, value)
    setattr(Fitter.Fits, 'scatterplots', scatterplots)
    
    prepare_fitters(categories_per_experiment[experiment], groups, reader)
    
    for group in groups:
        print(f'\n\nGROUP {group}\n')
        
        fitter = fitters[group]
        fitter.setup()
        
        ax = fitter.figure['axes']

        data, config_per_category, samples, averaged_samples, fit_diff_ev_least_sq, set_legend = fitter.categories, fitter.config_per_category, fitter.samples, fitter.averaged_samples, fitter.fit_diff_ev_least_sq, fitter.set_legend
        show_scatterplot = Fitter.show_scatterplot

        ascending_order = [ (averaged_samples[category].max(), category) for category in data ]; ascending_order.sort()
        colors = { category: f'C{index}' for index, (_, category) in enumerate(ascending_order) }
        
        fitter.errorbars_text = ax.text(1.01, 0.15, 'Error bars: standard deviation', fontsize = 'x-small', transform = ax.transAxes)
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
            if category not in fitter.legend_categories:
                fitter.legend_categories[category] = plt.scatter(**no_data, **styles_use[category])

        for index, category in enumerate(data):
            print(f'\n\nCATEGORY {category}, INDEX {index}\n')

            color = colors[category]
            config = config_per_category[category]
            max_as_max = config.get_setting('max_as_max')

            category_x = fitter.x[category]
            category_y = fitter.y[category]
            
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

                errorbars = fitter.errorbars[category][sample]
                errorbars_use, errorbars_ignore = errorbars['use'], errorbars['ignore']
                
                scatterplot = { 'pathcollections': pathcollections }
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
                
                fit_diff_ev_least_sq(curve = normalized_exponential, bounds = ((smallest, upperbound), (smallest, y_max)), other_args = {'maxiter': 1000}, **fitting_info)
                fit_diff_ev_least_sq(curve = onepercent_anchored_logistic, bounds = ((smallest, upperbound), (smallest, 10000)), other_args = {'maxiter': 1000}, **fitting_info)

            if save_all_fits: fitter.capture_all_fits(category)
            
        fitter.capture_all_averages(data, category_collections)

        GroupReport(fitter).report()

    timer.save_time('curve report generation')
    
    CurveReports(reader, fitters).report()
    
    timer.save_time()
    perf_time, monotonic_time = timer.time
    elapsed_perf_counter, elapsed_monotonic = timer.format_time(perf_time), timer.format_time(monotonic_time)
    with open(f'{output_path_base}/Notes.md', mode = 'w') as readme:
        readme.write('\n'.join((
            "Notes:",
            f"- {iterations} iterations were used to generate this output.",
            "- Run time:",
            f"\t- Measured by time.monotonic(): {int(elapsed_monotonic[0])} minute(s) and {elapsed_monotonic[1]} seconds.",
            f"\t- Measured by time.perf_counter(): {int(elapsed_perf_counter[0])} minute(s) and {elapsed_perf_counter[1]} seconds.") ))
    print(f'\nFinished.\nTime elapsed (time.monotonic()): {int(elapsed_monotonic[0])} minute(s) and {elapsed_monotonic[1]} seconds.\nTime elapsed (time.perf_counter()): {int(elapsed_perf_counter[0])} minute(s) and {elapsed_perf_counter[1]} seconds.')

if is_main: main()