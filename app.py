is_main = True if __name__ == '__main__' else False
from timer import Timer
timer = Timer()

from matplotlib import pyplot as plt

from collections import OrderedDict as OD
import os
import typing

from optimizers import *
from reports import GroupReport, CurveReports
from fitter import Fitter
from constants_calculation import smallest
from styles import *

from Curves.KWW import *
from Curves.logistic import *
from Curves.exponential import *


def main():
    my_path = os.path.realpath(__file__)
    root_path = os.path.dirname(my_path)
    Fitter.configure(root_path)
    fitters = Fitter.prepare_fitters()
    
    for group, fitter in fitters.items():
        print(f'\n\nGROUP {group}\n')
        
        fitter.setup()
        reader = fitter.reader
        config = reader.config
        
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
            category_config = config_per_category[category]
            max_as_max = category_config.get_setting('max_as_max')

            category_x = fitter.x[category]
            category_y = fitter.y[category]
            
            category_samples = samples[category]
            category_scatterplots = fitter.scatterplots[category]
            
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

            if config.save_all_fits: fitter.capture_all_fits(category)
            
        fitter.capture_all_averages(data, config.category_collections)

        GroupReport(fitter).report()

    timer.save_time('curve report generation')
    
    CurveReports().report()
    
    timer.save_time(ending = True)
    perf_time, monotonic_time = timer.time
    elapsed_perf_counter, elapsed_monotonic = timer.format_time(perf_time), timer.format_time(monotonic_time)
    output_path_base = Fitter.paths['output_path_base']
    with open(f'{output_path_base}/Notes.md', mode = 'w') as readme:
        readme.write('\n'.join((
            "Notes:",
            f"- {config.iterations} iterations were used to generate this output.",
            "- Run time:",
            f"\t- Measured by time.monotonic(): {int(elapsed_monotonic[0])} minute(s) and {elapsed_monotonic[1]} seconds.",
            f"\t- Measured by time.perf_counter(): {int(elapsed_perf_counter[0])} minute(s) and {elapsed_perf_counter[1]} seconds.") ))
    print(f'\nFinished.\nTime elapsed (time.monotonic()): {int(elapsed_monotonic[0])} minute(s) and {elapsed_monotonic[1]} seconds.\nTime elapsed (time.perf_counter()): {int(elapsed_perf_counter[0])} minute(s) and {elapsed_perf_counter[1]} seconds.')

if is_main: main()