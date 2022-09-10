from timer import Timer
timer = Timer()

from os.path import realpath, dirname
import typing

from fitter import Fitter
from reports import GroupReport, CurveReports
from constants_calculation import smallest

from Curves.KWW import *
from Curves.logistic import *
from Curves.exponential import *


if __name__ == '__main__':
    
    my_path = realpath(__file__)
    root_path = dirname(my_path)
    Fitter.configure(root_path)
    config = Fitter.reader.config
    fitters = Fitter.prepare_fitters()
    
    for group, fitter in fitters.items():
        print(f'\n\nGROUP: {group}')
        
        fitter.setup()
        
        colors, categories, config_per_category, samples = fitter.colors, fitter.categories, fitter.config_per_category, fitter.samples
        plot_sample, fit_diff_ev_least_sq = fitter.plot_sample, fitter.fit_diff_ev_least_sq

        for category in categories:
            print(f'\n\nFitting category {category}.')

            category_x, category_y = fitter.x[category], fitter.y[category]
            category_samples = samples[category]
            category_config = config_per_category[category]
            max_as_max = category_config.get_setting('max_as_max')
            
            for sample in category_samples:
                plot_sample(category, sample)

                x, y = category_x[sample], category_y[sample]
                fitting_info = { 'x': x['use'], 'y': y['use'], 'category': category, 'sample': sample, 'color': colors[category] }
                
                y_max, y_window = y['max'], y['window']
                upperbound = y_max if max_as_max else 10 * (y_window[1] - y_window[0])
                
                fit_diff_ev_least_sq(curve = normalized_exponential, bounds = ((smallest, upperbound), (smallest, y_max)), other_args = {'maxiter': 1000}, **fitting_info)
                fit_diff_ev_least_sq(curve = onepercent_anchored_logistic, bounds = ((smallest, upperbound), (smallest, 10000)), other_args = {'maxiter': 1000}, **fitting_info)

            if config.save_all_fits: fitter.capture_all_fits(category)
            
        fitter.capture_all_averages(categories, config.category_collections)

        GroupReport(fitter).report()

    timer.save_time('curve report generation')
    
    CurveReports().report()
    
    timer.end(write_note = True)