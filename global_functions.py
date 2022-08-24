import os
import datetime

from global_info import *
from fits import Fits
from constants_calculation import get_constants

def capture_all_fits(category, desk):
    group, ax, category_samples, winners_all_paths, capture_all = desk.group, desk.ax, desk.samples[category], desk.winners_all_paths, desk.capture_all
    DE_leastsquares_fits = fits[group][category][DE_leastsquares]
    for curve in curves:
        curve_fits = DE_leastsquares_fits[curve]
        all_winners = Fits(
            group, category, DE_leastsquares, curve, ax = ax,
            fits = tuple(curve_fits[sample] for sample in category_samples if sample not in special_samples) )
        
        filename_args = {'curve': curve, 'category': category, 'special': True, 'all_fits': True}
        capture_args = {'fits': all_winners, 'marks_visible': False, 'legend_visible': False, 'errorbars_visible': False, 'all_fits': True}
        
        presetzoom_folder = winners_all_paths['Preset']
        autozoom_folder = winners_all_paths['Autozoom']
        
        capture_all(capture_args, filename_args, presetzoom_folder, autozoom_folder)

def capture_all_averages(desk, data, category_collections):
    group, ax, winners_special_paths, capture_all = desk.group, desk.ax, desk.winners_special_paths, desk.capture_all
    for curve in curves:
        for name, collection in ({'All': data} | category_collections).items():
            all_averaged_samples = Fits(
                group, mode = DE_leastsquares, curve = curve, ax = ax,
                fits = tuple(fits[group][category][DE_leastsquares][curve]['Averaged'] for category in data if category in collection) )
            
            filename_args = {'curve': curve, 'sample': 'Averaged', 'special': True}
            capture_args = {'fits': all_averaged_samples, 'categories': collection, 'marks_visible': False, 'legend_visible': True, 'errorbars_visible': True, 'title_info': 'averaged'}
            if name == 'All':
                filename_args['all_categories'] = True
            else:
                capture_args['categories_title'] = name
                filename_args['category'] = name
            presetzoom_folder = winners_special_paths['Averaged']['Preset']
            autozoom_folder = winners_special_paths['Averaged']['Autozoom']

            capture_all(capture_args, filename_args, presetzoom_folder, autozoom_folder)

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


def update_optima(optima, new_values):
    for dimension in new_values:
        old_min, old_max = optima[dimension]
        new_min, new_max = new_values[dimension]
        replace_min, replace_max = False, False
        if old_min is None or new_min < old_min: replace_min = True
        if old_max is None or new_max > old_max: replace_max = True
        optima[dimension] = (new_min if replace_min else old_min, new_max if replace_max else old_max)