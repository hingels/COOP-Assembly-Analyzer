import os
import numpy as np
import re
from matplotlib import pyplot as plt
from functools import reduce

from global_info import *
from styles import default_margins
from fits import Fits
from desk import Desk
from category_config import CategoryConfig

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

        fig, ax = plt.subplots()
        plt.xlabel(reader.independent_variable_axis)
        plt.ylabel(reader.dependent_variable_axis)
        ax.set_title(f'{reader.figure_name}, {group}')
        desk.figure.update({'figure': fig, 'axes': ax, 'number': fig.number})

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

        def figure_paths(path):
            autozoom_path = f'{path}/Automatic zoom'
            presetzoom_path = f'{path}/Preset zoom'
            return {'Autozoom': autozoom_path, 'Preset': presetzoom_path}
        def special_paths(path):
            averaged_paths = figure_paths(f'{path}/Averaged samples')
            combined_paths = figure_paths(f'{path}/Combined samples')
            return {'Averaged': averaged_paths, 'Combined': combined_paths}
        
        output_path_base = Desk.paths['output_path_base']
        groupfolder_path = f'{output_path_base}/{group}'
        figures_path = f'{groupfolder_path}/Figures'

        candidates_path = f'{figures_path}/Candidates'
        candidates_individual = figure_paths(f'{candidates_path}/Individual samples')
        candidates_special = special_paths(f'{candidates_path}/All samples')

        winners_path = f'{figures_path}/Winners' if Desk.save_candidates else figures_path
        winners_individual_paths = figure_paths(f'{winners_path}/Individual samples')
        winners_special_paths = special_paths(f'{winners_path}/All samples')
        winners_all_paths = figure_paths(f'{winners_path}/All samples/All fits')

        paths['groups'][group] = {
            'groupfolder_path': groupfolder_path,
            'figures_path': figures_path,
            'candidates': {
                'individual': candidates_individual,
                'special': candidates_special },
            'winners': {
                'individual': winners_individual_paths,
                'special': winners_special_paths,
                'all': winners_all_paths } }

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