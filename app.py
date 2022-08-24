is_main = True if __name__ == '__main__' else False
from time import monotonic, perf_counter
initial_time = monotonic(), perf_counter()

from collections import OrderedDict as OD
import datetime
from functools import partial, reduce
from inspect import signature
import sys
import os
import shutil
import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import typing

from optimizers import *
from fits import Fits
from desk import Desk
from category_config import CategoryConfig
from prepare_sheets import ConfigReader
from global_info import *
from global_functions import *
from styles import *

from KWW import *
from logistic import *
from exponential import *


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
    category_collections = reader.category_collections
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

    for key, value in { 'iterations': iterations, 'save_candidates': save_candidates, 'save_averaged': save_averaged, 'save_combined': save_combined, 'zoom_settings': zoom_settings, 'groups': groups, 'fits': fits, 'names': names, 'scatterplots': scatterplots, 'paths': paths, 'desks': desks }.items():
        setattr(Desk, key, value)
    setattr(Fits, 'scatterplots', scatterplots)
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
        winners_individual_paths = figure_paths(f'{winners_path}/Individual samples')
        winners_special_paths = special_paths(f'{winners_path}/All samples')
        winners_all_paths = figure_paths(f'{winners_path}/All samples/All fits')

        paths[group] = {
            'groupfolder_path': groupfolder_path,
            'figures_path': figures_path,
            'candidates': {
                'individual': candidates_individual,
                'special': candidates_special },
            'winners': {
                'individual': winners_individual_paths,
                'special': winners_special_paths,
                'all': winners_all_paths } }

        group_report_filename = f'{group} report'
        
        fig, ax = plt.subplots()
        plt.xlabel(reader.independent_variable_axis)
        plt.ylabel(reader.dependent_variable_axis)
        ax.set_title(f'{names["figure title base"]}, {group}')
        
        desk = desks[group]
        desk.figure.update({'figure': fig, 'axes': ax, 'number': fig.number})
        desk.setup()
        desk.zoom(lens = 1)

        data, config_per_category, samples, averaged_samples, fit_diff_ev_least_sq, set_legend, capture_all = desk.categories, desk.config_per_category, desk.samples, desk.averaged_samples, desk.fit_diff_ev_least_sq, desk.set_legend, desk.capture_all
        show_scatterplot = Desk.show_scatterplot

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

            if save_all_fits: capture_all_fits(category, desk)
            
        capture_all_averages(desk, data, category_collections)

        
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