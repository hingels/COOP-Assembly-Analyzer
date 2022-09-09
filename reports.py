import os
import pandas as pd
import datetime
from inspect import signature
from collections import OrderedDict as OD
import numpy as np

from fitter import Fitter
from optimizers import *
from constants_calculation import get_constants

class GroupReport():
    def __init__(self, fitter):
        self.fitter = fitter
        self.group, self.categories, self.groupfolder_path = fitter.group, fitter.categories, fitter.groupfolder_path
    
    def sheet(self, minimal = False):
        group, categories = self.group, self.categories
        curves, fits, modes = Fitter.curves, Fitter.fits, Fitter.modes
        def make_body(group, mode, minimal = False):
            for curve in curves:
                group_fits = fits[group]
                combined_categories = pd.concat(
                    pd.concat(
                        {f'{category}: {sample}': fit_output(minimal = minimal)},
                        names = ['Category', 'Variable or Output'] ) 
                    for category in categories
                    for sample, fit_output in group_fits[category][mode][curve].items() )
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

    def report(self):
        fitter, group, groupfolder_path = self.fitter, self.group, self.groupfolder_path
        sheet = self.sheet
        with pd.ExcelWriter(f'{groupfolder_path}/{group} report.xlsx', engine = 'xlsxwriter') as writer:
            for name, (top, body) in zip(('Differential evolution (DE)', 'Nonlinear least squares (NLS)', 'Minimal DE', 'Minimal NLS'), (*sheet(), *sheet(minimal = True))):
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
                    value_column_index = len(body.columns) - 1
                    if column_index == value_column_index: return False
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

class CurveReports():
    def __init__(self):
        paths = Fitter.paths
        self.data_path, self.config_path = paths['data_path'], paths['config_path']
        self.folder = paths['output_path_base']

    def get_report_info(self, key):
        data_path, config_path = self.data_path, self.config_path
        if key == 'Growth-lag ratio':
            return {key: get_constants()['growth-lag ratio']}
        elif key == 'Input files':
            return {
                'Data file': os.path.basename(data_path),
                'Configuration file': os.path.basename(config_path) }
        elif key == 'Time of report generation':
            return {key: datetime.datetime.now()}
        if key == '': return {}
        raise Exception(f'Could not recognize {key} in report config settings.')
    
    def get_info_rows(self, report):
        get_report_info = self.get_report_info
        keys = list(key.strip() for key in report['info'].split(';'))
        defaults = ('Input files', 'Time of report generation')
        for default in defaults:
            if default not in keys: keys.append(default)
        for key in keys:
            for name, value in get_report_info(key).items():
                yield name, value
    
    def get_curve_reports(self):
        reader = Fitter.reader
        config = reader.config
        get_info_rows = self.get_info_rows
        names_to_curves = Fitter.names_to_curves
        for report in config.reports.values():
            yield OD({
                'curve': names_to_curves[report['curve']],
                'variable_names': tuple(
                    variable.strip()
                    for variable in report['variable_names'].split(';') ),
                **({
                    'info': OD(get_info_rows(report))}
                    if 'info' in report else {} ) })
    
    def report(self):
        folder = self.folder
        curve_reports = self.get_curve_reports()
        modes, fitters, fits, abbreviations, special_samples = Fitter.modes, Fitter.fitters, Fitter.fits, Fitter.abbreviations, Fitter.special_samples
        for report_args in curve_reports:
            variable_names, curve = report_args['variable_names'], report_args['curve']
            report_args['variable_names'] = ('RMSE_normalized', *variable_names)
            for if_averaging in (False, True):
                report_name = f'{curve.title} report' + ', averaged'*if_averaging
                with pd.ExcelWriter(f'{folder}/{report_name}.xlsx', engine = 'xlsxwriter') as writer:
                    sheets = OD()
                    for mode in modes:
                        if mode not in sheets:
                            sheets[mode] = list()
                        sheet = sheets[mode]
                        for group, fitter in fitters.items():
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