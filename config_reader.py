from collections import OrderedDict as OD
from copy import deepcopy
from functools import reduce
import re
import numpy as np
import pandas as pd
import string
import typing


class ConfigReader(OD):
    "Parses a configuration markdown (.md) file upon instantiation, then uses it to interpret a data file."

    base_level_settings = {
        'Iterations': 'iterations',
        
        'Name for group, singular': 'group_name',
        'Name for groups, plural': 'groups_name',
        'Name for category, singular': 'category_name',
        'Name for categories, plural': 'categories_name',
        
        'Save candidate figures': 'save_candidates',
        'Save "All fits" figures': 'save_all_fits',
        'Save "Averaged samples" figures': 'save_averaged',
        'Save "Combined samples" figures': 'save_combined',
        'Selected experiment': 'selected_experiment' }
    
    lists = {
        'Category configuration defaults:': 'category_config_defaults',
        'Category collections:': 'category_collections',
        'Reports:': 'reports',
        'Figure zoom settings:': 'zoom_settings' }
    category_config_keywords = {
        'end': 'end',
        'subtract_initial': 'subtract_initial',
        'subtract_min': 'subtract_min',
        'combine_samples': 'combine_samples',
        'max_as_max': 'max_as_max' }
    report_keywords = {
        'Curve': 'curve',
        'Variables to report': 'variable_names',
        'Other information': 'info' }
    
    subsections = { 'Figures:', 'Output file naming:', 'Column naming:', 'Settings for Excel (.xlsx) files:' }
    subsections_settings = {
        'Name': 'figure_name',
        'Independent variable axis': 'independent_variable_axis',
        'Dependent variable axis': 'dependent_variable_axis',
        'Font': 'font',
        
        'File name': 'output_filename',

        'Letters and numbers': 'letters_and_numbers',
        'Unnamed': 'unnamed',
        'Join sheets vertically': 'join_sheets',
        'Use sheet names as groups': 'sheet_names_as_groups' }
    
    option_specific_keywords = {
        'letters_and_numbers': { 'Maximum number': 'max_number', '(Optional) Independent variable column name': 'independent_var_column_name' },
        'unnamed': { 'n': 'n', '(Optional) Independent variable column name': 'independent_var_column_name' } }
    
    ignore = { 'HOW TO USE', 'Examples:', 'Notes:' }
    exclude_from_assertions = { 'SETTINGS' }
    hide = { 'SETTINGS' }

    box = re.compile(r'{left_bracket}{anything}{right_bracket}{optional_spaces}'.format(
        left_bracket = r'\[', anything = r'.*', right_bracket = r'\]', optional_spaces = r'\s*' ))
    equal_sign = re.compile(r'{optional_spaces}{equals}{optional_spaces}'.format(
        optional_spaces = r'\s*', equals = r'=' ))
    checkbox = re.compile(r'{left_bracket}{x}{right_bracket}'.format(
        left_bracket = r'\[', x = r'x', right_bracket = r'\]' ))
    wordbox = re.compile(r'{equals}{spaces}{left_bracket}{word}{right_bracket}'.format(
        equals = r'=', spaces = r'\s*', left_bracket = r'\[', word = r'\w+', right_bracket = r'\]' ))
    
    def __init__(self, filepath = None):
        super().__init__()
        if filepath is not None:
            self.read_configfile(filepath)
    def __repr__(self):
        return self.return_readable()
    def apply_functions(self, section_func = None, subsection_func = None, group_func = None, item_func = None, exclude = hide, only_section = None):
        for section in self.items():
            if only_section is not None and section[0] != only_section: continue
            if section[0] in exclude: continue
            if section_func is not None: yield section_func(section=section)
            for subsection in section[1].items():
                if subsection[0] in exclude: continue
                if subsection_func is not None: yield subsection_func(section=section, subsection=subsection)
                for group in subsection[1].items():
                    if group[0] in exclude: continue
                    if group_func is not None: yield group_func(section=section, subsection=subsection, group=group)
                    for item in group[1].items():
                        if item[0] in exclude: continue
                        if item_func is not None: yield item_func(section=section, subsection=subsection, group=group, item=item)

    def read_configfile(self, filepath):
        with open(filepath) as config_file:
            extracted_config = self.extract_config(config_file)
            current_line, current_indentation = extracted_config.__next__()
            self.read_open_configfile(extracted_config, current_line, current_indentation)
    def read_open_configfile(self, extracted_config, current_line, current_indentation):
        current_line, current_indentation = self.parse_configfile(extracted_config, current_line, current_indentation, stop_at = 'EXPERIMENTS')
        settings = self.apply_settings()
        self.parse_configfile(extracted_config, current_line, current_indentation)
        if settings['unnamed'] is True:
            self.numbers_to_columns()
        else:
            self.parse_column_range()
            self.column_range_to_columns()
    
    class Config():
        def __init__(self, reader, settings):
            experiment = settings.pop('selected_experiment')
            self.reader, self.experiment = reader, experiment
            self.names = {
                'figure title base': settings.pop('figure_name'),
                'group': {
                    'singular': settings.pop('group_name'),
                    'plural': settings.pop('groups_name') },
                'category': {
                    'singular': settings.pop('category_name'),
                    'plural': settings.pop('categories_name') } }
            iterations = settings.pop('iterations')
            assert iterations.isdigit()
            self.iterations = int(iterations)
            self.save_candidates = settings.pop('save_candidates')
            self.save_all_fits, self.save_averaged, self.save_combined = settings.pop('save_all_fits'), settings.pop('save_averaged'), settings.pop('save_combined')
            self.category_collections = settings.pop('category_collections') if 'category_collections' in settings else {}
            self.font = settings.pop('font') if 'font' in settings else 'Arial'
            self.zoom_settings = settings.pop('zoom_settings')
            for key, value in settings.items():
                setattr(self, key, value)
    def apply_settings(self):
        'Applies settings and returns them.'
        settings = self.get_settings()
        self.config = self.Config(self, settings)
        return settings

    def parse_configfile(self, extracted_config, current_line, current_indentation, stop_at = None):
        exclude_from_assertions = self.exclude_from_assertions
        ignore = self.ignore
        
        def try_next():
            nonlocal extracted_config
            try:
                current_line, current_indentation = extracted_config.__next__()
                return current_line, current_indentation
            except StopIteration: return None

        section_indentation = current_indentation
        def section():
            nonlocal extracted_config, current_line, current_indentation, section_indentation

            if current_indentation != section_indentation: return
            if stop_at is not None and current_line == stop_at: return
            section_line = current_line
            if all(line not in ignore for line in (section_line,)):
                self[section_line] = OD()

            next = try_next()
            if next is None: return
            current_line, current_indentation = next
            if all(line not in exclude_from_assertions and line not in ignore for line in (section_line,)):
                assert current_indentation > section_indentation, f'Section {section_line} appears to have no contents.'

            subsection_indentation = current_indentation
            def subsection():
                nonlocal extracted_config, current_line, current_indentation, section_line, section_indentation, subsection_indentation

                if current_indentation != subsection_indentation: return
                if subsection_indentation <= section_indentation: return
                if stop_at is not None and current_line == stop_at: return
                subsection_line = current_line
                if all(line not in ignore for line in (section_line, subsection_line)):
                    self[section_line][subsection_line] = OD()

                next = try_next()
                if next is None: return
                current_line, current_indentation = next
                if all(line not in exclude_from_assertions and line not in ignore for line in (section_line, subsection_line)):
                    assert current_indentation > subsection_indentation, f'Subsection {subsection_line} appears to have no contents.'

                group_indentation = current_indentation
                def group():
                    nonlocal extracted_config, current_line, current_indentation, section_line, section_indentation, subsection_line, subsection_indentation, group_indentation

                    if current_indentation != group_indentation: return
                    if group_indentation <= subsection_indentation: return
                    if stop_at is not None and current_line == stop_at: return
                    group_line = current_line
                    if all(line not in ignore for line in (section_line, subsection_line, group_line)):
                        self[section_line][subsection_line][group_line] = OD()

                    next = try_next()
                    if next is None: return
                    current_line, current_indentation = next
                    if all(line not in exclude_from_assertions and line not in ignore for line in (section_line, subsection_line, group_line)):
                        assert current_indentation > group_indentation, f'Group {group_line} appears to have no contents.'
                    
                    item_indentation = current_indentation
                    def item():
                        nonlocal extracted_config, current_line, current_indentation, section_indentation, item_indentation

                        while current_indentation > item_indentation:
                            next = try_next()
                            if next is None: return
                            current_line, current_indentation = next
                        if current_indentation != item_indentation: return
                        if item_indentation <= group_indentation: return
                        if stop_at is not None and current_line == stop_at: return
                        item_line = current_line
                        if all(line not in ignore for line in (section_line, subsection_line, group_line, item_line)):
                            category_config = dict()
                            if section_line == 'EXPERIMENTS':
                                pattern = re.compile(
                                    r'{word_boundary}{word}{spaces}={spaces}({word}{word_boundary}|{multiple})'.format(
                                        word_boundary = r'\b', word = r'\w*', multiple = r'\[.*?\]', spaces = r'\s*') )
                                matches = tuple(pattern.finditer(item_line))
                                if len(matches) != 0:
                                    def splitequals(match):
                                        split = match.group().split('=')
                                        key = split[0].strip()
                                        value = '='.join(split[1:]).strip()
                                        if value.startswith('[') and value.endswith(']'):
                                            def splitexpressions(value):
                                                for expression in value[1:-1].split(';'):
                                                    group_split = expression.split('=')
                                                    assert len(group_split) == 2, f'Too many equal signs in {value}, specifically in {expression}.'
                                                    group_name, group_value = group_split
                                                    yield {group_name.strip(): group_value.strip()}
                                            value = OD(reduce(lambda a, b: a | b, splitexpressions(value)))
                                        return key, value
                                    category_config.update({ key: value for key, value in map(splitequals, matches) })
                                    item_line = item_line[:matches[0].start()].rstrip()
                            self[section_line][subsection_line][group_line][item_line] = OD()
                            if len(category_config) != 0:
                                self[section_line][subsection_line][group_line][item_line]['category_config'] = category_config

                        next = try_next()
                        if next is None: return
                        current_line, current_indentation = next

                        item()
                    if item() is None:
                        group()
                if group() is None:
                    subsection()
            if subsection() is None:
                section()
        section()
        return current_line, current_indentation
    def extract_config(self, file):
        for line in file:
            if line.strip() == '': continue
            indentation = self.tabs_in_line(line)
            line = self.cleanup_line(line)
            yield line, indentation
    def cleanup_line(self, line):
        'Remove preceding hyphens and strip the line of surrounding whitespace.'
        line = line.lstrip()
        line = line if line[0] != '-' else line[1:]
        line = line.strip()
        return line
    def tabs_in_line(self, line):
        'Return the indentation of line, where increments of 1 are equivalent to 4 spaces indentation.'
        shortened_line = line.lstrip()
        spaces = len(line) - len(shortened_line)
        assert spaces % 4 == 0, 'Must use multiples of 4 spaces in indentations.'
        indentation = int(spaces / 4)
        return indentation
    def return_readable(self, item_property = None):
        section_func = lambda section: f'- {section[0]}'
        subsection_func = lambda subsection, **_: f'\t- {subsection[0]}'
        group_func = lambda group, **_: f'\t\t- {group[0]}'
        if item_property is None:
            item_func = lambda item, **_: f'\t\t\t- {item[0]}'
        else:
            item_func = lambda item, **_: f'\t\t\t- {item_property} = {item[1][item_property]}'
        return '\n'.join(self.apply_functions(section_func, subsection_func, group_func, item_func))
    def get_settings(self):
        box, equal_sign, checkbox, wordbox, base_level_settings, subsections_settings, lists = self.box, self.equal_sign, self.checkbox, self.wordbox, self.base_level_settings, self.subsections_settings, self.lists
        settings_dict = dict()
        combine_dicts = lambda dict1, dict2: dict1 | dict2
        def unpack_nested_dict(input, keywords, key = None):
            if isinstance(input, dict):
                for key, value in input.items():
                    if key in keywords and isinstance(value, dict):
                        checked = value is not False
                        yield {keywords[key]: checked}
                        continue
                    yield from unpack_nested_dict(value, keywords, key = key)
            else:
                assert key is not None
                if key in keywords:
                    yield {keywords[key]: input}
        def get_list(lines):
            for line in lines:
                sublines = lines[line]
                if len(sublines) != 0:
                    sub_sublines = dict(get_list(lines = sublines))
                    yield line, sub_sublines
                    continue
                yield line, None
        def parse_boxes(line = None, lines = None, keywords = None):
            assert keywords is not None
            assert (line and lines) is None, 'Only "line" or "lines" can be specified, not both.'
            assert (line or lines) is not None, 'Either "line" or "lines" must be specified.'
            def get_boxes(line = None, lines = None, depth = 0, args_per_depth = { 0: {'solo_checkable': True}, 1: {'solo_checkable': True} }):
                assert depth in args_per_depth, 'get_boxes has exceeded the depth covered in args_per_depth. Please modify args_per_depth to cover greater depth.'
                solo_checkable = args_per_depth[depth]['solo_checkable']
                
                option_selected = False
                if lines is None:
                    lines = {line: None}
                for setting in lines:
                    if box.search(setting) is None: continue

                    checked = (checkbox.search(setting) is not None)
                    worded = (wordbox.search(setting) is not None)
                    assert (checked and worded) is False, f'Can only use checked, numbered, or worded boxes one at a time. {setting=}'

                    box_name = None

                    equals_split, box_split = equal_sign.split(setting), box.split(setting)
                    has_equals, has_box = (len(equals_split) > 1), (len(box_split) > 1)
                    if has_equals:
                        box_name, box_value = equals_split[0], equals_split[1][1:-1]
                    elif has_box:
                        assert worded is False, f'No equals sign is present, but a wordbox was found. {setting=}'
                        box_name, box_value = box_split[1], checked
                    assert box_name is not None, f'Box name could not be determined. {setting=}'

                    if checked:
                        if solo_checkable: assert option_selected == False, f'Only one box may be selected. {setting=}'
                        option_selected = box_name
                        box_value = True
                    
                    subsettings = lines[setting]
                    if subsettings is not None and len(subsettings) != 0:
                        if checked or worded:
                            subsettings_values = dict(get_boxes(lines = subsettings, depth = depth + 1))
                            yield box_name, subsettings_values
                            continue
                    
                    yield box_name, box_value

            settings_tree = dict(get_boxes(line, lines))
            assert len(settings_tree) != 0, 'No boxes could be found in "{middle}".'.format(middle = '" and "'.join(lines.keys()))
            
            settings = reduce(combine_dicts, unpack_nested_dict(settings_tree, keywords))

            for option in ('letters_and_numbers', 'unnamed'):
                if option in settings and settings[option] is True:
                    settings = settings | reduce(combine_dicts, unpack_nested_dict(settings_tree, self.option_specific_keywords[option]))
                    break

            return settings
        
        entries = self['SETTINGS']
        for entry, contents in entries.items():
            if entry in lists:
                setting = lists[entry]
                if entry == 'Category configuration defaults:':
                    if setting not in settings_dict: settings_dict[setting] = dict()
                    parsed_settings = parse_boxes(lines = contents, keywords = self.category_config_keywords)
                    settings_dict[setting] = parsed_settings
                elif entry == 'Category collections:':
                    settings_dict[setting] = {key: tuple(value.keys()) for key, value in contents.items()}
                elif entry == 'Reports:':
                    if setting not in settings_dict: settings_dict[setting] = dict()
                    for curve, report_settings in contents.items():
                        if issubclass(type(report_settings), dict):
                            parsed_settings = parse_boxes(lines = report_settings, keywords = self.report_keywords)
                        else:
                            parsed_settings = parse_boxes(line = report_settings, keywords = self.report_keywords)
                        settings_dict[setting][curve] = parsed_settings
                elif entry == 'Figure zoom settings:':
                    def get_zoom_settings():
                        for key, value in contents.items():
                            zoom = (key == 'Zoom magnifications:')
                            if zoom:
                                yield {'zoom': tuple(value.keys())}
                                continue
                            yield parse_boxes(line = key, keywords = {'Autozoom': 'autozoom'})
                    settings_dict[setting] = OD(reduce(lambda a, b: a | b, get_zoom_settings()))
                continue
            if entry in self.subsections:
                settings_dict.update(parse_boxes(lines = contents, keywords = base_level_settings | subsections_settings))
                continue
            if any(setting in entry for setting in self.base_level_settings):
                settings_dict.update(parse_boxes(line = entry, keywords = base_level_settings | subsections_settings))
                continue
            split_setting = entry.split('=')
            assert len(split_setting) > 1, f'Setting "{entry}" appears to lack an equal sign. Cannot be interpreted.'
            assert len(split_setting) == 2, f'Setting "{entry}" appears to have more than one equal sign. Cannot be interpreted.'
            key, value = split_setting[0].strip(), split_setting[1].strip()
            settings_dict[key] = value
        return settings_dict

    def read_data(self, filepath = None):
        config = self.config
        if filepath.endswith('.csv'):
            sheet = pd.read_csv(filepath)
            dataframe = self.cleanup_sheet(sheet)
            return self.read_from_columns(dataframe, sheet_name = sheet_name)
        else:
            assert filepath.endswith('xlsx'), f'Currently, only .csv and .xlsx files are supported. {filepath} cannot be used.'
            excelfile = pd.ExcelFile(filepath)
            sheets = pd.read_excel(excelfile, sheet_name = None)
            if config.join_sheets:
                dataframe = self.concatenate_sheets(sheets.values())
                return self.read_from_columns(dataframe, sheet_name = sheet_name)
            elif config.sheet_names_as_groups:
                _categories_per_experiment = OD()
                for sheet_name, sheet_dataframe in sheets.items():
                    sheet_dataframe = self.cleanup_sheet(sheet_dataframe)
                    new_categories_per_experiment = self.read_from_columns(sheet_dataframe, sheet_name = sheet_name, sheet_names_as_groups = True)
                    for experiment, contents in new_categories_per_experiment.items():
                        if experiment not in _categories_per_experiment:
                            _categories_per_experiment[experiment] = dict()
                        _categories_per_experiment[experiment].update(contents)
                return _categories_per_experiment
    def cleanup_sheet(self, sheet):
        sheet = sheet.dropna()
        allowed_rows = (sheet != 0).any(axis=1)
        return sheet.loc[ allowed_rows, : ]
    def concatenate_sheets(self, sheets, cleanup = True):
        to_concatenate = []
        get_time_values = np.vectorize(lambda x: x.hour*60 + x.minute)
        for sheet in sheets:
            if cleanup:
                sheet = self.cleanup_sheet(sheet)
            sheet.iloc[:, 0] = get_time_values(sheet.iloc[:, 0])
            if len(to_concatenate) != 0:
                time_between_sheets = to_concatenate[-1].iloc[-1, 0] - to_concatenate[-1].iloc[-2, 0]
                sheet.iloc[:, 0] += to_concatenate[-1].iloc[-1, 0] + time_between_sheets
            to_concatenate.append(sheet)
        return pd.concat(to_concatenate)
    def parse_column_range(self):
        def item_func(section, subsection, group, item):
            section_name, subsection_name, group_name, item_name = section[0], subsection[0], group[0], item[0]
            column_ranges = list( word.strip() for word in item_name.split(',') )
            column_ranges[-1] = column_ranges[-1].split(' ')[0]
            translated = list()
            for column_range in column_ranges:
                assert column_range[-1] != '-', f'The last endpoint in the column range cannot be a hyphen. {column_range} does not meet this requirement.'
                column_range = column_range.upper()
                endpoints = column_range.split('-')
                alphabet = string.ascii_uppercase
                numbers = tuple( str(number) for number in range(10) )
                length = len(endpoints)
                assert length <= 2, f'One or two endpoints must be specified in the column range. {column_range} of length {length} does not meet the requirement.'
                if length == 2:
                    start_column, end_column = endpoints[0], endpoints[1]
                    assert start_column[0] in alphabet, f'The first endpoint in the column range must begin with a letter. {start_column} in {column_range} does not meet the requirement.'
                    for endpoint in endpoints: assert endpoint[-1] in numbers, f'Endpoints in the column range must end with a number. {endpoint} in {column_range} does not meet the requirement.'
                    if end_column[0] not in alphabet:
                        endpoints[1] = start_column[0] + end_column
                    else:
                        assert end_column[0] >= start_column[0], f'The end column in the column range should begin with a letter that either matches or succeeds the letter of the start point. {column_range} does not meet the requirement.'
                else:
                    column_name = column_range
                    assert column_name[0] in alphabet, f'The first endpoint in the column_name range must begin with a letter. {column_name} does not meet the requirement.'
                    assert column_name[-1] in numbers, f'Endpoints in the column_name range must end with a number. {column_name} does not meet the requirement.'
                for endpoint in endpoints: assert endpoint[1].upper() not in alphabet, f'Currently, columns whose names begin with more than one letter are not supported. {endpoint} in {endpoints} does not meet the requirement. Please rename to use single letters.'
                translated.append(endpoints)
            self[section_name][subsection_name][group_name][item_name]['column_ranges'] = translated
        tuple(self.apply_functions(item_func=item_func))
        return self
    def column_range_to_columns(self):
        config = self.config
        alphabet = string.ascii_uppercase
        max_number = int(config.max_number)
        def item_func(section, subsection, group, item):
            nonlocal alphabet
            section_name, subsection_name, group_name, item_name = section[0], subsection[0], group[0], item[0]
            column_ranges = self[section_name][subsection_name][group_name][item_name]['column_ranges']
            columns = list()
            for column_range in column_ranges:
                for column_name in column_range: assert int(column_name[1:]) <= max_number, f'Endpoints in the column range must have numbers that do not exceed the maximum number, currently set to {max_number}. {column_name}{" in " + str(column_range) if len(column_range)!=0 else ""} does not meet the requirement.'
                if len(column_range) == 1:
                    columns.append(column_range[0])
                    continue
                start_column, end_column = column_range[0], column_range[1]
                start_letter, end_letter = start_column[0], end_column[0]
                start_number, end_number = int(start_column[1:]), int(end_column[1:])
                if start_letter == end_letter:
                    letter = start_letter
                    all_columns = [ f'{letter}{number}' for number in range(start_number, end_number + 1) ]
                    columns += all_columns
                    continue
                start_letter_index = alphabet.index(start_letter)
                end_letter_index = alphabet.index(end_letter)
                all_letters = [ alphabet[index] for index in range(start_letter_index, end_letter_index + 1) ]
                all_columns = list()
                number = start_number
                for letter in all_letters:
                    while (count := number % (max_number + 1)) != 0:
                        if letter == end_letter and count > end_number: break
                        all_columns.append(f'{letter}{count}')
                        number += 1
                    number += 1
                columns += all_columns
            self[section_name][subsection_name][group_name][item_name]['column_names'] = columns
        tuple(self.apply_functions(item_func=item_func))
        return self
    def numbers_to_columns(self):
        config = self.config
        column_name = 0   # All indices will be shifted later to account for the possibility of the independent variable column not being column 0.
        def item_func(section, subsection, group, item):
            nonlocal column_name
            section_name, subsection_name, group_name, item_name = section[0], subsection[0], group[0], item[0]
            assert group_name != 'CONTROLS', 'Cannot use CONTROLS group in "unnamed" mode. Instead, enter controls in order under SAMPLES and add control=True.'
            n = int(config.n)
            columns = list()
            for i in range(n):
                columns.append(column_name)
                column_name += 1
            self[section_name][subsection_name][group_name][item_name]['column_names'] = columns
        tuple(self.apply_functions(item_func=item_func))
        return self
    def read_from_columns(self, dataframe, sheet_name = None, sheet_names_as_groups = False):
        config = self.config
        _categories_per_experiment = OD()
        def add_category(section, subsection, group, item):
            section_name, subsection_name, group_name, item_name = section[0], subsection[0], group[0], item[0]
            if sheet_names_as_groups is True:
                assert sheet_name is not None
                group_name = sheet_name
            assert section_name == 'EXPERIMENTS'
            experiment_name, category_name = subsection_name, item_name
            if _categories_per_experiment.get(experiment_name, None) is None:
                _categories_per_experiment[experiment_name] = OD()
            column_names = item[1]['column_names']
            category_config = dict()
            if 'category_config' in item[1]:
                category_config.update(item[1]['category_config'])
            independent_var_column_name = config.independent_var_column_name
            independent_var_column = dataframe.iloc[:, 0] if independent_var_column_name == '' else dataframe[independent_var_column_name]
            if config.unnamed:
                offset = 1
                if independent_var_column_name != '':
                    offset += dataframe.columns.get_loc(independent_var_column_name)
                column_indices = [ number + offset for number in column_names ]
                data = pd.concat( (dataframe.iloc[:, column_index] for column_index in column_indices), axis = 1 )
            else:
                data = pd.concat( (dataframe[column_name] for column_name in column_names), axis = 1 )
            if group_name not in _categories_per_experiment[experiment_name]:
                _categories_per_experiment[experiment_name][group_name] = dict()
            group_entry = _categories_per_experiment[experiment_name][group_name]
            group_entry.update({ category_name: { 'data': data, 'category_config': category_config } })
            if 'independent_var_column' not in group_entry:
                group_entry['independent_var_column'] = independent_var_column
        tuple(self.apply_functions(item_func=add_category, only_section = 'EXPERIMENTS'))
        return _categories_per_experiment

    def subtract_controls(self, data_per_category):
        control, control_dataframe = None, None
        for key, value in data_per_category.items():
            category_config = value['category_config']
            if 'control' in category_config and category_config['control'] is True:
                assert control is None, 'There appear to be multiple control groups. Please specify only one.'
                control = key
                control_dataframe = data_per_category[control]['data'].replace('OVRFLW', 0)
        assert control is not None, 'Control group cannot be found. Please specify one.'
        
        samples = ( sample for sample in data_per_category.keys() if sample != control )
        control_array = control_dataframe.astype(float).values
        for sample in samples:
            sample_dataframe = data_per_category[sample]['data']
            sample_dataframe = sample_dataframe.replace('OVRFLW', 0).astype(float)
            sample_dataframe = sample_dataframe - control_array
            data_per_category[sample]['data'] = sample_dataframe
        control_before_subtraction = deepcopy(data_per_category[control])
        data_per_category[control]['data'] = control_dataframe - control_array      # Should be zero; subtracting itself
        all_zero = bool( (data_per_category[control]['data'] == 0).all(axis=None) )
        assert all_zero is True, 'Subtracting controls failed to produce zero in control data. Please check for invalid types or report a bug.'
        return control_before_subtraction