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