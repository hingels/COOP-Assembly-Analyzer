from collections import OrderedDict as OD
import numpy as np

mark_styles = OD({
    'lagtime': {'title': 'Lag time', 'style': {'shape': 's', 'fill': 'white', 'hatch': '.'*8, 'outline': 'black', 'box_area': 25}},
    'leadtime': {'title': 'Lead time', 'style': {'shape': '^', 'fill': 'white', 'hatch': '.'*8, 'outline': 'black', 'box_area': 25}},
    't_99%': {'title': '99% time', 'style': {'shape': '^', 'fill': 'white', 'hatch': '/'*10, 'outline': 'black', 'box_area': 25}},
    't_95%': {'title': '95% time', 'style': {'shape': '^', 'fill': 'match4', 'outline': 'black', 'box_area': 25}},
    't_90%': {'title': '90% time', 'style': {'shape': '^', 'fill': 'white', 'outline': 'black', 'box_area': 25}},
    't_50%': {'title': '50% time', 'style': {'shape': '*', 'fill': 'match2', 'outline': 'black', 'box_area': 44}},
    't_10%': {'title': '10% time', 'style': {'shape': 's', 'fill': 'white', 'outline': 'black', 'box_area': 20}},
    't_5%': {'title': '5% time', 'style': {'shape': 's', 'fill': 'match3', 'outline': 'black', 'box_area': 20}},
    't_1%': {'title': '1% time', 'style': {'shape': 's', 'fill': 'white', 'hatch': '/'*10, 'outline': 'black', 'box_area': 20}},
    't0': {'title': 'Start time', 'style': {'shape': 'o', 'fill': 'match1', 'outline': 'black', 'box_area': 33}} })
scatter_styles = {
    'default': {
        'use': {'title': 'use', 'style': {'shape': 'o', 'fill': 'match', 'outline': 'match', 'box_area': 10}},
        'ignore': {'title': 'Ignored data', 'style': {'shape': 'o', 'fill': 'white', 'outline': 'match', 'box_area': 10}} }}
legend_match_combos = {
    'match1': { 'fillstyle': 'left', 'markerfacecolor': 'purple', 'markerfacecoloralt': 'cyan' },
    'match2': { 'fillstyle': 'left', 'markerfacecolor': 'royalblue', 'markerfacecoloralt': 'lime' },
    'match3': { 'fillstyle': 'left', 'markerfacecolor': 'gold', 'markerfacecoloralt': 'orangered' },
    'match4': { 'fillstyle': 'left', 'markerfacecolor': 'mediumpurple', 'markerfacecoloralt': 'deeppink' } }
max_height = 0
for name, style_dict in mark_styles.items():
    height = np.sqrt(style_dict['style']['box_area'])
    if height > max_height:
        max_height = height
spacing_in_points = max_height
points_per_fontsize_unit = 5
spacing_in_fontsize_units = 1.2 * (spacing_in_points / points_per_fontsize_unit)

default_legend_kwargs = {'fontsize': points_per_fontsize_unit, 'labelspacing': spacing if (spacing := spacing_in_fontsize_units - 1) >= 0 else 0, 'handleheight': 0, 'loc': 'lower left', 'mode': 'expand'}

class Margins(dict):
    default_bottom, default_top, default_left, default_right = 0.2, 0.2, 0.2, 0.2
    def __init__(self, bottom = default_bottom, top = default_top, left = default_left, right = default_right):
        self.bottom, self.top, self.left, self.right = bottom, top, left, right
        super().__init__({'top': top, 'bottom': bottom, 'left': left, 'right': right})
default_margins = Margins()