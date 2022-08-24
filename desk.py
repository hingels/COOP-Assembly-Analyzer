import os
from collections import OrderedDict as OD
from matplotlib.lines import Line2D
import typing
import warnings

from fits import Fits
from matplotlib import pyplot as plt
from global_info import *
from styles import *
from optimizers import *

class Desk():
    figure_box = None
    def __init__(self, group):
        self.group = group
        self.legend_handles_labels = OD()
        self.legend_categories = OD()
        self.figure = {}
        self.averaged_samples = {}
        self.standard_deviations = {}
        self.group_optima = {'x': (None, None), 'y': (None, None)}
        self.category_optima = {}
        self.sample_optima = {}
    def setup(self):
        group = self.group

        paths = self.paths['groups'][group]
        self.figures_path = paths['figures_path']
        self.groupfolder_path = paths['groupfolder_path']
        if 'candidates' in paths:
            candidates_paths = paths['candidates']
            self.candidates_individual = candidates_paths['individual']
            self.candidates_special = candidates_paths['special']
        if 'winners' in paths:
            winners_paths = paths['winners']
            self.winners_individual_paths = winners_paths['individual']
            self.winners_special_paths = winners_paths['special']
            self.winners_all_paths = winners_paths['all']
        
        figure_info = self.figure
        self.fig, self.ax, self.fig_number = figure_info['figure'], figure_info['axes'], figure_info['number']
    @staticmethod
    def show(line, visible = True, /, linestyle = None, marks_visible = True):
        if not visible: marks_visible = False
        line2d, marks = line['line2d'], line['marks']
        active_linestyle = '-' if linestyle is None else linestyle
        line2d.set_linestyle(active_linestyle if visible else 'None')
        for mark in marks.values():
            mark.set_visible(marks_visible)
    @staticmethod
    def show_scatterplot(scatterplot, visible = True):
        for pathcollection in scatterplot['pathcollections'].values():
            if pathcollection is None: continue
            pathcollection.set_visible(visible)
        if 'errorbarcontainers' not in scatterplot: return
        for errorbarcontainer in scatterplot['errorbarcontainers']:
            for component_tuple in errorbarcontainer.lines:
                if component_tuple is None: continue
                for component in component_tuple:
                    component.set_visible(visible)
    def show_all(self, fits, fits_visible = True, marks_visible = True, legend_visible = True, errorbars_visible = False, categories = None):
        'Shows or hides all lines and scatterplots in the given Fits object.'
        show_marks_on_legend = self.show_marks_on_legend
        show_scatterplot, show = Desk.show_scatterplot, Desk.show
        scatterplot = fits.scatterplot
        use_fits_scatterplot = (scatterplot is not None)
        show_ignored = False
        if use_fits_scatterplot:
            if scatterplot['pathcollections']['ignore'] is not None: show_ignored = True
            show_scatterplot(scatterplot, fits_visible)
        if not marks_visible or not fits_visible: show_marks_on_legend(marks_visible = False, legend_visible = legend_visible, errorbars_visible = errorbars_visible, categories = categories, show_ignored = show_ignored)
        if not fits_visible:
            for fit in fits:
                show(fit.line, False)
                if not use_fits_scatterplot:
                    fit_scatterplot = fit.scatterplot
                    if fit_scatterplot['pathcollections']['ignore'] is not None: show_ignored = True
                    show_scatterplot(fit_scatterplot, False)
            return
        if hasattr(fits, 'color'):
            for fit in fits:
                show(fit.line, linestyle = fit.initial_linestyle, marks_visible = marks_visible)
                if not use_fits_scatterplot:
                    fit_scatterplot = fit.scatterplot
                    if fit_scatterplot['pathcollections']['ignore'] is not None: show_ignored = True
                    show_scatterplot(fit_scatterplot)
            if marks_visible: show_marks_on_legend(fits.curve, legend_visible = legend_visible, errorbars_visible = errorbars_visible, categories = categories, show_ignored = show_ignored, color = fits.color)
            return
        curve = fits.curve
        for fit in fits:
            show(fit.line, linestyle = fit.initial_linestyle, marks_visible = marks_visible)
            if not use_fits_scatterplot:
                fit_scatterplot = fit.scatterplot
                if fit_scatterplot['pathcollections']['ignore'] is not None: show_ignored = True
                show_scatterplot(fit_scatterplot)
        if marks_visible: show_marks_on_legend(curve, legend_visible = legend_visible, errorbars_visible = errorbars_visible, categories = categories, show_ignored = show_ignored)
    
    def get_capture_filename(self, curve, category = None, sample = None, special = False, autozoom = False, lens = 1, all_fits = False, all_categories = False):
        out = []
        if special:
            if category is not None: out.append(f'{curve.title}_{category}')
            else: out.append(curve.title)
        if sample is not None: out.append(sample)
        if all_fits: out.append('All fits')
        if autozoom: out.append('Autozoom')
        else: out.append(f'ZoomX{lens}')
        if all_categories: out.append('All')
        return '_'.join(out) + '.png'
    def capture(self, fits: Fits, folder = None, filename = None, lens = 1, autozoom = False, marks_visible = True, legend_visible = True, errorbars_visible = False, all_fits = False, categories = None, title_info = None, categories_title = None):
        ax, names, show_all, zoom, offscreen_marks_transparent = self.ax, self.names, self.show_all, self.zoom, self.offscreen_marks_transparent
        show_all_args = {'marks_visible': marks_visible, 'legend_visible': legend_visible, 'errorbars_visible': errorbars_visible, 'categories': categories}

        plt.sca(ax)
        if autozoom: assert lens == 1, 'Lens must be 1 if autozoom is enabled.'
        else: folder += f'/Zoom x{lens}'
        os.makedirs(folder, exist_ok = True)
        fits_title = f'{fits.curve.title_lowercase} fit' if not all_fits else f'all {fits.curve.title_lowercase} fits'
        if categories_title is None:
            categories_title = f'{fits.category}' if categories is None else f"all {names['category']['plural']}"
        if title_info is None:
            ax.set_title(f"{names['figure title base']}, {self.group},\n{categories_title} ({fits_title})")
        else:
            ax.set_title(f"{names['figure title base']}, {self.group},\n{categories_title}, {title_info} ({fits_title})")

        show_all(fits, **show_all_args)
        if autozoom:
            if fits.category is not None:
                category = fits.category
                optima = self.category_optima[category]
            else:
                optima = self.group_optima
            bottom, top = optima['y']
            left, right = optima['x']
            zoom(bottom, top, left, right)
        else:
            zoom(lens = lens, margins = Margins(bottom = 0), bottom = 0, zoom_axis = 0)
        offscreen_marks_transparent(fits)
        plt.savefig(f'{folder}/{filename}', dpi = 300)
        show_all(fits, False, **show_all_args)
    def capture_all(self, capture_args, filename_args, presetzoom_folder, autozoom_folder):
        zoom_settings = Desk.zoom_settings
        capture, get_capture_filename = self.capture, self.get_capture_filename
        for setting in zoom_settings:
            if setting == 'autozoom':
                capture(autozoom = True, folder = autozoom_folder, filename = get_capture_filename(autozoom = True, **filename_args), **capture_args)
            else:
                assert setting == 'zoom', f'{setting} in {zoom_settings=} is unrecognized.'
                for lens in zoom_settings[setting]:
                    lens = float(lens)
                    capture(lens = lens, folder = presetzoom_folder, filename = get_capture_filename(lens = lens, **filename_args), **capture_args)
    
    def fit_diff_ev_least_sq(self, curve, bounds, x, y, category, sample, other_args, color = 'black', iterations = None):
        iterations, save_candidates, group, fits, lines_xdata = self.iterations, self.save_candidates, self.group, self.fits, self.lines_xdata
        ax = self.ax
        candidates_individual, winners_individual_paths = self.candidates_individual, self.winners_individual_paths
        capture_all, show_all = self.capture_all, self.show_all
        if iterations is None:
            iterations = self.iterations
        
        
        fit_input = OD({ 'func': SSE, 'args': (x, y, curve), 'bounds': bounds, **other_args })
        
        special = sample in special_samples
        if special:
            candidates_special, winners_special_paths = self.candidates_special[sample], self.winners_special_paths[sample]
        
        curve_name = curve.title
        curve_category = f'{curve_name}_{category}'
        
        candidates = Fits(group, category, DE_leastsquares, curve, sample, color, ax = ax)
        for iteration in range(iterations):
            fit_output = diff_ev(**fit_input)
            y_model = curve(lines_xdata, *fit_output.x)
            fitline_candidate = {
                'line2d': plt.plot(lines_xdata, y_model, color = color)[0],
                'marks': {} }
            candidates.add_fit(fit_input, fit_output, fitline_candidate)
        best = None
        for index, fit in enumerate(candidates):
            if best is None or fit.RMSE < candidates[best].RMSE:
                best = index
        candidates.set_winner(best).winner_setup()

        winner_fits, losers_fits = candidates.separate_winner()

        fit_output = candidates.winner
        DE_leastsquares_fits = fits[group][category][DE_leastsquares]
        DE_leastsquares_fits[curve][sample] = fit_output

        def delete_losers():
            for fit in losers_fits:
                line = fit.line
                line['line2d'].remove()
                for mark in line['marks'].values(): mark.remove()
                del line
        
        if ((not self.save_averaged and sample == 'Averaged') or
            (not self.save_combined and sample == 'Combined') ):
            show_all(candidates, False)
            delete_losers()
            return fit_output
        
        filename_args = {'curve': curve, 'category': category, 'sample': sample, 'special': special}
        
        if save_candidates:
            capture_args = {'fits': candidates, 'marks_visible': False, 'legend_visible': False}
            if special: capture_args['title_info'] = special_samples[sample]['lowercase']
            capture_args['errorbars_visible'] = (sample == 'Averaged')
            presetzoom_folder = capture_args['folder'] = f'{candidates_individual["Preset"]}/{curve_category}' if not special else f'{candidates_special["Preset"]}'
            autozoom_folder = capture_args['folder'] = f'{candidates_individual["Autozoom"]}/{curve_category}' if not special else f'{candidates_special["Autozoom"]}'
            capture_all(capture_args, filename_args, presetzoom_folder, autozoom_folder)
        
        delete_losers()

        capture_args = {'fits': winner_fits}
        if special: capture_args['title_info'] = special_samples[sample]['lowercase']
        capture_args['errorbars_visible'] = (sample == 'Averaged')
        presetzoom_folder = f'{winners_individual_paths["Preset"]}/{curve_category}' if not special else f'{winners_special_paths["Preset"]}'
        autozoom_folder = f'{winners_individual_paths["Autozoom"]}/{curve_category}' if not special else f'{winners_special_paths["Autozoom"]}'
        capture_all(capture_args, filename_args, presetzoom_folder, autozoom_folder)
        
        return fit_output
        
    def offscreen_marks_transparent(self, fits: Fits):
        """
        Makes any mark in "fits" transparent if it's out of bounds.
        """
        ax = self.ax
        for fit in fits:
            line = fit.line
            marks = line['marks']
            for mark in marks.values():
                x, y = tuple(mark.get_offsets()[0])
                left, right = ax.get_xlim()
                bottom, top = ax.get_ylim()
                if any((x < left, x > right, y < bottom, y > top)):
                    mark.set_alpha(0.5)
                else:
                    mark.set_alpha(1)
    
    def apply_margins(self, bottom, top, left, right, margins):
        margins_type = type(margins)
        if margins_type is Margins:
            bottom_margin, top_margin, left_margin, right_margin = margins.bottom, margins.top, margins.left, margins.right
        elif issubclass(margins_type, tuple):
            bottom_margin, top_margin, left_margin, right_margin = margins
        else:
            assert issubclass(margins_type, dict), f'Cannot recognize type {margins_type} of {margins=}.'
            bottom_margin, top_margin, left_margin, right_margin = margins['bottom'], margins['top'], margins['left'], margins['right']
        height, width = (top - bottom), (right - left)
        bottom -= height * bottom_margin
        top += height * top_margin
        left -= width * left_margin
        right += width * right_margin
        return bottom, top, left, right
    
    def zoom(self, bottom = None, top = None, left = None, right = None, lens = 1, margins = None, zoom_axis = 0.5):
        ax = self.ax
        if margins is None: margins = default_margins
        
        x_min, x_max = experiment_optima['x']
        if left is None: left = x_min
        if right is None: right = x_max
        y_min, y_max = experiment_optima['y']
        if bottom is None: bottom = y_min
        if top is None: top = y_max
        
        old_height = top - bottom
        new_height = old_height / lens
        delta = old_height - new_height
        bottom += delta * zoom_axis
        top -= delta * (1 - zoom_axis)
        bottom, top, left, right = self.apply_margins(
            bottom, top, left, right,
            margins = margins )
        
        ax.set_ylim(bottom, top)
        if ax.get_xlim() != (left, right):
            ax.set_xlim(left, right)
        
        return bottom, top, left, right

    @staticmethod
    def legend_sizing(ax):
        # https://stackoverflow.com/questions/4700614/how-to-put-the-legend-outside-the-plot/43439132#43439132
        window_origin, window_space = {'x': 0, 'y': 0}, {'x': 1, 'y': 1}
        ax_box = ax.get_position()
        ax_origin = OD({'x': ax_box.x0, 'y': ax_box.y0})
        ax_space_initial = OD({'x': ax_box.width, 'y': ax_box.height})
        window_margins_initial = {'right': (window_space['x'] - (ax_origin['x'] + ax_space_initial['x']))}
        scaling_factors = {'x': 0.8, 'y': 1}
        ax_space_scaled = {'x': (scaling_factors['x'] * ax_space_initial['x']), 'y': (scaling_factors['y'] * ax_space_initial['y'])}
        window_margins_final = {'right': (window_margins_initial['right'] + (ax_space_initial['x'] - ax_space_scaled['x']))}
        windowspace_to_axspace = lambda width, height: {'width': width/ax_space_scaled['x'], 'height': height/ax_space_scaled['y']}
        new_legend_dimensions = windowspace_to_axspace( window_margins_final['right'], 1 )
        figure_box = [*ax_origin.values(), *ax_space_scaled.values()]
        legend_box = (1, 0.2, new_legend_dimensions['width'], new_legend_dimensions['height'])
        return figure_box, legend_box
    
    def set_legend(self, hidden = None, visible = True, errorbars_visible = True, categories = None, color = None):
        ax, legend_handles_labels, legend_categories = self.ax, self.legend_handles_labels, self.legend_categories
        legend_sizing = self.legend_sizing
        figure_box = self.figure_box
        if figure_box is None:
            self.figure_box_initial = ax.get_position()
            figure_box, legend_box = legend_sizing(ax)
            default_legend_kwargs['bbox_to_anchor'] = legend_box
            self.figure_box = figure_box
        
        errorbars_visible = errorbars_visible and visible

        if hidden is not None:
            new_legend_handles_labels = OD()
            no_data = OD({ 'x': tuple(), 'y': tuple() })
            no_line = { 'linestyle': 'None', 'color': 'black' }
            for style_dict in (*mark_styles.values(), scatter_styles['default']['ignore']):
                label, style = style_dict['title'], style_dict['style']

                invisible_label = label if label.startswith('_') else f'_{label}'
                visible_label = label[1:] if label.startswith('_') else label
                newlabel = invisible_label if label in hidden else visible_label
                
                is_new = False
                if visible_label in legend_handles_labels:
                    handle = legend_handles_labels[visible_label]
                elif invisible_label in legend_handles_labels:
                    handle = legend_handles_labels[invisible_label]
                else:
                    handle = None
                    is_new = True

                fill, outline_color = style['fill'], style['outline']
                if outline_color == 'match': outline_color = 'black'
                matchfill = 'match' in fill
                hatch = 'hatch' in style
                if matchfill: assert hatch is False, 'Marks cannot use matched fill and hatching at the same time.'
                
                if hatch:
                    if is_new:
                        marker = { 'marker': style['shape'], 's': style['box_area'], 'c': 'white', 'edgecolors': outline_color, 'hatch': style['hatch'] }
                        handle = plt.scatter(**no_data, **marker)
                    new_legend_handles_labels[newlabel] = handle
                    continue
                
                marker = { 'marker': style['shape'], 'markersize': np.sqrt(style['box_area']), 'markeredgecolor': outline_color, 'fillstyle': 'full' }
                if matchfill:
                    if color is None: marker.update(legend_match_combos[fill])
                    else: marker.update({ 'markerfacecolor': color })
                else: marker.update({ 'markerfacecolor': fill })
                
                if is_new:
                    new_legend_handles_labels[newlabel], = plt.plot(*no_data.values(), **no_line, **marker)
                    continue
                assert type(handle) is Line2D
                handle.set(**marker)
                new_legend_handles_labels[newlabel] = handle
            if categories is not None:
                new_legend_handles_labels.update({key: value for key, value in legend_categories.items() if key in categories})
            legend_handles_labels.clear()
            legend_handles_labels.update(new_legend_handles_labels)
        with warnings.catch_warnings(record=True):
            labels_handles = tuple(map(list, zip(*legend_handles_labels.items())))
            if len(labels_handles) != 0:
                labels, handles = labels_handles
                legend = ax.legend(handles, labels, **default_legend_kwargs)
            else:
                legend = ax.legend(**default_legend_kwargs)
            legend.set_visible(visible)
            if hasattr(self, 'errorbars_text'): self.errorbars_text.set_visible(errorbars_visible)
            if visible: ax.set_position(figure_box)
            else: ax.set_position(self.figure_box_initial)
    def show_marks_on_legend(self, curve = None, marks_visible = True, legend_visible = True, errorbars_visible = False, categories = None, show_ignored = True, color = None):
        if curve is not None:
            if hasattr(curve, 'marks') is False: marks_visible = False
            styles = curve.styles
            show = curve.marks if marks_visible else []
        else:
            styles = mark_styles
            show = styles.keys() if marks_visible else []
        if not show_ignored: styles = styles | {'ignore': scatter_styles['default']['ignore']}
        hidden = tuple(style['title'] for mark, style in styles.items() if mark not in show)
        self.set_legend(hidden = hidden, visible = legend_visible, errorbars_visible = errorbars_visible, categories = categories, color = color)