from time import monotonic, perf_counter
from fitter import Fitter

class Timer(list):
    def __init__(self):
        initial_time = monotonic(), perf_counter()
        self.initial_time = initial_time
        self.time = initial_time
        self.description = None
        super().__init__()
    def format_time(self, times):
        try: iter(times)
        except TypeError:
            return divmod(times, 60)
        else:
            return tuple(divmod(time, 60) for time in times)
    def save_time(self, description = None, ending = False):
        "Saves the current time. If a description is provided, a message will be printed."
        initial_time = self.initial_time
        format_time = self.format_time

        new_monotonic, new_perf_counter = monotonic() - self.initial_time[0], perf_counter() - initial_time[1]
        self.time = new_monotonic, new_perf_counter
        self.append((new_monotonic, new_perf_counter, description))

        message = '\n\nBeginning' if not ending else '\n\nEnding'
        if description is None: message += ' '
        else: message += f' {description} '
        formatted_time = format_time(new_perf_counter)
        message += f'at {int(formatted_time[0])} minute(s) and {formatted_time[1]} seconds. (time.perf_counter())'
        print(message)

        try: old_monotonic, old_perf_counter, old_description = self[-1]
        except IndexError: return
        if old_description is None: return
        old_description = old_description[0].upper() + old_description[1:]
        duration = format_time(new_perf_counter - old_perf_counter)
        print(f'{old_description} took {int(duration[0])} minute(s) and {duration[1]} seconds. (time.perf_counter())')
    def end(self, save_time = True, write_note = False):
        if save_time: self.save_time(ending = True)
        paths, config = Fitter.paths, Fitter.reader.config
        elapsed_perf_counter, elapsed_monotonic = self.format_time(self.time)
        if write_note:
            with open(paths['output_path_base'] + '/Notes.md', mode = 'w') as notes:
                notes.write('\n'.join((
                    "Notes:",
                    f"- {config.iterations} iterations were used to generate this output.",
                    "- Run time:",
                    f"\t- Measured by time.monotonic(): {int(elapsed_monotonic[0])} minute(s) and {elapsed_monotonic[1]} seconds.",
                    f"\t- Measured by time.perf_counter(): {int(elapsed_perf_counter[0])} minute(s) and {elapsed_perf_counter[1]} seconds.") ))
        print(f'\nFinished.\nTime elapsed (time.monotonic()): {int(elapsed_monotonic[0])} minute(s) and {elapsed_monotonic[1]} seconds.\nTime elapsed (time.perf_counter()): {int(elapsed_perf_counter[0])} minute(s) and {elapsed_perf_counter[1]} seconds.')