from time import monotonic, perf_counter

class Timer(list):
    def __init__(self):
        initial_time = monotonic(), perf_counter()
        self.initial_time = initial_time
        self.time = initial_time
        self.description = None
        super().__init__()
    def format_time(self, time):
        return divmod(time, 60)
    def save_time(self, description = None, ending = False):
        "Saves the current time. If a description is provided, a message will be printed."
        initial_time = self.initial_time
        format_time = self.format_time

        new_monotonic, new_perf_counter = monotonic() - self.initial_time[0], perf_counter() - initial_time[1]
        self.time = new_monotonic, new_perf_counter
        self.append((new_monotonic, new_perf_counter, description))

        message = '\nBeginning' if not ending else '\nEnding'
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