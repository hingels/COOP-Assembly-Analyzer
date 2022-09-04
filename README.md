# CoOP-Assembly-Analyzer
A Python program designed to fit CoOP assembly data, particularly from thioflavin T measurements.

To install:
- Download the source code (CoOP-Assembly-Analyzer-main.zip), extract/decompress/unzip it, and move it wherever you'd like. The unzipped folder should be named CoOP-Assembly-Analyzer-main.
- Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html).
- Following Miniconda documentation according to your operating system, open the appropriate command line.
- Run the command `cd [path]`, replacing `[path]` with CoOP-Assembly-Analyzer-main's path.
- Run the command `conda env create -f environment_from-history.yml` to install the conda environment.
  - If this doesn't work, try running the command `cd CoOP-Assembly-Analyzer-main` and then trying again.
- To run the program, see below.

To run:
- Using your file browser in the Input folder, replace data.xlsx with your own data file and adjust the settings in config.md.
- Open the command line as described above, under "To install."
- Activate the conda environment with the command `conda activate ThT_analysis`.
- Run the command `cd [path]`, replacing `[path]` with CoOP-Assembly-Analyzer-main's path.
- Run the command `python app.py`. The results will appear in the Output folder!
