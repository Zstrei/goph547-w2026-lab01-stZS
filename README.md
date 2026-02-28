# goph547-w2026-lab01-stZS
Semester: W2026<br>
Instructor: B. Karchewski<br>
Author(s): Zac Strei

This repository contains the code and outputs for Lab 01 of GOPH 547 (Winter 2026). The purpose of this lab was to implement forward modelling of gravitational potential and vertical gravity effect for progressively more complex subsurface mass configurations. The modelling exercises demonstrate how source geometry, observation height, and survey resolution influence gravity anomaly characteristics, and highlight the inherent non-uniqueness of potential-field methods.

The recommended way to download or clone the repository is by navigating to the desired directory in a terminal using the command:

cd /path/to/directory

Then run either:

git clone <repository-url>

or

gh repo clone <repository-name>

if the GitHub CLI is installed. This will download the repository files into a new local directory. The repository can also be downloaded by clicking the green CODE button on GitHub and selecting “Download ZIP.” Be sure to navigate to the directory where the files will be stored before cloning.

It is recommended to set up a virtual environment when running the files. To create a virtual environment, navigate to the repository directory and run:

python -m venv .venv

To activate it in PowerShell, run:

..venv\Scripts\activate

On macOS/Linux, run:

source .venv/bin/activate

After activation, “(.venv)” should appear in the terminal.

Install the required packages using:

pip install numpy matplotlib setuptools

The script driver_single_mass.py, located in the examples folder, performs forward modelling for a single buried point mass of 1.0e7 kg located at a depth of 10 m. It computes gravitational potential (U) and vertical gravity effect (gz) over a grid extending from -100 m to +100 m in both the x and y directions. The fields are evaluated at observation heights of 0 m, 10 m, and 100 m using grid spacings of 5 m and 25 m. The script generates contour plots and saves the figures to the outputs folder.

The script driver_mass_anomaly.py extends the modelling to more complex configurations. It generates multiple random point-mass sets constrained to have the same total mass and centroid as the single-mass case, demonstrating superposition and non-uniqueness. It then loads a three-dimensional distributed density model, computes total mass, barycentre, maximum cell density, and mean densities, and performs forward modelling by treating each cell as a point mass. The script calculates vertical gravity effect at elevations of 0 m, 1 m, 100 m, and 110 m, approximates vertical derivatives using finite differences, and saves all generated plots to the outputs folder.

All modelling is performed in a Cartesian coordinate system with gravity defined as positive downward and expressed in SI units. The regional gravity field is neglected, and the synthetic data are assumed to be noise-free.