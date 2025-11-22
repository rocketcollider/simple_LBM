# Basic 2D Lattice Boltzmann Simulation

This project simulates fluids by dividing the simulation area into cells and tracking the velocity-distribution of the fluid inside each cell.

The method fits well for GPUs, but this implementation uses `numpy` and thereby runs on CPU. Be patient!

My goal was to implement a bunch of physical phenomena as best I can (and continue to do so), *not* break any speed records. Maybe precision records, one day ... ;)

## Development Strategy
I start from papers solving a particular challenge I encountered during development. (Sources coming soon!)

A fitting solution is tested for with a fresh test-scenario, initially with a simple setup, simulation loop and visual output. From here, implementation strategies are compared and testing-strategies developed. This process builds test and code in parallel.

The simulator is focused on highly readable, simple code. Precision and accuracy is a goal, though bare functionality is more important. Performance is not a concern.

## Running the showcase
A simple showcase scenario is provided. It's an airbubble rising in a column of compressible fluid. This demonstrates the ability to simulate immiscible fluids of different density.

 1. clone this repo:

        git clone https://github.com/rocketcollider/simple_LBM
        cd simple_LBM

    optional: install virtual environment

        python3 -m venv venv             #optional
        . venv/bin/activate              #optional

 2. install dependencies:

        pip install -r requirements.txt

    **BEWARE** to install a [backend for matplotlib](https://matplotlib.org/stable/users/explain/figure/backends.html) as well! This depends on your operating system and requires additional packages to be installed. `requirements.txt` comes with `PyQt6` and `PyQt5`, resulting in `qtagg` for matplotlib. Other backades require additonal pip *and* OS packages to be installed.

 3. run the showcase!

        python simshowcase.py 

    be patient! depending on your machine, this might take a while.

 4. enjoy the view!

    ![a rising bubble breaking of smaller bubbles in its wake](bubble_showcase.png "Left is density, right is upward-velocity.")
    Roughly simulation-step 6000.

## Running the tests
Install dependencies as described above. (testing- and production-depencies are not separated right now)

Run `pytest` to run everything. But I recommend looking through the test-files first and running specific ones.

Most tests have options in the test-files, such as enabling visualisation, running fast or precise tests, and so forth. I consider reading the test-file as part of the documentation.

## Sources
COMING SOON. (not all papers I read ended up in the code.)