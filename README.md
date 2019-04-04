# Audio_2AFC_Analysis
Simulations and analysis for our auditory change-point task
# Repository Structure
In the `Tutorials/` folder, we gather code that illustrates important concepts in our work.

The `simulations.ipynb` file is a Jupyter notebook that contains our current research notes and
figures.

The `mmcomplexity.py` file is a Python module in which we store the important functions and
classes that we define.

# Development Workflow
- All contributors pull/push from `dev`, and work on individual feature branches. 
- Commits must be frequent.
- All data and images should be tracked with [`git-lfs`](https://git-lfs.github.com/).
- Only tested code should be merged to `dev`.
- Only stable code should be merged to `master`.
- All functions and classes should be commented with [docstrings](https://en.wikipedia.org/wiki/Docstring#Python).
- I'll try to generate automatic documentation with [Sphinx](http://www.sphinx-doc.org/en/master/usage/quickstart.html).
- If we ever need a new module in our `conda` environment, we should update the `.yml` file in this repo.
# Reproduce Our `conda` Environment
- [Install `conda`](https://docs.anaconda.com/anaconda/install/) if it is not already installed on your computer.
- Make sure it is up to date by typing `$ conda update conda` in a terminal window (note: don't type the dollar sign, this just signifies that it is a terminal command).
- Get our `conda_environment.yml` file somewhere on your computer (the usual thing to do is to clone this repo)
- Follow [these 3 steps](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file) on the command line.  

Note: If your newly created environment doesn't show up in your list of Jupyter kernels, follow [these steps](https://stackoverflow.com/a/44786736).
