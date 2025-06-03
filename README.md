# BinaryLensWithBinarySource

BinaryLensWithBinarySource provides a Python toolkit for modelling gravitational microlensing events.
It bundles two related packages:

- **BinaryLensFitter** – core routines to fit a binary lens light curve using MCMC methods.
- **BinaryLensBinarySourceFitter** – an extension to handle binary source events by wrapping two `Binary_lens_mcmc` objects.

The modules implement parameter handling, chi–square likelihoods, plotting utilities and wrappers around GPU/C code for fast magnification map calculations.
A helper script `circles.py` adds a convenient function for drawing circles in matplotlib.

## Repository Layout

```
BinaryLensBinarySourceFitter/  # binary-source wrapper
BinaryLensFitter/             # base fitter and GPU code
circles.py                    # plotting utility
README.md                     # this file
```

### BinaryLensFitter
The main class `Binary_lens_mcmc` lives here alongside routines for parallax, orbital motion and Gaussian process modelling.
C utilities for the GPU code reside under `BinaryLensFitter/include/`.
Note that `_gpu_mag_maps_cuda.py` and `_gpu_image_centred.py` contain a hard-coded path to `nrutil.h` which must be edited after installation.

### BinaryLensBinarySourceFitter
A front-end for fitting events with two sources. It sets up two `Binary_lens_mcmc` instances and synchronises their parameters so they can be fitted together.
Plotting helpers allow you to display caustics, trajectories and light curves.

## Dependencies


The project depends on common scientific libraries (`numpy`, `scipy`, `matplotlib`), the MCMC sampler `emcee` (and optionally `zeus`), and GPU support via `pycuda` together with the compiled C code in `include/`. Additional modules such as `celerite` for Gaussian processes, `corner` for corner plots, `VBBinaryLensing` for lensing calculations and `Metropolis_Now` in the parallax module are also required.

All of these packages are listed in `requirements.txt` which can be installed with:

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from BinaryLensFitter import Binary_lens_mcmc
from BinaryLensBinarySourceFitter import Binary_lens_Binary_source_mcmc

# ``data`` is a dict mapping dataset names to (time, flux, error) arrays
# ``initial_params`` holds starting values for log s, log q, log rho, etc.

fitter = Binary_lens_mcmc(data, initial_params)
# or use Binary_lens_Binary_source_mcmc for binary-source events
```

Use the class methods to run MCMC fits and call the plotting utilities to visualise results.

## Notes

See `BinaryLensFitter/README` for details about adjusting `nrutil.h` paths needed by the GPU code.
