<h1>6x2pt_sim - Simulation and Combined Cosmological Analysis of Next-Generation Weak Lensing, Galaxy Clustering, and 
CMB Lensing </h1>

This is a pipeline that generates a mock data vector for a $6\times2\mathrm{pt}$ observation - the joint combination of 
galaxy weak lensing/cosmic shear, photometric galaxy clustering, and CMB lensing. From a fiducial cosmology, we:

* generate correlated (tomographic) realisations of the $6\times2\mathrm{pt}$ signal from their harmonic power spectra, 
using `Flask` (<https://github.com/hsxavier/flask>, Xavier et al. 2016) and `CCL` (<https://github.com/LSSTDESC/CCL>, 
Chisari et al. 2019), and their dependencies. The data vector includes both contributions from the cosmological signal, 
and a range of systematic effects (see [reference] and the `set_config` module for more details).
* measure Pseudo- $C_{\ell}$ bandpowers for the $6\times2\mathrm{pt}$ signal over a given survey footprint (using 
`NaMaster`, <https://github.com/LSSTDESC/NaMaster>, Alonso et al. 2019) 
* propagate the measured $6\times2\mathrm{pt}$ bandpowers into cosmological parameter constraints on 
$w_{0}w_{a}\mathrm{CDM}$. This includes:

    * a construction of the covariance matrix - either numerical from the simulations, or analytic, using the improved 
  narrow kernel approximation (Nicola et al. 2021)
    * a Gaussian likelihood analysis to derive Bayesian parameter constraints on $w_{0}w_{a}\mathrm{CDM}$ - performed 
  using nested sampling with `nautilus` (<https://nautilus-sampler.readthedocs.io/en/latest/>, Lange 2023). This 
  parameter sampling can be performed using HPC (see reference example below).

Each of these are controlled by the `catalogue_sim`, `pcl_measurement`, and `likelihood` modules respectively. This
codebase builds on routines developed in <https://github.com/j-hw-wong/SWEPT>, 
<https://github.com/robinupham/gaussian_cl_likelihood>, and <https://github.com/robinupham/angular_binning>.

A standard analysis should typically simulate the $6\times2\mathrm{pt}$ signal, then immediately measure the Pseudo- 
$C_{\ell}$ bandpowers, and finally delete the maps that were made for the realisation. Since at least
$\mathcal{O}(1000\mathrm{s})$ of realisations would be needed for deriving the numerical covariance matrix, this
ensures that disk space is not overloaded. This analysis is performed by executing the `run_sim.py` script, which
reads in simulation paramaters that are defined in a config file (examples are shown in the `set_config` directory). 
Please see each of these modules for further documentation and guidance for use.

Following the generation of a simulated $6\times2\mathrm{pt}$ data vector and a corresponding covariance matrix, the
likelihood analysis can be performed using the `run_likelihood.py` script. This reads in simulation/cosmological
parameters from a config file (this does not have to be identical to the parameters that were used to generate the 
synthetic data, in case you want to explore biases etc.) and also defines which other parameters are sampled and
marginalised over. Please read the guidelines in this script for further details on its use.


