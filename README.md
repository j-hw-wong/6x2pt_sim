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

## Instructions for Use

To execute the simulation that generates a mock $3\times2\mathrm{pt}$ or $6\times2\mathrm{pt}$ data vector, use the 
`run_sim.py` script. The instructions to use this are as follows:

1. Set up a config file detailing the simulation and cosmological parameters for the synthetic data. This can be made
using a template `.ini` file in the `set_config` directory. Here, you will find further information on each parameter
and how they can be defined.
2. In `run_sim.py`, set the `pipeline_variables_path` variable to the path of your template config file.
3. If you want to save all map by-products from `Flask`, set `clean=False` in the `main` function in `run_sim.py`.
Otherwise, set `clean=True`, which will delete the maps (recommended to save disk space).
4. If there are some steps of the analysis that are not needd (e.g. if you just want the maps but don't want to measure 
the Pseudo- $C_{\ell}$ power spectra), comment out the relevant parts of the chain in `run_sim.py`.
5. Execute `run_sim.py` after ensuring that Python dependencies/requirements are met (see `requirements.txt` for further
details).

To execute the parameter sampling to constrain $w_{0}w_{a}\mathrm{CDM}$ parameters and/or nuisance parameters from the
synthetic data generated from `run_sim.py`, use the `run_likelihood.py` script. The instructions to use this are as
follows:

1. Set up a config file detailing the simulation and cosmological parameters for the synthetic data. This can be made
using a template `.ini` file in the `set_config` directory. Here, you will find further information on each parameter
and how they can be defined. Note that the config file does not need to be identical to the config file used for 
simulating the synthetic data (useful for exploring biases etc). The format of the config file must be the same, hoever.
**Also note that for parameters that are sampled over (see steps 2, 3), the sampler will 
overwrite the fiducial values defined in the `.ini` file.**
2. In `run_likelihood.py`, set the `pipeline_variables_path` variable to the path of your template config file for the
sampling routine.
3. Set the `covariance_matrix_type` in `run_likelihood.py` to either `analytic` or `numerical` (to use either an
analytic or numerical covariance matrix). If this is `numerical`, the sampler will find a numerical covariance matrix
saved on disk that has been generated as an automatic by-product of `run_sim.py`. If `analytic` is chosen, the code will
generate a (mode-coupled) analytic covariance matrix using the improved narrow kernel approximation.
4. Set and define the parameters to sample over, and their priors. In order to define a parameter to sample, you will
need to append a tuple to the `priors` list (examples are shown in the `run_likelihood.py` script). The first element
in the tuple is a string describing the given parameter. The string must match the string name convention in the config
`.ini` file (e.g. ``w0`` for $w_{0}$). The second element of the tuple is the prior. For a uniform prior, this is just
another tuple, e.g. `(a,b)` for prior boundaries `a,b`, and for a Gaussian prior, you can use 
`scipy.stats.norm(loc=mu,scale=sigma)`.
5. Note that the sampling for some parameters is dependent on tomographic bin. These are:

   * `b1` - linear galaxy bias term
   * `m` - shear bias
   * `Delta_z` - photo-z uncertainty
   * `A1` - IA tidal alignment amplitude term
   
   for these parameters, you will need to add a parameter/prior for each bin, labelling with an underscore, e.g 
  `[m_1, m_2, m_3]` for the $m$ -bias parameters for an analysis with 3 tomographic bins. **Also note that if you want
  to sample through a per-bin value for any of these parameters, you must then set 
  `bi_marg, mi_marg, Dzi_marg, A1i_marg=True` (respectively) in the `sampler.execute` function in `run_likelihood.py`.**
6. After setting the priors, execute the parameter sampling via nautilus by running `run_likelihood.py` (again after 
ensuring that Python dependencies/requirements are met - see `requirements.txt` for further details).
7. Generate corner plots using the `plotting.plot_posteriors.py` routines. An example for using this is shown in the
`run_likelihood.py` script. 






