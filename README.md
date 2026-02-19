<h1>6x2pt_sim - Simulation and Combined Cosmological Analysis of Next-Generation Weak Lensing, Galaxy Clustering, and 
CMB Lensing </h1>

This is a pipeline that generates a mock data vector for a $6\times2\mathrm{pt}$ observation - the joint combination of galaxy 
weak lensing/cosmic shear, photometric galaxy clustering, and CMB lensing. From a fiducial cosmology, we:

<ul>
<li>generate correlated (tomographic) realisations of the $6\times2\mathrm{pt}$ signal from their harmonic power spectra, using 
`Flask` (<https://github.com/hsxavier/flask>, Xavier et al. 2016) and `CCL` (<https://github.com/LSSTDESC/CCL>, 
Chisari et al. 2019), and their dependencies. The data vector includes both contributions from the cosmological signal, 
and a range of systematic effects (see [reference] and the `set_config` module for more details).</li>
<li>measure Pseudo-$C_{\ell}$ bandpowers for the $6\times2\mathrm{pt}$ signal over a given survey footprint (using `NaMaster`, 
<https://github.com/LSSTDESC/NaMaster>, Alonso et al. 2019)</li>
<li>propagate the measured $6\times2\mathrm{pt}$ bandpowers into cosmological parameter constraints on $w_{0}w_{a}\mathrm{CDM}$. This 
includes:

<ul>
<li>a construction of the covariance matrix - either numerical from the simulations, or analytic, using the 
improved narrow kernel approximation (Nicola et al. 2021)</li>
<li>a Gaussian likelihood analysis to derive Bayesian parameter constraints on $w_{0}w_{a}\mathrm{CDM}$ - performed using nested
sampling with `nautilus` (<https://nautilus-sampler.readthedocs.io/en/latest/>, Lange 2023). This parameter sampling 
can be performed using HPC (see reference example below).</li>
</ul>
</li>

</ul>

