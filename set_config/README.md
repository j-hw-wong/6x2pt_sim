# Here we will set parameters to define the global cosmology and simulation parameters for the ```6x2pt_sim``` analysis.

The global cosmology and simulation parameters are used to make a fiducial 6x2pt data vector, which is fed into `Flask`
in order to make realisations on the sky. Systematic effects are added either on the power spectrum or map level (see
Wong 2025, PhD thesis for further details). 

The parameters are defined using a config file - please see one of the `set_variables.ini` files for guidance on 
the different parameters and how they propagate into the analysis. Note that the format of the config file is identical
for both the data vector generation, and the parameter sampling. However, for the purposes of e.g. ``mismodelling``
effects, you may wish to have different parameter/measurement values for the sampling routine. In this case, you can 
point the sampler to a different config file than was used for the data vector simulation. Note that for parameters 
that are actually sampled through, the sampler will overwrite values defined in the config file.

Also note that the `flask_3x2pt.config` file does not need to be changed. A file in this format is necessary to run
`Flask`, but parameters are written onto the `flask_3x2pt.config` file from the `set_variables.ini` file.