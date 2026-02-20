import os
import sys
import time
# import mpi4py
# from mpi4py.futures import MPIPoolExecutor
import corner
import nautilus
import configparser
import numpy as np
import pyccl as ccl
import pymaster as nmt
import likelihood.mask as mask
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from likelihood.model_pcl import PCl_bandpowers_1x2pt, PCl_bandpowers_3x2pt, PCl_bandpowers_6x2pt
from catalogue_sim import generate_cls
from plotting import spider_plot, plot_posteriors
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['mathtext.fontset'] = 'cm'
# plt.rcParams['font.serif'] = 'cm'
from scipy.stats import norm


def parse_nautilus_params_for_ccl(params):
    """
    nautilus propogates params as a dictionary, with each value an unshaped ndarray
    """

    return {key:val[()] for key, val in params.items()}


def split_ccl_parameters(params, n_zbin, bi_marg=False, b2i_marg=False, mi_marg=False, Dzi_marg=False, A1i_marg=False):

    """
    Parameters to sample may be either cosmology/CCL parameters, or nuisance parameters that come into the model later.
    Prepare a dictionary of CCL and nuisance parameters, in the correct format (e.g. a scalar or vector)

    Parameters
    ----------
    params (dict):  Dictionary of parameters
    n_zbin (int):   Number of tomographic redshift bins
    bi_marg (bool): If true, prepare for marginalisation over a bin-dependent b1 (galaxy bias) value
    b2i_marg (bool):    If true, prepare for marginalisation over a bin-dependent b2 (galaxy bias) value
    mi_marg (bool):     If true, prepare for marginalisation over a bin-dependent m-bias value
    Dzi_marg (bool):    If true, prepare for marginalisation over a bin-dependent Delta z (photo-z error) value
    A1i_marg (bool):    If true, prepare for marginalisation over a bin-dependent A1 (IA amplitude) value

    Returns
    -------
    Dictionaries of CCL and nuisance parameters, in the correct format for CCL/the rest of the code
    """

    ccl_params = {}
    nuisance_params = {}

    # Some list of nuisance parameters
    are_nuisance_params = [
        'Omega_m',
        '_b1',
        '_b2',
        '_bs',
        '_A1',
        '_A2',
        '_bTA',
        '_eta1',
        '_eta2',
        '_m',
        '_s0',
        '_s1',
        '_s2',
        '_s3'
    ]

    # nuisance_params['bi_marg'] = bi_marg
    # nuisance_params['mi_marg'] = mi_marg
    # nuisance_params['A1i_marg'] = A1i_marg

    if bi_marg:
        for i in range(n_zbin):
            are_nuisance_params.append('_b1_{}'.format(i+1))
    if b2i_marg:
        for i in range(n_zbin):
            are_nuisance_params.append('_b2_{}'.format(i+1))
    if mi_marg:
        for i in range(n_zbin):
            are_nuisance_params.append('_m_{}'.format(i+1))
    if Dzi_marg:
        for i in range(n_zbin):
            are_nuisance_params.append('_Dz_{}'.format(i+1))
    if A1i_marg:
        for i in range(n_zbin):
            are_nuisance_params.append('_A1_{}'.format(i + 1))
    # are_nuisance_params = [f"m_{_i}" for _i in range(1, Z_N_BIN + 1)] + [f"dz_{_i}" for _i in range(1, Z_N_BIN + 1)]

    for key, val in params.items():
        # Nautilus passes parameters as unshaped arrays
        # JW - I am not sure what these lines do
        # if isinstance(val, np.ndarray):
        #     assert x.shape == (), "Only unshaped arrays are supported"
        #     val = val[()]

        if key in are_nuisance_params:
            nuisance_params[key] = val
        else:
            ccl_params[key] = float(val)

        # I am not sure what this does
        # split_key = key.split("_")

    return ccl_params, nuisance_params


def sampler_config(pipeline_variables_path):

    """
    Create a dictionary of parameters that will be useful to calculate measure Pseudo bandpowers (e.g. number of bins,
    realisations, mask etc.)

    Parameters
    ----------
    pipeline_variables_path (str):  Path to location of pipeline variables file ('set_variables_3x2pt_measurement.ini')

    Returns
    -------
    Dictionary of parameters used by this script to measure 3x2pt Pseudo bandpowers
    """

    config = configparser.ConfigParser()
    config.read(pipeline_variables_path)

    save_dir = str(config['simulation_setup']['SIMULATION_SAVE_DIR'])
    nz_filename = str(config['redshift_distribution']['NZ_TABLE_NAME'])
    nbins = int(config['redshift_distribution']['N_ZBIN'])
    nside = int(float(config['simulation_setup']['NSIDE']))
    no_iter = int(config['simulation_setup']['REALISATIONS'])
    mask_path = str(config['measurement_setup']['PATH_TO_MASK'])
    mask_path_cmb = str(config['measurement_setup']['PATH_TO_CMB_MASK'])
    n_pool = int(config['simulation_setup']['POOL'])

    input_lmin = int(float(config['simulation_setup']['INPUT_ELL_MIN']))
    input_lmax = int(float(config['simulation_setup']['INPUT_ELL_MAX']))

    output_lmin_shear = int(float(config['measurement_setup']['OUTPUT_ELL_MIN_EE']))
    output_lmax_shear = int(float(config['measurement_setup']['OUTPUT_ELL_MAX_EE']))

    output_lmin_galaxy_shear = int(float(config['measurement_setup']['OUTPUT_ELL_MIN_NE']))
    output_lmax_galaxy_shear = int(float(config['measurement_setup']['OUTPUT_ELL_MAX_NE']))

    output_lmin_galaxy = int(float(config['measurement_setup']['OUTPUT_ELL_MIN_NN']))
    output_lmax_galaxy = int(float(config['measurement_setup']['OUTPUT_ELL_MAX_NN']))

    output_lmin_cmbkk = int(float(config['measurement_setup']['OUTPUT_ELL_MIN_CMBKK']))
    output_lmax_cmbkk = int(float(config['measurement_setup']['OUTPUT_ELL_MAX_CMBKK']))

    output_lmin_cmbkk_galaxy = int(float(config['measurement_setup']['OUTPUT_ELL_MIN_CMBKK_N']))
    output_lmax_cmbkk_galaxy = int(float(config['measurement_setup']['OUTPUT_ELL_MAX_CMBKK_N']))

    output_lmin_cmbkk_shear = int(float(config['measurement_setup']['OUTPUT_ELL_MIN_CMBKK_E']))
    output_lmax_cmbkk_shear = int(float(config['measurement_setup']['OUTPUT_ELL_MAX_CMBKK_E']))

    obs_spec = str(config['measurement_setup']['OBS_TYPE'])
    obs_field = str(config['measurement_setup']['FIELD'])

    n_bandpowers = int(float(config['measurement_setup']['N_BANDPOWERS']))
    bandpower_spacing = str(config['measurement_setup']['BANDPOWER_SPACING'])
    accepted_bp_spacings = {'log', 'lin', 'custom'}

    if bandpower_spacing not in accepted_bp_spacings:
        print('Error! Bandpower Spacing Not Recognised - Exiting...')
        sys.exit()

    elif bandpower_spacing == 'log':
        # bp_bin_edges = np.logspace(np.log10(output_lmin + 1e-5), np.log10(output_lmax + 1e-5), n_bandpowers + 1)
        bp_bin_edges_galaxy = np.logspace(np.log10(output_lmin_galaxy + 1e-5), np.log10(output_lmax_galaxy + 1e-5), n_bandpowers + 1)
        bp_bin_edges_galaxy_shear = np.logspace(np.log10(output_lmin_galaxy_shear + 1e-5), np.log10(output_lmax_galaxy_shear + 1e-5), n_bandpowers + 1)
        bp_bin_edges_shear = np.logspace(np.log10(output_lmin_shear + 1e-5), np.log10(output_lmax_shear + 1e-5), n_bandpowers + 1)
        bp_bin_edges_cmbkk = np.logspace(np.log10(output_lmin_cmbkk + 1e-5), np.log10(output_lmax_cmbkk + 1e-5), n_bandpowers + 1)
        bp_bin_edges_cmbkk_galaxy = np.logspace(np.log10(output_lmin_cmbkk_galaxy + 1e-5), np.log10(output_lmax_cmbkk_galaxy + 1e-5), n_bandpowers + 1)
        bp_bin_edges_cmbkk_shear = np.logspace(np.log10(output_lmin_cmbkk_shear + 1e-5), np.log10(output_lmax_cmbkk_shear + 1e-5), n_bandpowers + 1)

    elif bandpower_spacing == 'lin':
        # bp_bin_edges = np.linspace(output_lmin + 1e-5, output_lmax + 1e-5, n_bandpowers + 1)
        bp_bin_edges_galaxy = np.linspace(output_lmin_galaxy + 1e-5, output_lmax_galaxy + 1e-5, n_bandpowers + 1)
        bp_bin_edges_galaxy_shear = np.linspace(output_lmin_galaxy_shear + 1e-5, output_lmax_galaxy_shear + 1e-5, n_bandpowers + 1)
        bp_bin_edges_shear = np.linspace(output_lmin_shear + 1e-5, output_lmax_shear + 1e-5, n_bandpowers + 1)
        bp_bin_edges_cmbkk = np.linspace(output_lmin_cmbkk + 1e-5, output_lmax_cmbkk + 1e-5, n_bandpowers + 1)
        bp_bin_edges_cmbkk_galaxy = np.linspace(output_lmin_cmbkk_galaxy + 1e-5, output_lmax_cmbkk_galaxy + 1e-5, n_bandpowers + 1)
        bp_bin_edges_cmbkk_shear = np.linspace(output_lmin_cmbkk_shear + 1e-5, output_lmax_cmbkk_shear + 1e-5, n_bandpowers + 1)

    else:
        # Bandpower type not recognised
        sys.exit()

    bp_bins_galaxy = nmt.NmtBin.from_edges(
        ell_ini=np.ceil(bp_bin_edges_galaxy).astype(int)[:-1],
        ell_end=np.ceil(bp_bin_edges_galaxy).astype(int)[1:])
    ell_arr_galaxy = bp_bins_galaxy.get_effective_ells()

    bp_bins_galaxy_shear = nmt.NmtBin.from_edges(
        ell_ini=np.ceil(bp_bin_edges_galaxy_shear).astype(int)[:-1],
        ell_end=np.ceil(bp_bin_edges_galaxy_shear).astype(int)[1:])
    ell_arr_galaxy_shear = bp_bins_galaxy_shear.get_effective_ells()

    bp_bins_shear = nmt.NmtBin.from_edges(
        ell_ini=np.ceil(bp_bin_edges_shear).astype(int)[:-1],
        ell_end=np.ceil(bp_bin_edges_shear).astype(int)[1:])
    ell_arr_shear = bp_bins_shear.get_effective_ells()

    bp_bins_cmbkk = nmt.NmtBin.from_edges(
        ell_ini=np.ceil(bp_bin_edges_cmbkk).astype(int)[:-1],
        ell_end=np.ceil(bp_bin_edges_cmbkk).astype(int)[1:])
    ell_arr_cmbkk = bp_bins_cmbkk.get_effective_ells()

    bp_bins_cmbkk_galaxy = nmt.NmtBin.from_edges(
        ell_ini=np.ceil(bp_bin_edges_cmbkk_galaxy).astype(int)[:-1],
        ell_end=np.ceil(bp_bin_edges_cmbkk_galaxy).astype(int)[1:])
    ell_arr_cmbkk_galaxy = bp_bins_cmbkk_galaxy.get_effective_ells()

    bp_bins_cmbkk_shear = nmt.NmtBin.from_edges(
        ell_ini=np.ceil(bp_bin_edges_cmbkk_shear).astype(int)[:-1],
        ell_end=np.ceil(bp_bin_edges_cmbkk_shear).astype(int)[1:])
    ell_arr_cmbkk_shear = bp_bins_cmbkk_shear.get_effective_ells()

    pbl_shear = mask.get_binning_matrix(
        n_bandpowers=n_bandpowers,
        output_lmin=output_lmin_shear,
        output_lmax=output_lmax_shear,
        bp_spacing=bandpower_spacing)

    pbl_galaxy_shear = mask.get_binning_matrix(
        n_bandpowers=n_bandpowers,
        output_lmin=output_lmin_galaxy_shear,
        output_lmax=output_lmax_galaxy_shear,
        bp_spacing=bandpower_spacing)

    pbl_galaxy = mask.get_binning_matrix(
        n_bandpowers=n_bandpowers,
        output_lmin=output_lmin_galaxy,
        output_lmax=output_lmax_galaxy,
        bp_spacing=bandpower_spacing)

    pbl_cmbkk = mask.get_binning_matrix(
        n_bandpowers=n_bandpowers,
        output_lmin=output_lmin_cmbkk,
        output_lmax=output_lmax_cmbkk,
        bp_spacing=bandpower_spacing)

    pbl_cmbkk_galaxy = mask.get_binning_matrix(
        n_bandpowers=n_bandpowers,
        output_lmin=output_lmin_cmbkk_galaxy,
        output_lmax=output_lmax_cmbkk_galaxy,
        bp_spacing=bandpower_spacing)

    pbl_cmbkk_shear = mask.get_binning_matrix(
        n_bandpowers=n_bandpowers,
        output_lmin=output_lmin_cmbkk_shear,
        output_lmax=output_lmax_cmbkk_shear,
        bp_spacing=bandpower_spacing)

    # Read in cosmology parameters
    Omega_c = float(config['cosmology']['Omega_c'])
    Omega_b = float(config['cosmology']['Omega_b'])
    h = float(config['cosmology']['h'])
    #A_s = float(config['cosmology']['A_s'])
    sigma8 = float(config['cosmology']['sigma8'])
    n_s = float(config['cosmology']['n_s'])
    Omega_k = float(config['cosmology']['Omega_k'])
    w0 = float(config['cosmology']['w0'])
    wa = float(config['cosmology']['wa'])

    # Prepare config dictionary
    config_dict = {
        'save_dir': save_dir,
        'nbins': nbins,
        'nside': nside,
        'nz_filename': nz_filename,
        'no_iter': no_iter,
        'n_pool': n_pool,
        'mask_path': mask_path,
        'mask_path_cmb': mask_path_cmb,
        'input_lmin': input_lmin,
        'input_lmax': input_lmax,
        'n_bandpowers': n_bandpowers,
        'bandpower_spacing': bandpower_spacing,
        'output_lmin_shear': output_lmin_shear,
        'output_lmax_shear': output_lmax_shear,
        'pbl_shear': pbl_shear,
        #'bp_bins_shear': bp_bins_shear,
        'ell_arr_shear': ell_arr_shear,
        'output_lmin_galaxy_shear': output_lmin_galaxy_shear,
        'output_lmax_galaxy_shear': output_lmax_galaxy_shear,
        'pbl_galaxy_shear': pbl_galaxy_shear,
        #'bp_bins_galaxy_shear': bp_bins_galaxy_shear,
        'ell_arr_galaxy_shear': ell_arr_galaxy_shear,
        'output_lmin_galaxy': output_lmin_galaxy,
        'output_lmax_galaxy': output_lmax_galaxy,
        'pbl_galaxy': pbl_galaxy,
        #'bp_bins_galaxy': bp_bins_galaxy,
        'ell_arr_galaxy': ell_arr_galaxy,
        'output_lmin_cmbkk': output_lmin_cmbkk,
        'output_lmax_cmbkk': output_lmax_cmbkk,
        'pbl_cmbkk': pbl_cmbkk,
        #'bp_bins_cmbkk': bp_bins_cmbkk,
        'ell_arr_cmbkk': ell_arr_cmbkk,
        'output_lmin_cmbkk_galaxy': output_lmin_cmbkk_galaxy,
        'output_lmax_cmbkk_galaxy': output_lmax_cmbkk_galaxy,
        'pbl_cmbkk_galaxy': pbl_cmbkk_galaxy,
        #'bp_bins_cmbkk_galaxy': bp_bins_cmbkk_galaxy,
        'ell_arr_cmbkk_galaxy': ell_arr_cmbkk_galaxy,
        'output_lmin_cmbkk_shear': output_lmin_cmbkk_shear,
        'output_lmax_cmbkk_shear': output_lmax_cmbkk_shear,
        'pbl_cmbkk_shear': pbl_cmbkk_shear,
        #'bp_bins_cmbkk_shear': bp_bins_cmbkk_shear,
        'ell_arr_cmbkk_shear': ell_arr_cmbkk_shear,
        'obs_spec': obs_spec,
        'obs_field': obs_field,
        'Omega_c': Omega_c,
        'Omega_b': Omega_b,
        'h': h,
        'n_s': n_s,
        #'A_s': A_s,
        'sigma8': sigma8,
        'Omega_k': Omega_k,
        'w0': w0,
        'wa': wa
    }

    return config_dict


def mysplit(s):

    """
    Function to split string into float and number. Used to extract which field and which tomographic bin should be
    identified and collected.

    Parameters
    ----------
    s (str):    String describing field and tomographic bin number

    Returns
    -------
    head (str): String describing field
    tail (float):   Float describing tomographic bin id
    """

    head = s.rstrip('0123456789')
    tail = s[len(head):]
    return head, tail


def generate_pseudo_bps_model(cosmo_params, pipeline_variables_path, config_dict, mixmats, bi_marg=False, b2i_marg=False, mi_marg=False, Dzi_marg=False, A1i_marg=False):

    """
    3x2pt or 6x2pt Pseudo-bandpowers, ordered in a data vector

    Parameters
    ----------
    cosmo_params (dict):    Dictionary of cosmological parameters
    config_dict (dict):     Dictionary describing simulation/measurement parameters. Ones that will be sampled are overwritten
    pipeline_variables_path (str):  Path to location of pipeline variables file (e.g. 'set_variables_3x2pt_measurement.ini')
    mixmats (dict):     Dictionary of 6x2pt spin-0 and spin-2 mixing matrices
    bi_marg (bool):     Marginalise over b1 galaxy bias (True) or not (False). Prepares formatting for parameter input to CCL
    b2i_marg (bool):    Marginalise over b2 galaxy bias (True) or not (False). Prepares formatting for parameter input to CCL
    mi_marg (bool):     Marginalise over shear m-bias (True) or not (False). Prepares formatting for parameter input to CCL
    Dzi_marg (bool):    Marginalise over Delta z photo-z uncertainty (True) or not (False). Prepares formatting for parameter input to CCL
    A1i_marg (bool):    Marginalise over IA amplitude (True) or not (False). Prepares formatting for parameter input to CCL

    Returns
    -------
    Data vector of 3x2pt or 6x2pt Pseudo-bandpowers
    """

    save_dir = config_dict['save_dir']
    nz_filename = config_dict['nz_filename']
    input_lmin = config_dict['input_lmin']
    input_lmax = config_dict['input_lmax']
    nbins = config_dict['nbins']

    output_lmin_shear = config_dict['output_lmin_shear']
    output_lmax_shear = config_dict['output_lmax_shear']

    output_lmin_galaxy_shear = config_dict['output_lmin_galaxy_shear']
    output_lmax_galaxy_shear = config_dict['output_lmax_galaxy_shear']

    output_lmin_galaxy = config_dict['output_lmin_galaxy']
    output_lmax_galaxy = config_dict['output_lmax_galaxy']

    output_lmin_cmbkk = config_dict['output_lmin_cmbkk']
    output_lmax_cmbkk = config_dict['output_lmax_cmbkk']

    output_lmin_cmbkk_galaxy = config_dict['output_lmin_cmbkk_galaxy']
    output_lmax_cmbkk_galaxy = config_dict['output_lmax_cmbkk_galaxy']

    output_lmin_cmbkk_shear = config_dict['output_lmin_cmbkk_shear']
    output_lmax_cmbkk_shear = config_dict['output_lmax_cmbkk_shear']

    obs_spec = config_dict['obs_spec']
    obs_field = config_dict['obs_field']
    n_bp = config_dict['n_bandpowers']

    nz_dat = np.loadtxt(f"{save_dir}{nz_filename}")
    z = nz_dat[:, 0]

    ells = np.arange(input_lmin, input_lmax + 1, 1)

    # Read in cosmology parameters
    Omega_c = config_dict['Omega_c']
    Omega_b = config_dict['Omega_b']
    h = config_dict['h']
    #A_s = config_dict['A_s']
    sigma8 = config_dict['sigma8']
    n_s = config_dict['n_s']
    Omega_k = config_dict['Omega_k']
    w0 = config_dict['w0']
    wa = config_dict['wa']

    # A fiducial cosmology
    fid_cosmo_dict = {
        'Omega_c': Omega_c,
        'Omega_b': Omega_b,
        'h': h,
        #'A_s': A_s,
        'sigma8': sigma8,
        'n_s': n_s,
        'Omega_k': Omega_k,
        'w0': w0,
        'wa': wa,
        'extra_parameters': {'camb':{'dark_energy_model':'ppf'}}
    }

    ccl_params, nuisance_params = split_ccl_parameters(cosmo_params, n_zbin=nbins, bi_marg=bi_marg, b2i_marg=b2i_marg, mi_marg=mi_marg, Dzi_marg=Dzi_marg, A1i_marg=A1i_marg)

    if 'Omega_m' in nuisance_params:
        fid_cosmo_dict['Omega_c'] = nuisance_params['Omega_m'] - Omega_b
    fid_cosmo_dict.update(ccl_params)
    # print(fid_cosmo_dict)
    fid_cosmo = ccl.Cosmology(**fid_cosmo_dict)
    start_time=time.time()

    nz_boundaries = np.loadtxt(f"{save_dir}z_boundaries.txt")

    sampler_systematics = {}

    # bi_marg = nuisance_params['bi_marg']
    if bi_marg:
        b_1 = np.ones(len(z))
        for i in range(nbins-1):
            ids = np.where((z.round(decimals=2) >= nz_boundaries[:, 0][i].round(decimals=2)) & (
                        z.round(decimals=2) < nz_boundaries[:, 2][i].round(decimals=2)))[0]
            b_1[ids] = nuisance_params["_b1_{}".format(i+1)]
        ids_last = np.where((z.round(decimals=2) >= nz_boundaries[:, 0][-2].round(decimals=2)) & (
                    z.round(decimals=2) <= nz_boundaries[:, 2][-2].round(decimals=2)))[0]
        b_1[ids_last] = nuisance_params["_b1_{}".format(nbins)]
        sampler_systematics['b1'] = b_1

    if b2i_marg:
        b_2 = np.ones(len(z))
        for i in range(nbins-1):
            ids = np.where((z.round(decimals=2) >= nz_boundaries[:, 0][i].round(decimals=2)) & (
                        z.round(decimals=2) < nz_boundaries[:, 2][i].round(decimals=2)))[0]
            b_2[ids] = nuisance_params["_b2_{}".format(i+1)]
        ids_last = np.where((z.round(decimals=2) >= nz_boundaries[:, 0][-2].round(decimals=2)) & (
                    z.round(decimals=2) <= nz_boundaries[:, 2][-2].round(decimals=2)))[0]
        b_2[ids_last] = nuisance_params["_b2_{}".format(nbins)]
        sampler_systematics['b2'] = b_2

    if mi_marg:
        mi_dat = []
        for i in range(nbins):
            mi_dat.append(nuisance_params["_m_{}".format(i+1)])
        sampler_systematics['mi'] = np.asarray(mi_dat)

    if A1i_marg:
        A_1 = np.ones(len(z))
        for i in range(nbins-1):
            ids = np.where((z.round(decimals=2) >= nz_boundaries[:, 0][i].round(decimals=2)) & (
                        z.round(decimals=2) < nz_boundaries[:, 2][i].round(decimals=2)))[0]
            A_1[ids] = nuisance_params["_A1_{}".format(i+1)]
        ids_last = np.where((z.round(decimals=2) >= nz_boundaries[:, 0][-2].round(decimals=2)) & (
                    z.round(decimals=2) <= nz_boundaries[:, 2][-2].round(decimals=2)))[0]
        A_1[ids_last] = nuisance_params["_A1_{}".format(nbins)]
        sampler_systematics['A1'] = A_1

    # Now loop over some other constant nuisance parameters that need to be converted to arrays along z
    constant_nuisance_params_to_z = [
        'b1',
        'b2',
        'bs',
        'mi',
        'A1'
    ]

    for count, constant_nuisance_param in enumerate(['_b1', '_b2', '_bs', '_m', '_A1']):
        if constant_nuisance_param in nuisance_params:
            nuisance_param_dat = float(nuisance_params[constant_nuisance_param]) * np.ones(len(z))
            sampler_systematics[constant_nuisance_params_to_z[count]] = nuisance_param_dat

    # Now loop over some other constant nuisance parameters that just need to be kept as constant
    constant_nuisance_params = [
        'A2',
        'bTA',
        'eta1',
        'eta2',
        's0',
        's1',
        's2',
        's3'
    ]

    for count, mag_bias_param in enumerate(['_A2', '_bTA', '_eta1', '_eta2', '_s0', '_s1', '_s2', '_s3']):
        if mag_bias_param in nuisance_params:
            sampler_systematics[constant_nuisance_params[count]] = float(nuisance_params[mag_bias_param])

    if Dzi_marg:
        Dzi_dat = []
        for i in range(nbins):
            Dzi_dat.append(nuisance_params["_Dz_{}".format(i+1)])
        # sampler_systematics['Dzi'] = np.asarray(Dzi_dat)

    else:
        Dzi_dat = None

    systematics_dict = generate_cls.setup_systematics_dict(pipeline_variables_path=pipeline_variables_path)
    systematics_dict.update(sampler_systematics)
    
    cls_dict = generate_cls.setup_6x2pt_cls(
        save_dir=save_dir,
        nz_filename=nz_filename,
        n_zbin=nbins,
        ell_min=input_lmin,
        ell_max=input_lmax,
        fid_cosmo=fid_cosmo,
        systematics_dict=systematics_dict,
        mode='dict',
        Dzi_dat=Dzi_dat)
    
    print(time.time()-start_time)

    # Now we need to convert to Pseudo-Cls and then bandpowers
    theory_cls_dir = save_dir + 'inference_chains/'
    obs_noise_cls_dir = save_dir + 'raw_noise_cls/'

    # create_null_spectras(nbins=nbins, lmin=input_lmin, lmax=input_lmax, output_dir=theory_cls_dir)

    if obs_spec == '6X2PT':
        model_bps = PCl_bandpowers_6x2pt(
            cls_dict=cls_dict,
            n_bp=n_bp,
            n_zbin=nbins,
            lmax_like_galaxy=output_lmax_galaxy,
            lmin_like_galaxy=output_lmin_galaxy,
            lmax_like_galaxy_shear=output_lmax_galaxy_shear,
            lmin_like_galaxy_shear=output_lmin_galaxy_shear,
            lmax_like_shear=output_lmax_shear,
            lmin_like_shear=output_lmin_shear,
            lmax_like_galaxy_kCMB=output_lmax_cmbkk_galaxy,
            lmin_like_galaxy_kCMB=output_lmin_cmbkk_galaxy,
            lmax_like_shear_kCMB=output_lmax_cmbkk_shear,
            lmin_like_shear_kCMB=output_lmin_cmbkk_shear,
            lmax_like_kCMB=output_lmax_cmbkk,
            lmin_like_kCMB=output_lmin_cmbkk,
            lmax_in=input_lmax,
            lmin_in=input_lmin,
            noise_path=obs_noise_cls_dir,
            mixmats=mixmats,
            bandpower_spacing='log')

    elif obs_spec == '3X2PT':
        model_bps = PCl_bandpowers_3x2pt(
            cls_dict=cls_dict,
            n_bp=n_bp,
            n_zbin=nbins,
            lmax_like_nn=output_lmax_galaxy,
            lmin_like_nn=output_lmin_galaxy,
            lmax_like_ne=output_lmax_galaxy_shear,
            lmin_like_ne=output_lmin_galaxy_shear,
            lmax_like_ee=output_lmax_shear,
            lmin_like_ee=output_lmin_shear,
            lmax_in=input_lmax,
            lmin_in=input_lmin,
            noise_path=obs_noise_cls_dir,
            mixmats=mixmats,
            bandpower_spacing='log')

    else:
        assert obs_spec == '1X2PT'

        model_bps = PCl_bandpowers_1x2pt(
            cls_dict=cls_dict,
            n_bp=n_bp,
            n_zbin=nbins,
            lmax_like_galaxy=output_lmax_galaxy,
            lmin_like_galaxy=output_lmin_galaxy,
            lmax_like_galaxy_shear=output_lmax_galaxy_shear,
            lmin_like_galaxy_shear=output_lmin_galaxy_shear,
            lmax_like_shear=output_lmax_shear,
            lmin_like_shear=output_lmin_shear,
            lmax_like_galaxy_kCMB=output_lmax_cmbkk_galaxy,
            lmin_like_galaxy_kCMB=output_lmin_cmbkk_galaxy,
            lmax_like_shear_kCMB=output_lmax_cmbkk_shear,
            lmin_like_shear_kCMB=output_lmin_cmbkk_shear,
            lmax_like_kCMB=output_lmax_cmbkk,
            lmin_like_kCMB=output_lmin_cmbkk,
            lmax_in=input_lmax,
            lmin_in=input_lmin,
            field=obs_field,
            noise_path=obs_noise_cls_dir,
            mixmats=mixmats,
            bandpower_spacing='log')

    return model_bps


def log_normal_likelihood_ccl(params, config_dict, pipeline_variables_path, mixmats, data_vector, inverse_covariance, bi_marg=False, b2i_marg=False, mi_marg=False, Dzi_marg=False, A1i_marg=False):

    """
    Evaluate the likelihood (Gaussian) for a given point in parameter space.

    Parameters
    ----------
    params (dict):                  Parameters (cosmology and nuisance) with their values
    config_dict (dict):             Dictionary describing simulation/measurement parameters. Ones that will be sampled are overwritten
    pipeline_variables_path (str):  Path to location of pipeline variables file (e.g. 'set_variables_3x2pt_measurement.ini')
    mixmats (dict):                 Dictionary of 6x2pt spin-0 and spin-2 mixing matrices
    data_vector (dict):             6x2pt or 3x2pt data vector
    inverse_covariance (2D array):  Inverse covariance matrix
    bi_marg (bool):     Marginalise over b1 galaxy bias (True) or not (False). Prepares formatting for parameter input to CCL
    b2i_marg (bool):    Marginalise over b2 galaxy bias (True) or not (False). Prepares formatting for parameter input to CCL
    mi_marg (bool):     Marginalise over shear m-bias (True) or not (False). Prepares formatting for parameter input to CCL
    Dzi_marg (bool):    Marginalise over Delta z photo-z uncertainty (True) or not (False). Prepares formatting for parameter input to CCL
    A1i_marg (bool):    Marginalise over IA amplitude (True) or not (False). Prepares formatting for parameter input to CCL

    Returns
    -------
    Log likelihood at a given point in parameter space (float)
    """

    model_vector = generate_pseudo_bps_model(cosmo_params=params, pipeline_variables_path=pipeline_variables_path, config_dict=config_dict, mixmats=mixmats, bi_marg=bi_marg, b2i_marg=b2i_marg, mi_marg=mi_marg, Dzi_marg=Dzi_marg, A1i_marg=A1i_marg)

    # Need to stack data vector
    assert model_vector.shape == data_vector.shape
    n_spec, n_bandpower = model_vector.shape

    n_data = n_spec * n_bandpower
    model_vector = np.reshape(model_vector, n_data)
    data_vector = np.reshape(data_vector, n_data)

    d_vector = model_vector - data_vector

    return -0.5 * d_vector @ inverse_covariance @ d_vector


def test(params, config_dict, pipeline_variables_path, mixmats, data_vector):

    """
    Debugging test to investigate data vector preparation for log likelihood calculation

    Parameters
    ----------
    params (dict):  Parameters (cosmology and nuisance) with their values
    config_dict (dict): Dictionary describing simulation/measurement parameters. Ones that will be sampled are overwritten
    pipeline_variables_path (str): Path to location of pipeline variables file (e.g. 'set_variables_3x2pt_measurement.ini')
    mixmats (dict): Dictionary of 6x2pt spin-0 and spin-2 mixing matrices
    data_vector (dict): 6x2pt or 3x2pt data vector

    Returns
    -------
    Prints 3x2pt or 6x2pt data vector
    """

    model_vector = generate_pseudo_bps_model(cosmo_params=params, pipeline_variables_path=pipeline_variables_path, config_dict=config_dict, mixmats=mixmats)
    print(model_vector)
    # Need to stack data vector
    assert model_vector.shape == data_vector.shape
    n_spec, n_bandpower = model_vector.shape
    '''
    n_data = n_spec * n_bandpower
    model_vector = np.reshape(model_vector, n_data)
    data_vector = np.reshape(data_vector, n_data)

    xs = np.arange(1, n_bandpower + 1)

    for i in range(n_spec):
        fig, ax = plt.subplots()
        ax.plot(xs, model_vector[(i*n_bandpower):(i*n_bandpower)+n_bandpower], color='0')
        ax.plot(xs, data_vector[(i*n_bandpower):(i*n_bandpower)+n_bandpower], marker = 'x', linestyle='None')
        plt.show()
    '''


def run_nautilus(sampler_config_dict, pipeline_variables_path, mixmats, data_vector, inverse_covariance,
                 sampler_checkpoint_file, priors, bi_marg=False, b2i_marg=False, mi_marg=False, Dzi_marg=False, A1i_marg=False, run=True):

    """
    Run parameter sampling with nautilus (nested sampling)

    Parameters
    ----------
    sampler_config_dict (dict):     Dictionary describing simulation/measurement parameters. Ones that will be sampled are overwritten
    pipeline_variables_path (str):  Path to location of pipeline variables file (e.g. 'set_variables_3x2pt_measurement.ini')
    mixmats (dict):                 Dictionary of 6x2pt spin-0 and spin-2 mixing matrices
    data_vector (array):            6x2pt or 3x2pt data vector
    inverse_covariance (2D array):  Inverse covariance matrix
    sampler_checkpoint_file (str):  Name of checkpoint file for sampler outputs
    priors (dict):      Parameter and its prior values/type. Prior can either be a tuple for a uniform prior, or e.g. scipy.stats.norm for a Gaussian prior
    bi_marg (bool):     Marginalise over b1 galaxy bias (True) or not (False). Prepares formatting for parameter input to CCL
    b2i_marg (bool):    Marginalise over b2 galaxy bias (True) or not (False). Prepares formatting for parameter input to CCL
    mi_marg (bool):     Marginalise over shear m-bias (True) or not (False). Prepares formatting for parameter input to CCL
    Dzi_marg (bool):    Marginalise over Delta z photo-z uncertainty (True) or not (False). Prepares formatting for parameter input to CCL
    A1i_marg (bool):    Marginalise over IA amplitude (True) or not (False). Prepares formatting for parameter input to CCL
    run (bool):         Run the parameter sampling (True) or not (False). Set run=False to just load the posterior sampling points for plotting etc.

    Returns
    -------
    nautilus sampler object.
    """

    prior = nautilus.Prior()

    n_params = len(priors)

    for p in range(n_params):
        prior.add_parameter(priors[p][0], dist=priors[p][1])

    n_pool = sampler_config_dict['n_pool']

    sampler = nautilus.Sampler(
        prior, log_normal_likelihood_ccl, n_live=200,
        likelihood_kwargs={
            "config_dict": sampler_config_dict,
            "pipeline_variables_path": pipeline_variables_path,
            "mixmats": mixmats,
            "data_vector": data_vector,
            "inverse_covariance": inverse_covariance,
            "bi_marg": bi_marg,
            "b2i_marg": b2i_marg,
            "mi_marg": mi_marg,
            "Dzi_marg": Dzi_marg,
            "A1i_marg": A1i_marg
        },  # could e.g. add bi_marg=True if marginalising over tomographic bin-dependent b parameters
        filepath=sampler_checkpoint_file,
        pool=n_pool
        # pool=MPIPoolExecutor()
    )

    if run:
        sampler.run(verbose=True, discard_exploration=True)

    return sampler


def execute(pipeline_variables_path, covariance_matrix_type, priors, checkpoint_filename, bi_marg=False, b2i_marg=False,  mi_marg=False, Dzi_marg=False, A1i_marg=False):

    """
    Prepare and execute the parameter sampling

    Parameters
    ----------
    pipeline_variables_path (str):  Path to location of pipeline variables file (e.g. 'set_variables_3x2pt_measurement.ini')
    covariance_matrix_type (str):   Either 'analytic' or 'numerical'. Will look for a covariance matrix on disk with relevant filename
    priors (dict):                  Parameter and its prior values/type. Prior can either be a tuple for a uniform prior, or e.g. scipy.stats.norm for a Gaussian prior
    checkpoint_filename (str):      Name of checkpoint file for sampler outputs
    bi_marg (bool):     Marginalise over b1 galaxy bias (True) or not (False). Prepares formatting for parameter input to CCL
    b2i_marg (bool):    Marginalise over b2 galaxy bias (True) or not (False). Prepares formatting for parameter input to CCL
    mi_marg (bool):     Marginalise over shear m-bias (True) or not (False). Prepares formatting for parameter input to CCL
    Dzi_marg (bool):    Marginalise over Delta z photo-z uncertainty (True) or not (False). Prepares formatting for parameter input to CCL
    A1i_marg (bool):    Marginalise over IA amplitude (True) or not (False). Prepares formatting for parameter input to CCL

    Returns
    -------
    Instance of nautilus sampler run
    """

    sampler_config_dict = sampler_config(pipeline_variables_path=pipeline_variables_path)
    # systematics_dict = generate_cls.setup_systematics_dict(pipeline_variables_path=pipeline_variables_path)

    save_dir = sampler_config_dict['save_dir']
    n_bps = sampler_config_dict['n_bandpowers']
    nbins = sampler_config_dict['nbins']
    no_iter = sampler_config_dict['no_iter']
    obs_spec = sampler_config_dict['obs_spec']
    obs_field = sampler_config_dict['obs_field']

    inference_dir = save_dir + 'inference_chains/'
    if not os.path.exists(inference_dir):
        os.makedirs(inference_dir)

    data_vector = np.load(save_dir + 'measured_6x2pt_bps/obs_{}bp.npz'.format(n_bps))['obs_bp']
    sampler_checkpoint_file = inference_dir + checkpoint_filename

    if covariance_matrix_type == 'analytic':
        covariance_matrix = np.load(save_dir + 'analytic_covariance/cov_{}bp.npz'.format(n_bps))['cov']
        inverse_covariance = np.linalg.inv(covariance_matrix)
    else:
        assert covariance_matrix_type == 'numerical'
        covariance_matrix = np.load(save_dir + 'numerical_covariance/cov_{}bp.npz'.format(n_bps))['cov']
        inverse_covariance = np.linalg.inv(covariance_matrix)
        n_sims = no_iter
        if obs_spec == '6X2PT':
            n_dim = ((nbins*((2*nbins)+1)) + (2*nbins) + 1)*n_bps
        elif obs_spec == '3X2PT':
            n_dim = ((nbins*((2*nbins)+1)))*n_bps
        else:
            assert obs_spec == '1X2PT'
            if obs_field == 'E' or obs_field == 'N':
                n_dim = ((nbins*(nbins+1))/2)*n_bps
            elif obs_field == 'EK' or obs_field == 'NK':
                n_dim = nbins*n_bps
            else:
                assert obs_field == 'K'
                n_dim = 1*n_bps
        hartlap_correction = (n_sims-n_dim-2)/(n_sims-1)
        inverse_covariance = inverse_covariance * hartlap_correction

    mix_mats_save_path = save_dir + 'mixmats.npz'
    mixmats_all = np.load(mix_mats_save_path)

    mixmats = {'mixmat_nn_to_nn': mixmats_all['mixmat_nn_to_nn'],
               'mixmat_ne_to_ne': mixmats_all['mixmat_ne_to_ne'],
               'mixmat_ee_to_ee': mixmats_all['mixmat_ee_to_ee'],
               'mixmat_bb_to_ee': mixmats_all['mixmat_bb_to_ee'],
               'mixmat_kk_to_kk': mixmats_all['mixmat_kk_to_kk'],
               'mixmat_nn_to_kk': mixmats_all['mixmat_nn_to_kk'],
               'mixmat_ke_to_ke': mixmats_all['mixmat_ke_to_ke']
               }

    # Uncomment for some useful debugging
    # log_normal_likelihood_ccl({'w0':-1,'wa':0}, sampler_config_dict, pipeline_variables_path, mixmats, data_vector, inverse_covariance)
    # test({'Omega_c':0.3, 'h':0.7, 'w0':-1.0, 'wa':0.0}, sampler_config_dict, pipeline_variables_path, mixmats, data_vector)
    
    sampler_run = run_nautilus(
        sampler_config_dict=sampler_config_dict,
        pipeline_variables_path=pipeline_variables_path,
        mixmats=mixmats,
        data_vector=data_vector,
        inverse_covariance=inverse_covariance,
        sampler_checkpoint_file=sampler_checkpoint_file,
        priors=priors,
        bi_marg=bi_marg,
        b2i_marg=b2i_marg,
        mi_marg=mi_marg,
        Dzi_marg=Dzi_marg,
        A1i_marg=A1i_marg
    )

    return sampler_run