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
from plotting import spider_plot
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

    ParametersSS
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
    cosmo_params: dict
        Dictionary of parameters to overwrite in default CCL cosmology
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
    # Now we have to make the Cls, PCls, bandpowers, then combined data vector
    # cls_dict = {'galaxy_cl': {'ell':ells},
    #             'shear_cl': {'ell':ells},
    #             'galaxy_shear_cl': {'ell':ells},
    #             'cmbkappa_cl': {'ell':ells},
    #             'galaxy_cmbkappa_cl': {'ell':ells},
    #             'shear_cmbkappa_cl': {'ell':ells},
    #             'null_spectra': {'ell':ells}
    #             }

    nz_boundaries = np.loadtxt(f"{save_dir}z_boundaries.txt")

    sampler_systematics = {}

    # if "_b1" in nuisance_params:
    #     # need to convert to b(z)
    #     b_1 = float(nuisance_params["_b1"]) * np.ones(len(z))
    #     sampler_systematics['b1'] = b_1

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

    # if "_b2" in nuisance_params:
    #     # need to convert to b(z)
    #     b_2 = float(nuisance_params["_b2"]) * np.ones(len(z))
    #     sampler_systematics['b2'] = b_2

    # if "_m" in nuisance_params:
    #     # need to convert to b(z)
    #     mi = float(nuisance_params["_m"]) * np.ones(len(z))
    #     sampler_systematics['mi'] = mi

    # mi_marg = nuisance_params['mi_marg']
    if mi_marg:
        mi_dat = []
        for i in range(nbins):
            mi_dat.append(nuisance_params["_m_{}".format(i+1)])
        sampler_systematics['mi'] = np.asarray(mi_dat)

    # if "_A1" in nuisance_params:
    #     # need to convert to b(z)
    #     A_1 = float(nuisance_params["_A1"]) * np.ones(len(z))
    #     sampler_systematics['A1'] = A_1

    # A1i_marg = nuisance_params['A1i_marg']
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
    '''
    b_1 = systematics_dict['b_1']
    b_2 = systematics_dict['b_2']
    b_s = systematics_dict['b_s']
    A_1 = systematics_dict['A_1']
    A_2 = systematics_dict['A_2']
    A_1d = systematics_dict['A_1d']
    c_1 = systematics_dict['c_1']
    c_delta = systematics_dict['c_delta']
    c_2 = systematics_dict['c_2']
    mi_dat = systematics_dict['mi']
    s0 = systematics_dict['s0']
    s1 = systematics_dict['s1']
    s2 = systematics_dict['s2']
    s3 = systematics_dict['s3']
    sz = systematics_dict['sz']

    # These parameters can probably be read in from a config file
    # Galaxy bias
    b = np.ones_like(z)
    # bz = 0.95/ccl.growth_factor(fid_cosmo,1./(1+z))

    # Intrinsic alignment amplitude
    A_IA = nuisance_params['A1'] * np.ones_like(z)
    # print(A_IA)
    # Magnification bias
    sz = np.ones_like(z)

    # CMB lensing
    k_CMB = ccl.CMBLensingTracer(fid_cosmo, z_source=1100)

    cl_kCMB = ccl.angular_cl(fid_cosmo, k_CMB, k_CMB, ells)
    cls_dict['cmbkappa_cl']['bin_1_1'] = cl_kCMB

    for i in range(nbins):

        # Bin i number
        bin_i = i + 1

        # Galaxy clustering bin i
        g_i = ccl.NumberCountsTracer(fid_cosmo, has_rsd=False, dndz=(z, nz_dat[:, bin_i]), bias=(z, b))

        # Cosmic shear with intrinsic alignments bin i
        y_i = ccl.WeakLensingTracer(fid_cosmo, dndz=(z, nz_dat[:, bin_i]), has_shear=True, ia_bias=(z, A_IA))

        # Galaxy clustering - CMB kappa Cl cross-correlation
        cl_g_kCMB = ccl.angular_cl(fid_cosmo, g_i, k_CMB, ells)
        cls_dict['galaxy_cmbkappa_cl']['bin_{}_1'.format(bin_i)] = cl_g_kCMB

        # Cosmic shear - CMB kappa Cl cross-correlation
        cl_y_kCMB = ccl.angular_cl(fid_cosmo, y_i, k_CMB, ells)
        cls_dict['shear_cmbkappa_cl']['bin_{}_1'.format(bin_i)] = cl_y_kCMB

        for j in range(nbins):

            # Bin j number
            bin_j = j+1

            cls_dict['null_spectra']['bin_{}_{}'.format(bin_i, bin_j)] = np.zeros(len(ells))

            # Galaxy clustering bin j
            g_j = ccl.NumberCountsTracer(fid_cosmo, has_rsd=False, dndz=(z, nz_dat[:,bin_j]), bias=(z, b))

            # Cosmic shear with intrinsic alignments bin j
            y_j = ccl.WeakLensingTracer(fid_cosmo, dndz=(z, nz_dat[:,bin_j]), has_shear=True, ia_bias=(z, A_IA))

            # Tomographic galaxy-galaxy lensing Cl
            cl_gy = ccl.angular_cl(fid_cosmo, g_i, y_j, ells)
            cls_dict['galaxy_shear_cl']['bin_{}_{}'.format(bin_i, bin_j)] = cl_gy

            if i >= j:

                # Tomographic angular clustering Cl
                cl_gg = ccl.angular_cl(fid_cosmo, g_i, g_j, ells)
                cls_dict['galaxy_cl']['bin_{}_{}'.format(bin_i, bin_j)] = cl_gg

                # Tomographic cosmic shear Cl
                cl_yy = ccl.angular_cl(fid_cosmo, y_i, y_j, ells)
                cls_dict['shear_cl']['bin_{}_{}'.format(bin_i, bin_j)] = cl_yy
    '''
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
'''
def generate_mixmats(sampler_config_dict):

    save_dir = sampler_config_dict['save_dir']
    inference_dir = save_dir + 'inference_chains/'

    nside = sampler_config_dict['nside']
    input_lmin = sampler_config_dict['input_lmin']
    input_lmax = sampler_config_dict['input_lmax']

    output_lmin_galaxy = sampler_config_dict['output_lmin_galaxy']
    output_lmax_galaxy = sampler_config_dict['output_lmax_galaxy']

    output_lmin_galaxy_shear = sampler_config_dict['output_lmin_galaxy_shear']
    output_lmax_galaxy_shear = sampler_config_dict['output_lmax_galaxy_shear']

    output_lmin_shear = sampler_config_dict['output_lmin_shear']
    output_lmax_shear = sampler_config_dict['output_lmax_shear']

    output_lmin_cmbkk = sampler_config_dict['output_lmin_cmbkk']
    output_lmax_cmbkk = sampler_config_dict['output_lmax_cmbkk']

    output_lmin_cmbkk_galaxy = sampler_config_dict['output_lmin_cmbkk_galaxy']
    output_lmax_cmbkk_galaxy = sampler_config_dict['output_lmax_cmbkk_galaxy']

    output_lmin_cmbkk_shear = sampler_config_dict['output_lmin_cmbkk_shear']
    output_lmax_cmbkk_shear = sampler_config_dict['output_lmax_cmbkk_shear']

    obs_type = sampler_config_dict['obs_spec']
    obs_field = sampler_config_dict['obs_field']

    # Need to create this directory somewhere!
    mask_dir = sampler_config_dict['mask_path']
    mask_dir_cmb = sampler_config_dict['mask_path_cmb']

    mix_mats_save_path = inference_dir + 'mixmats.npz'

    # Calculate mixing matrices from mask
    mask.get_6x2pt_mixmats(mask_path=mask_dir,
                           mask_path_cmb=mask_dir_cmb,
                           nside=nside,
                           lmin=input_lmin,
                           input_lmax=input_lmax,
                           lmax_out_nn=output_lmax_galaxy,
                           lmax_out_ne=output_lmax_galaxy_shear,
                           lmax_out_ee=output_lmax_shear,
                           lmax_out_ek=output_lmax_cmbkk_shear,
                           lmax_out_nk=output_lmax_cmbkk_galaxy,
                           lmax_out_kk=output_lmax_cmbkk,
                           save_path=mix_mats_save_path)
'''

def run_nautilus(sampler_config_dict, pipeline_variables_path, mixmats, data_vector, inverse_covariance,
                 sampler_checkpoint_file, priors, bi_marg=False, b2i_marg=False, mi_marg=False, Dzi_marg=False, A1i_marg=False):

    # systematics_dict = generate_cls.setup_systematics_dict(pipeline_variables_path=pipeline_variables_path)
    # n_zbin = sampler_config_dict['nbins']
    # b_1 = systematics_dict['b_1']
    # mi = systematics_dict['mi']

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

    # sampler.run(verbose=True,discard_exploration=True)

    '''
    The below code is just used for plotting purposes. In order to extract the posterior sampling point, you can just
    use load an instance of nautilus.sampler. But there is probably a better way to do this instead of putting this
    in the run_nautilus function - e.g. make a separate plotting function that loads in the posterior information
    directly from a given file.
    '''

    points, log_w, log_l = sampler.posterior()
    # print(points, log_w, log_l)

    # max_like_id = log_l.argmax()
    # print(max_like_id)
    # w0_m, wa_m, Omega_c_m, h_m,  = points[max_like_id]
    # print(w0_m, wa_m, Omega_c_m, h_m)
    # corner.corner(
    #     points, weights=np.exp(log_w), bins=30, labels=prior.keys
    # )
    # plt.show()

    points = points[:, 0:7]

    # max_like_id = log_l.argmax()
    # print(max_like_id)
    # w0_m, wa_m, Omega_c_m, h_m,  = points[max_like_id]
    # print(w0_m, wa_m, Omega_c_m, h_m)

    one_sigma_1d = 0.683

    q_lower = 1 / 2 - one_sigma_1d / 2
    q_upper = 1 / 2 + one_sigma_1d / 2
    colours = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3']
    colours = ['#1b9e77','#d95f02','#7570b3','#e7298a']
    colours = ['royalblue', 'darkred', '#417B5A']
    # colour_3x2pt_analytic = 'xkcd:faded red'
    colour_3x2pt_analytic = 'darkred'
    colour_3x2pt_numerical = 'xkcd:midnight purple'
    colour_6x2pt_analytic = 'xkcd:soft blue'
    colour_6x2pt_numerical = 'xkcd:very dark green'
    # colour_6x2pt_numerical = 'xkcd:dusty green'
    # colour_6x2pt_numerical = 'xkcd:dull teal'
    # colours = ['xkcd:lightish blue', 'darkred', 'xkcd:midnight purple','xkcd:midnight purple']
    # colours = ['xkcd:lightish blue', '#FF7070', 'xkcd:midnight purple','xkcd:midnight purple']
    # hist_bins = [90,90,120,80,90,150,250,500,500,500,500,500,500]
    # hist_bins = [60, 40, 100, 30, 40, 100, 100]
    hist_bins = [20, 20, 100, 30, 40, 100, 100]
    # hist_bins = [100,100,100]
    # hist_bins = [60,40,100,30,40,100,100,3000,100]
    # hist_bins = [60,100,30,40,100,100,3000,100]
    # hist_bins = [60,40,100,30,40,100,100,1500, 1500, 500, 500, 500]
    # hist_bins = [50,30,100,30,40,100,100,1500, 1500, 500, 500, 500]
    # hist_bins = [50,30,100,30,40,100,250] #,1000,1000,200]
    figure = corner.corner(
        points,
        weights=np.exp(log_w),
        # bins=250,
        # bins=[3000,2000,300],
        # bins=[50,30,100,30,40,100,100,1500, 1500, 500, 500, 500],
        # bins=[70,100,200,70,70,200,200],
        bins=hist_bins,  # ,5000,300],
        # bins=[50,30,100,30,40,100,250,1000,1000,200],
        # bins=[50,30,100,30,40,100,100, 300, 300, 100],
        # labels=prior.keys[0:],
        plot_density=False,
        fill_contours=True,
        # color='royalblue',
        color=colour_6x2pt_analytic,
        data_kwargs={'color': '0.45', 'ms': '0'},
        label_kwargs={'fontsize': '20'},
        hist_kwargs={'linewidth':1.75},
        labels=[r'$w_{0}$', r'$w_{a}$', r'$\Omega_{m}$', r'$h$', r'$\Omega_{b}$', r'$n_{s}$', r'$\sigma_{8}$', r'$A_{1}$', r'$A_{2}$', r'$b_{TA}$', r'$\eta_{1}$', r'$\eta_{2}$'],
        # labels=[r'$w_{0}$', r'$w_{a}$', r'$\Omega_{m}$', r'$h$', r'$\Omega_{b}$', r'$n_{s}$', r'$\sigma_{8}$', r'$A_{1}$', r'$\eta_{1}$'],
        # labels=[r'$w_{0}$', r'$\Omega_{m}$', r'$h$', r'$\Omega_{b}$', r'$n_{s}$', r'$\sigma_{8}$', r'$A_{1}$', r'$\eta_{1}$'],
        # labels=[r'$w_{0}$', r'$w_{a}$', r'$\Omega_{m}$', r'$h$', r'$\Omega_{b}$', r'$n_{s}$', r'$\sigma_{8}$'],
        # labels=[r'$w_{0}$', r'$\Omega_{m}$', r'$\sigma_{8}$'],
        # labels=[r'$w_{0}$', r'$w_{a}$', r'$\Omega_{m}$', r'$h$', r'$\Omega_{b}$', r'$n_{s}$', r'$\sigma_{8}$', r'$b_{1}$', r'$b_{2}$', r'$b_{s}$'],
        # labels=[r'$w_{0}$', r'$w_{a}$', r'$\Omega_{m}$', r'$h$', r'$\Omega_{b}$', r'$n_{s}$', r'$\sigma_{8}$', r'$b_{1}$', r'$b_{2}$', r'$b_{3}$', r'$b_{4}$', r'$b_{5}$', r'$b_{6}$'],
        labelpad=0.025,
        levels=(0.683, 0.955),
        smooth=1.5,
        smooth1d=True,
        # title_quantiles=[q_lower, 0.5, q_upper],
        # show_titles=True,
        # title_fmt='.4f'
    )

    prior2 = nautilus.Prior()
    # # # prior2.add_parameter("w0", dist=(-1.05, -0.95))
    # # # prior2.add_parameter("wa", dist=(-0.25, 0.25))
    # #
    prior2.add_parameter("w0", (-1.25, -0.75))
    prior2.add_parameter("wa", (-0.5, 0.5))
    prior2.add_parameter("Omega_m", (0.2, 0.4))
    prior2.add_parameter("h", (0.5, 0.8))
    prior2.add_parameter("Omega_b", (0.02, 0.08))
    prior2.add_parameter("n_s", (0.8, 1.2))
    prior2.add_parameter("sigma8", (0.75, 0.9))

    # prior2.add_parameter("_A1", (-8, 8))
    # prior2.add_parameter("_A2", (-8, 8))
    # prior2.add_parameter("_bTA", (-6, 6))
    # prior2.add_parameter("_eta1", (-6, 6))
    # prior2.add_parameter("_eta2", (-6, 6))

    # prior2.add_parameter("_Dz_1", norm(loc=0, scale=0.01))
    # prior2.add_parameter("_Dz_2", norm(loc=0, scale=0.01))
    # prior2.add_parameter("_Dz_3", norm(loc=0, scale=0.01))
    # prior2.add_parameter("_Dz_4", norm(loc=0, scale=0.01))
    # prior2.add_parameter("_Dz_5", norm(loc=0, scale=0.01))
    # prior2.add_parameter("_Dz_6", norm(loc=0, scale=0.01))

    # prior2 = prior
    sampler2 = nautilus.Sampler(
        # prior, log_normal_likelihood_ccl, n_live=200,
        prior2, log_normal_likelihood_ccl, n_live=200,
        likelihood_kwargs={
            "config_dict": sampler_config_dict,
            "pipeline_variables_path": pipeline_variables_path,
            "mixmats": mixmats,
            "data_vector": data_vector,
            "inverse_covariance": inverse_covariance,
            "bi_marg": bi_marg,
            "mi_marg": mi_marg,
            "Dzi_marg": Dzi_marg,
            "A1i_marg": A1i_marg
        },  # could e.g. add bi_marg=True if marginalising over tomographic bin-dependent b parameters
        filepath='/raid/scratch/wongj/mywork/3x2pt/6x2pt_sim_data/b/inference_chains/Cosmology_3x2pt_analytic.hdf5',
        pool=n_pool
    )

    # sampler2.run(verbose=True)

    points2, log_w2, log_l2 = sampler2.posterior()
    points2 = points2[:,0:7]

    corner.corner(
        points2,
        weights=np.exp(log_w2),
        bins=hist_bins,
        plot_density=False,
        # fill_contours=True,
        no_fill_contours=True,
        color=colour_3x2pt_analytic,
        # color='xkcd:very dark green',
        # contour_kwargs={'linestyles':'--','linewidths':2.5,},
        contour_kwargs={'linestyles':'-'},
        data_kwargs={'color': '0.45', 'ms': '0'},
        label_kwargs={'fontsize': '20'},
        labelpad=0.025,
        levels=(0.683, 0.955),
        smooth=1.5,
        smooth1d=True,
        fig=figure,
        # hist_kwargs={'linestyle':'--','linewidth':2.5,'dashes':(8, 6)},
        hist_kwargs={'linestyle':'-','linewidth':1.75},
        hist2d_kwargs={'contour_kwargs':{'linestyles': '--'}}
    )

    # prior3 = prior

    # prior3 = nautilus.Prior()
    # prior3.add_parameter("w0", (-1.25, -0.75))
    # prior3.add_parameter("wa", (-0.5, 0.5))
    # prior3.add_parameter("Omega_m", (0.2, 0.4))
    # prior3.add_parameter("h", (0.5, 0.8))
    # prior3.add_parameter("Omega_b", (0.02, 0.08))
    # prior3.add_parameter("n_s", (0.8, 1.2))
    # prior3.add_parameter("sigma8", (0.75, 0.9))
    #
    # prior3.add_parameter("_A1", (-8, 8))
    # # prior3.add_parameter("_A2", (-8, 8))
    # # prior3.add_parameter("_bTA", (-6, 6))
    # prior3.add_parameter("_eta1", (-6, 6))
    # # prior3.add_parameter("_eta2", (-6, 6))
    #
    # prior3.add_parameter("_Dz_1", norm(loc=0, scale=0.01))
    # prior3.add_parameter("_Dz_2", norm(loc=0, scale=0.01))
    # prior3.add_parameter("_Dz_3", norm(loc=0, scale=0.01))
    # # prior3.add_parameter("_Dz_4", norm(loc=0, scale=0.01))
    # # prior3.add_parameter("_Dz_5", norm(loc=0, scale=0.01))
    # # prior3.add_parameter("_Dz_6", norm(loc=0, scale=0.01))
    #
    # sampler3 = nautilus.Sampler(
    #     # prior, log_normal_likelihood_ccl, n_live=200,
    #     prior3, log_normal_likelihood_ccl, n_live=200,
    #     likelihood_kwargs={
    #         "config_dict": sampler_config_dict,
    #         "pipeline_variables_path": pipeline_variables_path,
    #         "mixmats": mixmats,
    #         "data_vector": data_vector,
    #         "inverse_covariance": inverse_covariance,
    #         "bi_marg": bi_marg,
    #         "mi_marg": mi_marg,
    #         "Dzi_marg": Dzi_marg,
    #         "A1i_marg": A1i_marg
    #     },  # could e.g. add bi_marg=True if marginalising over tomographic bin-dependent b parameters
    #     filepath='/raid/scratch/wongj/mywork/3x2pt/6x2pt_sim_data/b/inference_chains/Cosmology_3bin_6x2pt_NLAz_Photz.hdf5',
    #     pool=n_pool
    # )
    #
    # # sampler3.run(verbose=True)
    #
    # points3, log_w3, log_l3 = sampler3.posterior()
    # points3 = points3[:, 0:7]
    #
    # corner.corner(
    #     points3,
    #     weights=np.exp(log_w3),
    #     bins=hist_bins,
    #     plot_density=False,
    #     # no_fill_contours=False,
    #     fill_contours=True,
    #     color='xkcd:faded red',
    #     # contour_kwargs={'linestyles': '--', 'linewidths': 1.5},
    #     data_kwargs={'color': '0.45', 'ms': '0'},
    #     label_kwargs={'fontsize': '20'},
    #     labelpad=0.025,
    #     levels=(0.683, 0.955),
    #     smooth=1.5,
    #     smooth1d=True,
    #     fig=figure,
    #     hist_kwargs={'linestyle': '-', 'linewidth': 1.75},
    #     # hist2d_kwargs={'contour_kwargs': {'linestyles': '-'}}
    # )
    #
    # prior4 = prior3
    # sampler4 = nautilus.Sampler(
    #     # prior, log_normal_likelihood_ccl, n_live=200,
    #     prior4, log_normal_likelihood_ccl, n_live=200,
    #     likelihood_kwargs={
    #         "config_dict": sampler_config_dict,
    #         "pipeline_variables_path": pipeline_variables_path,
    #         "mixmats": mixmats,
    #         "data_vector": data_vector,
    #         "inverse_covariance": inverse_covariance,
    #         "bi_marg": bi_marg,
    #         "mi_marg": mi_marg,
    #         "Dzi_marg": Dzi_marg,
    #         "A1i_marg": A1i_marg
    #     },  # could e.g. add bi_marg=True if marginalising over tomographic bin-dependent b parameters
    #     filepath='/raid/scratch/wongj/mywork/3x2pt/6x2pt_sim_data/b/inference_chains/Cosmology_3bin_3x2pt_NLAz_Photz.hdf5',
    #     pool=n_pool
    # )
    #
    # # sampler4.run(verbose=True)
    #
    # points4, log_w4, log_l4 = sampler4.posterior()
    # points4 = points4[:, 0:7]
    #
    # corner.corner(
    #     points4,
    #     weights=np.exp(log_w4),
    #     bins=hist_bins,
    #     plot_density=False,
    #     no_fill_contours=True,
    #     color='xkcd:midnight purple',
    #     contour_kwargs={'linestyles': ':', 'linewidths': 2.5},
    #     data_kwargs={'color': '0.45', 'ms': '0'},
    #     label_kwargs={'fontsize': '20'},
    #     labelpad=0.025,
    #     levels=(0.683, 0.955),
    #     smooth=1.5,
    #     smooth1d=True,
    #     fig=figure,
    #     # hist_kwargs={'linestyle': ':', 'linewidth': 3.},
    #     hist_kwargs={'linestyle': ':', 'linewidth': 3.,'dashes': (1, 1.75)},
    #     hist2d_kwargs={'contour_kwargs': {'linestyles': '-'}}
    # )

    ndim = len(prior.keys[0:7])

    xranges = []

    # for i in range(ndim):
    #     q_lo, q_mid, q_hi = corner.quantile(
    #         points[:,i], [q_lower, 0.5, q_upper], weights=np.exp(log_w)
    #     )
    #     q_m, q_p = q_mid - q_lo, q_hi - q_mid
    #
    #     print(np.array([q_mid, q_p, q_m]))
    #
    # for i in range(ndim):
    #     q_lo, q_mid, q_hi = corner.quantile(
    #         points2[:,i], [q_lower, 0.5, q_upper], weights=np.exp(log_w2)
    #     )
    #     q_m, q_p = q_mid - q_lo, q_hi - q_mid
    #
    #     print(np.array([q_mid, q_p, q_m]))
    #
    # for i in range(ndim):
    #     q_lo, q_mid, q_hi = corner.quantile(
    #         points3[:,i], [q_lower, 0.5, q_upper], weights=np.exp(log_w3)
    #     )
    #     q_m, q_p = q_mid - q_lo, q_hi - q_mid
    #
    #     print(np.array([q_mid, q_p, q_m]))
    #
    # for i in range(ndim):
    #     q_lo, q_mid, q_hi = corner.quantile(
    #         points4[:,i], [q_lower, 0.5, q_upper], weights=np.exp(log_w4)
    #     )
    #     q_m, q_p = q_mid - q_lo, q_hi - q_mid
    #
    #     print(np.array([q_mid, q_p, q_m]))

    one_sigma_1d = 0.999999999919680
    # one_sigma_1d = 0.99999999999999999
    # one_sigma_1d = 0.99

    q_lower = 1 / 2 - one_sigma_1d / 2
    q_upper = 1 / 2 + one_sigma_1d / 2

    for i in range(ndim):
        # print(len(points[:,i]))
        # print(len(np.exp(log_w)))
        q_lo, q_mid, q_hi = corner.quantile(
            points[:, i], [q_lower, 0.5, q_upper], weights=np.exp(log_w)
        )
        q_m, q_p = q_mid - q_lo, q_hi - q_mid
        xranges.append([q_lo, q_hi])
    # print(xranges)
    #
    # # Format the quantile display.
    # fmt = "{{0:{0}}}".format(title_fmt).format
    # title = r"${{{0}}}_{{-{1}}}^{{+{2}}}$"
    # title = title.format(fmt(q_mid), fmt(q_m), fmt(q_p))

    for ax in figure.get_axes():
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')
        ax.tick_params(axis='both', direction='in', labelsize=20)

    axes = np.array(figure.axes).reshape((ndim, ndim))
    print(xranges)
    # xranges = [priors[p][1] for p in range(len(priors))]
    # xranges = [
    #     [-1.0974883388998782, -0.9025670345168216],
    #     [-0.324571436758127, 0.3238736406626768],
    #     [0.29316393395984386, 0.34167398549331107],
    #     [0.5797220782621462, 0.7964894292448723],
    #     [0.021619364573003622, 0.06535712696165026],
    #     [0.9191568740510337, 0.9890296592069205],
    #     [0.829838895659455, 0.850700113144348]]
    # xranges = [[-1.216916504630569, -0.7890377104557618], [-0.49981806656290967, 0.4974932190964215], [0.28177182970458586, 0.3490016113217602], [0.5636557941320344, 0.796340818593143], [0.02438191988410677, 0.06797016104624742], [0.9152202196960018, 0.9970257747513469], [0.8129275002336059, 0.8702363410286397]]
    # xranges = [[-1.1418536913299713, -0.870641853469964], [-0.49318537181348365, 0.4989471024174993], [0.28420764357146755, 0.3434156498474476], [0.5226405520836365, 0.7999782316526677], [0.020137555386880304, 0.07484035086864482], [0.904764361505146, 1.025924395826439], [0.8273672668107686, 0.8553300489872568]]
    # xranges = [[-1.1920864137235616, -0.8315688372813771], [-0.4999941901086503, 0.4998923961617449], [0.2824880935050607, 0.34834707550562577], [0.5093574602379654, 0.7999991657669437], [0.020001949882868814, 0.07227840900793263], [0.8899127875188986, 1.0215697092801574], [0.8208971166689492, 0.8613794221103482]]

    xranges = [[-1.1364594543123059, -0.8523614448749834], [-0.4963665738858712, 0.4964892057529142], [0.2860484225541756, 0.3432946752922872], [0.5210528238876092, 0.7999342569642058], [0.020006745320333977, 0.07078393509341487], [0.9022951564615705, 1.0184208118442217], [0.8255727669130657, 0.8551206326385061]]

    # xranges = [
    #     [-1.25, -0.75],
    #     [-0.5, 0.5],
    #     [0.27, 0.39],
    #     [0.5, 0.8],
    #     [0.02, 0.075],
    #     [0.88, 1.05],
    #     [0.8, 0.875]]

    # xranges = [[-1.1082122406038604, -0.8588045186598293], [-0.39754263231340053, 0.3371413415903779], [0.2900175517374249, 0.3356140987085587], [0.5713208498678132, 0.7987513280949396], [0.02637529907399582, 0.06713496640576874], [0.9234352733703783, 0.9935215656362673], [0.825117825745075, 0.8521469754037889]]

    yranges = xranges[1:]

    # fid_vals = [-1, 0]
    # fid_vals = [-1,0,0.315,0.67,0.045,0.96,0.84048,0.7,-1.7] # for IA
    # fid_vals = [-1,0.315,0.67,0.045,0.96,0.84048,0.7,-1.7] # for IA
    # fid_vals = [-1,0,0.315,0.67,0.045,0.96,0.84048] # for cosmo
    # fid_vals = [-1,0.315,0.84048] # for cosmo
    # fid_vals = [-1,0,0.315,0.67,0.045,0.96,0.84048, 2.07, 0.5, -0.611] # for nl bias
    # fid_vals = [-1, 0, 0.315, 0.67, 0.045, 0.96, 0.84048, 2.07, 2.07, 2.07, 2.07, 2.07, 2.07]  # for bias
    fid_vals = [-1,0,0.315,0.67,0.045,0.96,0.84048,0.7,-1.36,1.0,-1.7,-2.5] # for IA
    # fid_vals = [-1,0,0.315,0.67,0.045,0.96,0.84048,2.07,0.5,-0.611] # for IA
    # fid_vals = [2.07, 0.5, -0.611]
    # fid_vals = [-1,0,0.315,0.67,0.045,0.96,0.84048,2.07,0.5,-0.611]

    for i in range(ndim):
        ax = axes[i, i]
        ax.set_xlim(xranges[i][0], xranges[i][1])
        ax.axvline(fid_vals[i], color='0.5', linestyle=':')

    for yi in range(ndim):
        for xi in range(yi):
            ax = axes[yi, xi]
            ax.set_xlim(xranges[xi][0], xranges[xi][1])
            ax.set_ylim(yranges[yi - 1][0], yranges[yi - 1][1])
            ax.axvline(fid_vals[xi], color='0.5', linestyle=':')
            ax.axhline(fid_vals[yi], color='0.5', linestyle=':')

    # patch1 = mpatches.Patch(color='darkred', label='3'+r'$\times$'+'2pt')
    # patch2 = mpatches.Patch(color='royalblue', label='Stage IV-like 3x2pt + \nSO-like CMB lensing\n(6x2pt)')
    # patch1 = mpatches.Patch(color='darkred', label='Stage IV-like 3x2pt')
    patch1 = mpatches.Patch(color=colour_6x2pt_analytic, label='Stage IV-like 6x2pt (6 Bin)\nTATT')
    patch2 = mpatches.Patch(color=colour_3x2pt_analytic, label='Stage IV-like 6x2pt (6 Bin)\nNLA-'+r'$z$')
    # patch3 = mpatches.Patch(color=colour_6x2pt_analytic, label='Stage IV-like 6x2pt (3 Bin)\n' + r'$w_{0}w_{a}$'+'CDM + TATT + ' + r'$\Delta z_{i}$' )
    # patch4 = mpatches.Patch(color='xkcd:faded red', label='Stage IV-like 6x2pt (3 Bin)\n' + r'$w_{0}w_{a}$'+'CDM + NLA-'+r'$z$'+ ' + ' + r'$\Delta z_{i}$')
    # line1 = Line2D([0], [0], color='xkcd:very dark green', lw=4, ls='--', label='Stage IV-like 3x2pt (3 Bin)\n' + r'$w_{0}w_{a}$'+'CDM + TATT + '+ r'$\Delta z_{i}$')
    # line2 = Line2D([0], [0], color='xkcd:midnight purple', lw=4, ls=':', label='Stage IV-like 3x2pt (3 Bin)\n' + r'$w_{0}w_{a}$'+'CDM + NLA-'+r'$z$' + ' + ' + r'$\Delta z_{i}$')
    figure.legend(handles=[patch1, patch2], loc='center right', fontsize=20)  # legend can be centre right for big plots
    # figure.legend(handles=[patch2, patch1], loc='center right', fontsize=20)  # legend can be centre right for big plots
    save_fig_dir = '/raid/scratch/wongj/mywork/3x2pt/6x2pt_sim_data/b/inference_chains/P4.png'

    # if os.path.exists(save_fig_dir):
    #     print('WARNING! File exists, did not overwrite')
    # else:
    #     plt.savefig(save_fig_dir,dpi=200)

    plt.show()

    spider_plot.spiderplot(spider_plot.spider_data4(sampler1=sampler2, sampler2=sampler, sampler3=sampler2, sampler4=sampler))



def execute(pipeline_variables_path, covariance_matrix_type, priors, checkpoint_filename, bi_marg=False, b2i_marg=False,  mi_marg=False, Dzi_marg=False, A1i_marg=False):

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
        print(hartlap_correction)
        inverse_covariance = inverse_covariance * hartlap_correction
    # covariance_matrix = np.load(save_dir + 'cov_fromsim/cov_{}bp.npz'.format(n_bps))['cov']
    # covariance_matrix = np.load(covariance_matrix_path)['cov']

    #inverse_covariance = np.linalg.inv(covariance_matrix)
    #inverse_covariance = inverse_covariance*((1500-910-2)/(1500-1))
    # generate_mixmats(sampler_config_dict=sampler_config_dict)

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

    #log_normal_likelihood_ccl({'w0':-1,'wa':0}, sampler_config_dict, pipeline_variables_path, mixmats, data_vector, inverse_covariance)

    #test({'Omega_c':0.333217311871365, 'h':0.7844042785622749, 'w0':-1.0792991827817433, 'wa':0.0422195196140166}, sampler_config_dict, pipeline_variables_path, mixmats, data_vector)
    
    run_nautilus(
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
    
