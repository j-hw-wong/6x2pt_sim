import os
import sys
import time
import corner
import nautilus
import configparser
import numpy as np
import pyccl as ccl
import pymaster as nmt
import likelihood.mask as mask
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from likelihood.model_pcl import PCl_bandpowers_1x2pt, PCl_bandpowers_3x2pt, PCl_bandpowers_6x2pt
from catalogue_sim import generate_cls
from plotting import spider_plot
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['mathtext.fontset'] = 'cm'
# plt.rcParams['font.serif'] = 'cm'

def parse_nautilus_params_for_ccl(params):
    """
    nautilus propogates params as a dictionary, with each value an unshaped ndarray
    """

    return {key:val[()] for key, val in params.items()}


def split_ccl_parameters(params, n_zbin, bi_marg=False, mi_marg=False, Dzi_marg=False, A1i_marg=False):
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
    # A_s = float(config['cosmology']['A_s'])
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
        'bp_bins_shear': bp_bins_shear,
        'ell_arr_shear': ell_arr_shear,
        'output_lmin_galaxy_shear': output_lmin_galaxy_shear,
        'output_lmax_galaxy_shear': output_lmax_galaxy_shear,
        'pbl_galaxy_shear': pbl_galaxy_shear,
        'bp_bins_galaxy_shear': bp_bins_galaxy_shear,
        'ell_arr_galaxy_shear': ell_arr_galaxy_shear,
        'output_lmin_galaxy': output_lmin_galaxy,
        'output_lmax_galaxy': output_lmax_galaxy,
        'pbl_galaxy': pbl_galaxy,
        'bp_bins_galaxy': bp_bins_galaxy,
        'ell_arr_galaxy': ell_arr_galaxy,
        'output_lmin_cmbkk': output_lmin_cmbkk,
        'output_lmax_cmbkk': output_lmax_cmbkk,
        'pbl_cmbkk': pbl_cmbkk,
        'bp_bins_cmbkk': bp_bins_cmbkk,
        'ell_arr_cmbkk': ell_arr_cmbkk,
        'output_lmin_cmbkk_galaxy': output_lmin_cmbkk_galaxy,
        'output_lmax_cmbkk_galaxy': output_lmax_cmbkk_galaxy,
        'pbl_cmbkk_galaxy': pbl_cmbkk_galaxy,
        'bp_bins_cmbkk_galaxy': bp_bins_cmbkk_galaxy,
        'ell_arr_cmbkk_galaxy': ell_arr_cmbkk_galaxy,
        'output_lmin_cmbkk_shear': output_lmin_cmbkk_shear,
        'output_lmax_cmbkk_shear': output_lmax_cmbkk_shear,
        'pbl_cmbkk_shear': pbl_cmbkk_shear,
        'bp_bins_cmbkk_shear': bp_bins_cmbkk_shear,
        'ell_arr_cmbkk_shear': ell_arr_cmbkk_shear,
        'obs_spec': obs_spec,
        'obs_field': obs_field,
        'Omega_c': Omega_c,
        'Omega_b': Omega_b,
        'h': h,
        'n_s': n_s,
        # 'A_s': A_s,
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


def generate_pseudo_bps_model(cosmo_params, pipeline_variables_path, config_dict, mixmats, bi_marg=False, mi_marg=False, Dzi_marg=False, A1i_marg=False):
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
    # A_s = config_dict['A_s']
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
        # 'A_s': A_s,
        'sigma8': sigma8,
        'n_s': n_s,
        'Omega_k': Omega_k,
        'w0': w0,
        'wa': wa,
        'extra_parameters': {'camb':{'dark_energy_model':'ppf'}}
    }
    # print(fid_cosmo_dict)
    ccl_params, nuisance_params = split_ccl_parameters(cosmo_params, n_zbin=nbins, bi_marg=bi_marg, mi_marg=mi_marg, Dzi_marg=Dzi_marg, A1i_marg=A1i_marg)

    if 'Omega_m' in nuisance_params:
        fid_cosmo_dict['Omega_c'] = nuisance_params['Omega_m'] - Omega_b
    fid_cosmo_dict.update(ccl_params)
    # print(fid_cosmo_dict)
    fid_cosmo = ccl.Cosmology(**fid_cosmo_dict)

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

    if "_b1" in nuisance_params:
        # need to convert to b(z)
        b_1 = float(nuisance_params["_b1"]) * np.ones(len(z))
        sampler_systematics['b1'] = b_1

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

    if "_m" in nuisance_params:
        # need to convert to b(z)
        mi = float(nuisance_params["_m"]) * np.ones(len(z))
        sampler_systematics['mi'] = mi

    # mi_marg = nuisance_params['mi_marg']
    if mi_marg:
        mi_dat = []
        for i in range(nbins):
            mi_dat.append(nuisance_params["_m_{}".format(i+1)])
        sampler_systematics['mi'] = np.asarray(mi_dat)

    if "_A1" in nuisance_params:
        # need to convert to b(z)
        A_1 = float(nuisance_params["_A1"]) * np.ones(len(z))
        sampler_systematics['A1'] = A_1

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
        'b2',
        'bs',
    ]

    for count, constant_nuisance_param in enumerate(['_b2','_bs']):
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


def log_normal_likelihood_ccl(params, config_dict, pipeline_variables_path, mixmats, data_vector, inverse_covariance, bi_marg=False, mi_marg=False, Dzi_marg=False, A1i_marg=False):

    model_vector = generate_pseudo_bps_model(cosmo_params=params, pipeline_variables_path=pipeline_variables_path, config_dict=config_dict, mixmats=mixmats, bi_marg=bi_marg, mi_marg=mi_marg, Dzi_marg=Dzi_marg, A1i_marg=A1i_marg)
    print(model_vector.shape, data_vector.shape)
    # Need to stack data vector
    assert model_vector.shape == data_vector.shape
    n_spec, n_bandpower = model_vector.shape

    n_data = n_spec * n_bandpower
    model_vector = np.reshape(model_vector, n_data)
    data_vector = np.reshape(data_vector, n_data)
    # inverse_covariance = np.diag(np.ones(n_data))

    d_vector = model_vector - data_vector
    l = -0.5 * d_vector @ inverse_covariance @ d_vector
    print(l)
    return -0.5 * d_vector @ inverse_covariance @ d_vector


def test(params, config_dict, pipeline_variables_path, mixmats, data_vector, inverse_covariance):

    print('Starting likelihood call')
    test_start_time = time.time()
    model_vector = generate_pseudo_bps_model(cosmo_params=params, pipeline_variables_path=pipeline_variables_path, config_dict=config_dict, mixmats=mixmats, Dzi_marg=True)

    # Need to stack data vector
    assert model_vector.shape == data_vector.shape
    n_spec, n_bandpower = model_vector.shape

    n_data = n_spec * n_bandpower
    model_vector = np.reshape(model_vector, n_data)
    data_vector = np.reshape(data_vector, n_data)

    d_vector = model_vector - data_vector

    l = -0.5 * d_vector @ inverse_covariance @ d_vector
    print(time.time()-test_start_time)
    xs = np.arange(1, n_bandpower + 1)

    for i in range(n_spec):
        fig, ax = plt.subplots()
        ax.plot(xs, model_vector[(i*n_bandpower):(i*n_bandpower)+n_bandpower], color='0')
        ax.plot(xs, data_vector[(i*n_bandpower):(i*n_bandpower)+n_bandpower], marker = 'x', linestyle='None')
        plt.show()

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
                 sampler_checkpoint_file, priors, bi_marg=False, mi_marg=False, Dzi_marg=False, A1i_marg=False):

    # systematics_dict = generate_cls.setup_systematics_dict(pipeline_variables_path=pipeline_variables_path)
    # n_zbin = sampler_config_dict['nbins']
    # b_1 = systematics_dict['b_1']
    # mi = systematics_dict['mi']

    prior = nautilus.Prior()

    n_params = len(priors)

    for p in range(n_params):
        prior.add_parameter(priors[p][0], dist=priors[p][1])

    # prior = nautilus.Prior()
    # prior.add_parameter("w0", dist=(-1.5, -0.5))
    # prior.add_parameter("wa", dist=(-0.5, 0.5))
    # prior.add_parameter("Omega_m", dist=(0.2, 0.4))
    # prior.add_parameter("h", dist=(0.5, 0.8))

    # For a constant global galaxy bias, we can have e.g.
    # prior.add_parameter('_b1', dist=(0,3))   # for a constant global galaxy bias

    # Or we specify b1 as a constant that is different for each bin, e.g for a 3 bin analysis.
    # prior.add_parameter('_b1_1', dist=(0,3))   # for a constant galaxy bias in bin 1
    # prior.add_parameter('_b1_2', dist=(0,3))   # for a constant galaxy bias in bin 2
    # prior.add_parameter('_b1_3', dist=(0,3))   # for a constant galaxy bias in bin 3
    # and in this case we need to have bi_marg=True in the sampler args below

    # Can also repeat this for m-bias marginalisation. If a global m-bias (independent of tomographic bin)
    # then we have to do
    # prior.add_parameter('_m', dist=(0,3))   # for a constant global m-bias
    # Otherwise, we set the m-bias per tomographic bin, i.e
    # prior.add_parameter('_m_1', dist=(0,3))   # for a constant m-bias in bin 1
    # prior.add_parameter('_m_2', dist=(0,3))   # for a constant m-bias in bin 2
    # prior.add_parameter('_m_3', dist=(0,3))   # for a constant m-bias in bin 3
    # and in this case we need to set mi_marg=True in the sampler args below

    # Can also repeat this for the A1 amplitude of IA TATT/NLA model. If a global A1 value (independent of tomographic
    # bin) then we have to do
    # prior.add_parameter('_A1', dist=(0,3))   # for a constant global m-bias
    # Otherwise, we set the A1 value per tomographic bin, i.e
    # prior.add_parameter('_A1_1', dist=(0,3))   # for a constant A1 in bin 1
    # prior.add_parameter('_A1_2', dist=(0,3))   # for a constant A1 in bin 2
    # prior.add_parameter('_A1_3', dist=(0,3))   # for a constant A1 in bin 3
    # and in this case we need to set A1i_marg=True in the sampler args below

    # For marginalisation over shift paramaters for the n(z) we have to add a shift parameter per bin, i.e
    # prior.add_parameter('_Dz_1', dist=(0,3))   # for a constant m-bias in bin 1
    # prior.add_parameter('_Dz_2', dist=(0,3))   # for a constant m-bias in bin 2
    # prior.add_parameter('_Dz_3', dist=(0,3))   # for a constant m-bias in bin 3
    # and we need to set Dzi_marg=True in the sampler args below

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
            "mi_marg": mi_marg,
            "Dzi_marg": Dzi_marg,
            "A1i_marg": A1i_marg
        },  # could e.g. add bi_marg=True if marginalising over tomographic bin-dependent b parameters
        filepath=sampler_checkpoint_file,
        pool=n_pool
    )

    sampler.run(verbose=True)

    points, log_w, log_l = sampler.posterior()

    # points = points[:,0:]
    points = points[:,0:7]

    # max_like_id = log_l.argmax()
    # print(max_like_id)
    # w0_m, wa_m, Omega_c_m, h_m,  = points[max_like_id]
    # print(w0_m, wa_m, Omega_c_m, h_m)

    one_sigma_1d = 0.683

    q_lower = 1 / 2 - one_sigma_1d / 2
    q_upper = 1 / 2 + one_sigma_1d / 2

    # hist_bins = [90,90,120,80,90,150,250,500,500,500,500,500,500]
    hist_bins = [60,40,100,30,40,100,100]
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
        bins=hist_bins, #,5000,300],
        # bins=[50,30,100,30,40,100,250,1000,1000,200],
        # bins=[50,30,100,30,40,100,100, 300, 300, 100],
        # labels=prior.keys[0:],
        plot_density=False,
        fill_contours=True,
        color='royalblue',
        data_kwargs={'color':'0.45','ms':'0'},
        label_kwargs={'fontsize':'20'},
        # hist_kwargs={'linewidth':0},
        # labels=[r'$w_{0}$', r'$w_{a}$', r'$\Omega_{m}$', r'$h$', r'$\Omega_{b}$', r'$n_{s}$', r'$\sigma_{8}$', r'$A_{1}$', r'$A_{2}$', r'$b_{TA}$', r'$\eta_{1}$', r'$\eta_{2}$'],
        # labels=[r'$w_{0}$', r'$w_{a}$', r'$\Omega_{m}$', r'$h$', r'$\Omega_{b}$', r'$n_{s}$', r'$\sigma_{8}$', r'$A_{1}$', r'$\eta_{1}$'],
        # labels=[r'$w_{0}$', r'$\Omega_{m}$', r'$h$', r'$\Omega_{b}$', r'$n_{s}$', r'$\sigma_{8}$', r'$A_{1}$', r'$\eta_{1}$'],
        labels=[r'$w_{0}$', r'$w_{a}$', r'$\Omega_{m}$', r'$h$', r'$\Omega_{b}$', r'$n_{s}$', r'$\sigma_{8}$'],
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

    # prior2 = nautilus.Prior()
    # prior2.add_parameter("w0", dist=(-1.05, -0.95))
    # prior2.add_parameter("wa", dist=(-0.25, 0.25))

    # prior2.add_parameter("w0", (-1.25, -0.75))
    # prior2.add_parameter("wa", (-0.5, 0.5))
    # prior2.add_parameter("Omega_m", (0.2, 0.4))
    # prior2.add_parameter("h", (0.5, 0.8))
    # prior2.add_parameter("Omega_b", (0.02, 0.08))
    # prior2.add_parameter("n_s", (0.8, 1.2))
    # prior2.add_parameter("sigma8", (0.75, 0.9))
    #
    # prior2.add_parameter("_A1", (-8, 8))
    # prior2.add_parameter("_eta1", (-6, 6))

    # prior2 = prior
    # sampler2 = nautilus.Sampler(
    #     # prior, log_normal_likelihood_ccl, n_live=200,
    #     prior2, log_normal_likelihood_ccl, n_live=200,
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
    #     filepath='/raid/scratch/wongj/mywork/3x2pt/6x2pt_sim_data/b/inference_chains/Cosmology_3x2pt_analytic.hdf5',
    #     pool=n_pool
    # )

    # sampler2.run(verbose=True)

    # points2, log_w2, log_l2 = sampler2.posterior()
    # points2 = points2[:,0:7]
    #
    # corner.corner(
    #     points2,
    #     weights=np.exp(log_w2),
    #     # bins=250,
    #     # bins=[3000, 2000, 300],
    #     # bins=[50, 30, 100, 30, 40, 100, 250, 1000, 1000, 200],
    #     # bins=[50,30,100,30,40,100,100,1500, 1500, 500, 500, 500],
    #     bins=hist_bins, #, 5000, 300],
    #     plot_density=False,
    #     no_fill_contours=True,
    #     color='darkred',
    #     # contour_kwargs={'linestyles':'--','linewidths':1.5},
    #     contour_kwargs={'linestyles':'--','linewidths':1.5},
    #     data_kwargs={'color': '0.45', 'ms': '0'},
    #     label_kwargs={'fontsize': '20'},
    #     # labels=[r'$w_{0}$', r'$w_{a}$', r'$\Omega_{m}$', r'$h$', r'$\Omega_{c}$', r'$n_{s}$', r'$\sigma_{8}$',
    #     #         r'$A_{1}$', r'$A_{2}$', r'$b_{TA}$', r'$\eta_{1}$', r'$\eta_{2}$'],
    #     labelpad=0.025,
    #     levels=(0.683, 0.955),
    #     smooth=1.5,
    #     smooth1d=True,
    #     fig=figure,
    #     # hist_kwargs={'linestyle':'--','linewidth':2.5,'dashes':(5, 6)},
    #     hist_kwargs={'linestyle':'-','linewidth':1.5},
    #     # hist2d_kwargs={'contour_kwargs':{'linestyles': '--'}},
    #     hist2d_kwargs={'contour_kwargs':{'linestyles': '-'}},
    #     # title_quantiles=[q_lower, 0.5, q_upper],
    #     # show_titles=True,
    #     # title_fmt='.4f'
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

    # one_sigma_1d = 0.999999999919680
    one_sigma_1d = 0.99999999999999999
    # one_sigma_1d = 0.99

    q_lower = 1 / 2 - one_sigma_1d / 2
    q_upper = 1 / 2 + one_sigma_1d / 2

    for i in range(ndim):
        # print(len(points[:,i]))
        # print(len(np.exp(log_w)))
        q_lo, q_mid, q_hi = corner.quantile(
            points[:,i], [q_lower, 0.5, q_upper], weights=np.exp(log_w)
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
        ax.tick_params(axis='both', direction='in',labelsize=20)

    axes = np.array(figure.axes).reshape((ndim,ndim))
    # print(xranges)
    # xranges = [priors[p][1] for p in range(len(priors))]
    # xranges = [[-1.15, -0.85], [-0.4, 0.4], [0.28,0.34], [0.55, 0.80], [0.02,0.07], [0.91, 1.01], [0.82,0.86], [0.58, 0.8], [-1.45,-1.25], [0.7,1.3], [-2.1,-1.3], [-2.1,-1.3]] # for IA
    # xranges = [[-1.15, -0.85], [-0.4, 0.4], [0.28,0.34], [0.55, 0.80], [0.02,0.07], [0.88, 1.04], [0.81,0.87], [1.95, 2.2], [0.3,0.7], [-1.5,0.25]] # for bias
    # xranges = [
    #     [-1.1390991431074178, -0.8614864430873895], [-0.45008765349943214, 0.4203948398332733],
    #     [0.2920769720310088, 0.3385946457273819], [0.5605128366682348, 0.7954783920595907],
    #     [0.02041858354543234, 0.06615883786084358], [0.9195099181351285, 0.9933313155491182],
    #     [0.8264167875194324, 0.8553209133826235], [0.581842574722146, 0.8295851618565178],
    #     [-1.4764214674201828, -1.2512921101586283], [0.6626333033051865, 1.3735536951789],
    #     [-2.10791258916056, -1.2916628125374878], [-2.8954472552587878, -2.122759974693125]]
    # xranges = [
    #     [-1.199502489274048, -0.7867440358341983],
    #     [-0.4999142828491949, 0.49999147868940874],
    #     [0.2813111893420181, 0.3494832657889116],
    #     [0.5119303393828264, 0.7999730342349741],
    #     [0.020031018898387015, 0.0725787093784679],
    #     [0.8992434347089814, 1.0279282252230308],
    #     [0.8174936257293406, 0.8635702219755874]]
    # xranges=[
    #     [-1.199502489274048, -0.7867440358341983],
    #     [-0.4999142828491949, 0.49999147868940874],
    #     [0.2813111893420181, 0.3494832657889116],
    #     [0.5119303393828264, 0.7999730342349741],
    #     [0.020031018898387015, 0.0725787093784679],
    #     [0.8992434347089814, 1.0279282252230308],
    #     [0.8174936257293406, 0.8635702219755874],
    #     [0.48343353896669466, 0.8787529651998153],
    #     [-1.5999785092077812, -1.1386689629390465],
    #     [0.5531661949053119, 1.560211252697805],
    #     [-2.5962369463664765, -0.8159752193277147],
    #     [-3.2361310581234783, -1.8170991845488031]]
    yranges = xranges[1:]

    # fid_vals = [-1, 0]
    # fid_vals = [-1,0,0.315,0.67,0.045,0.96,0.84048,0.7,-1.7] # for IA
    # fid_vals = [-1,0.315,0.67,0.045,0.96,0.84048,0.7,-1.7] # for IA
    # fid_vals = [-1,0,0.315,0.67,0.045,0.96,0.84048] # for cosmo
    # fid_vals = [-1,0.315,0.84048] # for cosmo
    # fid_vals = [-1,0,0.315,0.67,0.045,0.96,0.84048, 2.07, 0.5, -0.611] # for nl bias
    fid_vals = [-1,0,0.315,0.67,0.045,0.96,0.84048, 2.07, 2.07, 2.07, 2.07, 2.07, 2.07] # for bias
    # fid_vals = [-1,0,0.315,0.67,0.045,0.96,0.84048,0.7,-1.36,1.0,-1.7,-2.5] # for IA
    # fid_vals = [-1,0,0.315,0.67,0.045,0.96,0.84048,2.07,0.5,-0.611] # for IA
    # fid_vals = [2.07, 0.5, -0.611]
    # fid_vals = [-1,0,0.315,0.67,0.045,0.96,0.84048,2.07,0.5,-0.611]

    for i in range(ndim):
        ax = axes[i,i]
        ax.set_xlim(xranges[i][0], xranges[i][1])
        ax.axvline(fid_vals[i], color='0.5', linestyle=':')

    for yi in range(ndim):
        for xi in range(yi):
            ax = axes[yi, xi]
            ax.set_xlim(xranges[xi][0],xranges[xi][1])
            ax.set_ylim(yranges[yi-1][0],yranges[yi-1][1])
            ax.axvline(fid_vals[xi],color='0.5', linestyle=':')
            ax.axhline(fid_vals[yi],color='0.5', linestyle=':')

    # patch1 = mpatches.Patch(color='darkred', label='3'+r'$\times$'+'2pt')
    patch2 = mpatches.Patch(color='royalblue', label='Stage IV-like 3x2pt + \nSO-like CMB lensing\n(6x2pt)')
    patch1 = mpatches.Patch(color='darkred', label='Stage IV-like 3x2pt')
    # patch2 = mpatches.Patch(color='royalblue', label='Stage IV-like 3x2pt (Analytic Covariance)')
    # patch1 = mpatches.Patch(color='darkred', label='Stage IV-like 3x2pt (Numerical Covariance)')

    figure.legend(handles=[patch1, patch2],loc='center right',fontsize=15)   #legend can be centre right for big plots
    # save_fig_dir = '/raid/scratch/wongj/mywork/3x2pt/6x2pt_sim_data/ff/inference_chains/bias_l500.png'
    #
    # if os.path.exists(save_fig_dir):
    #     print('WARNING! File exists, did not overwrite')
    # else:
    #     plt.savefig(save_fig_dir,dpi=200)

    plt.show()
    #
    # sampler2 = nautilus.Sampler(
    #     prior, log_normal_likelihood_ccl, n_live=200,
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
    #     filepath='/raid/scratch/wongj/mywork/3x2pt/6x2pt_sim_data/ff/inference_chains/Cosmology_IA_3x2pt_numerical.hdf5',
    #     pool=n_pool
    # )
    #
    # sampler3 = nautilus.Sampler(
    #     prior, log_normal_likelihood_ccl, n_live=200,
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
    #     filepath='/raid/scratch/wongj/mywork/3x2pt/6x2pt_sim_data/ff/inference_chains/Cosmology_IA_3x2pt_analytic.hdf5',
    #     pool=n_pool
    # )
    #
    # sampler4 = nautilus.Sampler(
    #     prior, log_normal_likelihood_ccl, n_live=200,
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
    #     filepath='/raid/scratch/wongj/mywork/3x2pt/6x2pt_sim_data/ff/inference_chains/Cosmology_IA_6x2pt_analytic.hdf5',
    #     pool=n_pool
    # )
    #
    # spider_plot.spiderplot(spider_plot.spider_data4(sampler1=sampler2, sampler2=sampler, sampler3=sampler3, sampler4=sampler4))
    # #
    # spider_plot.spiderplot(spider_plot.spider_data(sampler1=sampler2, sampler2=sampler))



def execute(pipeline_variables_path, covariance_matrix_type, priors, checkpoint_filename, bi_marg=False, mi_marg=False, Dzi_marg=False, A1i_marg=False):

    sampler_config_dict = sampler_config(pipeline_variables_path=pipeline_variables_path)
    # systematics_dict = generate_cls.setup_systematics_dict(pipeline_variables_path=pipeline_variables_path)

    save_dir = sampler_config_dict['save_dir']
    n_bps = sampler_config_dict['n_bandpowers']

    inference_dir = save_dir + 'inference_chains/'
    if not os.path.exists(inference_dir):
        os.makedirs(inference_dir)

    data_vector = np.load(save_dir + 'measured_6x2pt_bps/obs_{}bp.npz'.format(n_bps))['obs_bp']
    sampler_checkpoint_file = inference_dir + checkpoint_filename

    if covariance_matrix_type == 'analytic':
        covariance_matrix = np.load(save_dir + 'analytic_covariance/cov_{}bp.npz'.format(n_bps))['cov']

    else:
        assert covariance_matrix_type == 'numerical'
        covariance_matrix = np.load(save_dir + 'numerical_covariance/cov_{}bp.npz'.format(n_bps))['cov']

    # covariance_matrix = np.load(save_dir + 'cov_fromsim/cov_{}bp.npz'.format(n_bps))['cov']
    # covariance_matrix = np.load(covariance_matrix_path)['cov']

    inverse_covariance = np.linalg.inv(covariance_matrix)

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
    '''
    # log_normal_likelihood_ccl({'w0':-1,'wa':0}, sampler_config_dict, mix_mats, data_vector, inverse_covariance)
    log_normal_likelihood_ccl(
        # {'Omega_c':0.3252,
        #  'Omega_b':0.0361,
        #  'h':0.5342,
        #  'n_s':1.00,
        #  'sigma8':0.8159,
        #  'Omega_k':0.0,
        #  'w0':-0.7559,
        #  'wa':-0.4476,
        #  '_A1':0.1803,
        #  '_eta1':1.2903},
        {'Omega_c': 0.27,
         'Omega_b': 0.045,
         'h': 0.67,
         'n_s': 0.96,
         'sigma8': 0.840,
         'Omega_k': 0.0,
         'w0': -1.0,
         'wa': 0.0,
         '_A1': 0.7,
         '_eta1': -1.7},
        config_dict=sampler_config_dict,
        pipeline_variables_path=pipeline_variables_path,
        mixmats=mixmats,
        data_vector=data_vector,
        inverse_covariance=inverse_covariance)
    '''
    # test({'_Dz_1':0.0,'_Dz_2':0.0,'_Dz_3':0.0}, config_dict=sampler_config_dict, pipeline_variables_path=pipeline_variables_path, mixmats=mixmats, data_vector=data_vector, inverse_covariance=inverse_covariance)

    run_nautilus(
        sampler_config_dict=sampler_config_dict,
        pipeline_variables_path=pipeline_variables_path,
        mixmats=mixmats,
        data_vector=data_vector,
        inverse_covariance=inverse_covariance,
        sampler_checkpoint_file=sampler_checkpoint_file,
        priors=priors,
        bi_marg=bi_marg,
        mi_marg=mi_marg,
        Dzi_marg=Dzi_marg,
        A1i_marg=A1i_marg
    )
