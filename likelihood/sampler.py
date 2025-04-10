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
from likelihood.model_pcl import PCl_bandpowers_1x2pt, PCl_bandpowers_3x2pt, PCl_bandpowers_6x2pt


def parse_nautilus_params_for_ccl(params):
    """
    nautilus propogates params as a dictionary, with each value an unshaped ndarray
    """

    return {key:val[()] for key, val in params.items()}


def split_ccl_parameters(params):
    ccl_params = {}
    nuisance_params = {}

    # Some list of nuisance parameters
    are_nuisance_params = ['a']
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
    A_s = float(config['cosmology']['A_s'])
    # sigma8 = float(config['cosmology']['sigma8'])
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
        'A_s': A_s,
        # 'sigma8': sigma8,
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


'''
def conv_3x2pt_bps(n_zbin, n_bp, save_dir, recov_cat_bps_path):

    """
    Collect all 3x2pt tomographic Pseudo bandpowers and store into single array.

    Parameters
    ----------
    n_zbin (float): Number of tomographic redshift bins
    n_bp (float):   Number of bandpowers
    save_dir (str): Path to directory that stores all measurement data (MEASUREMENT_SAVE_DIR from the
                    set_variables_3x2pt_measurement.ini file)
    recov_cat_bps_path (str): Location to store combined 3x2pt data vector as .npz file
    obs_type (str): Use the data measured from simulation ('obs') or the fiducial data ('fid') to generate the
                    combined data vector

    Returns
    -------
    Saves array in .npz format of the combined 3x2pt data vector.
    """

    n_field = 2 * n_zbin
    n_spec = n_field * (n_field + 1) // 2

    # Form list of power spectra
    fields = [f'{f}{z}' for z in range(1, n_zbin + 1) for f in ['N', 'E']]
    assert len(fields) == n_field

    spectra = [fields[row] + fields[row + diag] for diag in range(n_field) for row in range(n_field - diag)]

    spec_1 = [fields[row] for diag in range(n_field) for row in range(n_field - diag)]
    spec_2 = [fields[row + diag] for diag in range(n_field) for row in range(n_field - diag)]

    # Identify fields and redshift bin ids used to generate specific power spectrum

    field_1 = [mysplit(spec_1_id)[0] for spec_1_id in spec_1]
    zbin_1 = [mysplit(spec_1_id)[1] for spec_1_id in spec_1]
    field_2 = [mysplit(spec_2_id)[0] for spec_2_id in spec_2]
    zbin_2 = [mysplit(spec_2_id)[1] for spec_2_id in spec_2]

    obs_bp = np.full((n_spec, n_bp), np.nan)

    for spec_idx in range(len(spectra)):
        f1 = field_1[spec_idx]
        f2 = field_2[spec_idx]
        z1 = int(zbin_1[spec_idx])
        z2 = int(zbin_2[spec_idx])

        if f1 == 'N' and f2 == 'N':
            measured_bp_file = save_dir + 'inference_chains/galaxy_cl/PCl_Bandpowers_gal_gal_bin_{}_{}.txt'. \
                format(max(z1, z2), min(z1, z2))

        elif f1 == 'E' and f2 == 'E':
            measured_bp_file = save_dir + 'inference_chains/shear_cl/PCl_Bandpowers_EE_bin_{}_{}.txt'. \
                format(max(z1, z2), min(z1, z2))

        elif f1 == 'N' and f2 == 'E':
            measured_bp_file = save_dir + 'inference_chains/galaxy_shear_cl/PCl_Bandpowers_gal_E_bin_{}_{}.txt'. \
                format(z1, z2)

        elif f1 == 'E' and f2 == 'N':
            # switch around, i.e. open f2z2f1z1
            measured_bp_file = save_dir + 'inference_chains/galaxy_shear_cl/PCl_Bandpowers_gal_E_bin_{}_{}.txt'. \
                format(z2, z1)

        else:
            print('Unexpected Spectra Found - Please check inference pipeline')
            sys.exit()

        measured_bp = np.loadtxt(measured_bp_file)
        obs_bp[spec_idx, :] = measured_bp

    obs_bp_path = os.path.join(recov_cat_bps_path, f'obs_{n_bp}bp.npz')
    obs_bp_header = (f'Observed bandpowers for 3x2pt simulation')
    np.savez_compressed(obs_bp_path, obs_bp=obs_bp, header=obs_bp_header)
    return obs_bp


def conv_1x2pt_bps(n_zbin, n_bp, save_dir, recov_cat_bps_path, field='E'):

    """
    Collect all Pseudo bandpowers for a tomographic 1x2pt (shear only or clustering only) and store into single array.

    Parameters
    ----------
    n_zbin (float): Number of tomographic redshift bins
    n_bp (float):   Number of bandpowers
    save_dir (str): Path to directory that stores all measurement data (MEASUREMENT_SAVE_DIR from the
                    set_variables_3x2pt_measurement.ini file)
    recov_cat_bps_path (str): Location to store combined 1x2pt data vector as .npz file
    obs_type (str): Use the data measured from simulation ('obs') or the fiducial data ('fid') to generate the
                    combined data vector
    field (str):    'E' or 'N' to specify the 1x2pt measurement is cosmic shear only or angular clustering only

    Returns
    -------
    Saves array in .npz format of the combined 3x2pt data vector.
    """

    n_field = n_zbin
    n_spec = n_field * (n_field + 1) // 2

    # Form list of power spectra
    if field == 'E':
        fields = [f'E{z}' for z in range(1, n_zbin + 1)]
    else:
        assert field == 'N'
        fields = [f'N{z}' for z in range(1, n_zbin + 1)]

    assert len(fields) == n_field
    spectra = [fields[row] + fields[row + diag] for diag in range(n_field) for row in range(n_field - diag)]

    spec_1 = [fields[row] for diag in range(n_field) for row in range(n_field - diag)]
    spec_2 = [fields[row + diag] for diag in range(n_field) for row in range(n_field - diag)]

    field_1 = [mysplit(spec_1_id)[0] for spec_1_id in spec_1]
    zbin_1 = [mysplit(spec_1_id)[1] for spec_1_id in spec_1]
    field_2 = [mysplit(spec_2_id)[0] for spec_2_id in spec_2]
    zbin_2 = [mysplit(spec_2_id)[1] for spec_2_id in spec_2]

    obs_bp = np.full((n_spec, n_bp), np.nan)

    for spec_idx in range(len(spectra)):
        f1 = field_1[spec_idx]
        f2 = field_2[spec_idx]
        z1 = int(zbin_1[spec_idx])
        z2 = int(zbin_2[spec_idx])

        if field == 'E':
            measured_bp_file = save_dir + 'inference_chains/shear_cl/PCl_Bandpowers_EE_bin_{}_{}.txt'. \
                format(max(z1, z2), min(z1, z2))
            measured_bp = np.loadtxt(measured_bp_file)
            obs_bp[spec_idx, :] = measured_bp

        else:
            assert field == 'N'
            measured_bp_file = save_dir + 'inference_chains/galaxy_cl/PCl_Bandpowers_gal_gal_bin_{}_{}.txt'. \
                format(max(z1, z2), min(z1, z2))
            measured_bp = np.loadtxt(measured_bp_file)
            obs_bp[spec_idx, :] = measured_bp

    obs_bp_path = os.path.join(recov_cat_bps_path, f'obs_{n_bp}bp.npz')
    obs_bp_header = (f'Observed bandpowers for 1x2pt simulation')
    np.savez_compressed(obs_bp_path, obs_bp=obs_bp, header=obs_bp_header)
    return obs_bp
'''


def generate_pseudo_bps_model(cosmo_params, config_dict, mixmats):
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
    A_s = config_dict['A_s']
    # sigma8 = config_dict['sigma8']
    n_s = config_dict['n_s']
    Omega_k = config_dict['Omega_k']
    w0 = config_dict['w0']
    wa = config_dict['wa']

    # A fiducial cosmology
    fid_cosmo_dict = {
        'Omega_c': Omega_c,
        'Omega_b': Omega_b,
        'h': h,
        'A_s': A_s,
        # 'sigma8': sigma8,
        'n_s': n_s,
        'Omega_k': Omega_k,
        'w0': w0,
        'wa': wa,
        'extra_parameters': {'camb':{'dark_energy_model':'ppf'}}
    }

    ccl_params, nuisance_params = split_ccl_parameters(cosmo_params)
    fid_cosmo_dict.update(ccl_params)
    fid_cosmo = ccl.Cosmology(**fid_cosmo_dict)

    # Now we have to make the Cls, PCls, bandpowers, then combined data vector
    cls_dict = {'galaxy_cl': {'ell':ells},
                'shear_cl': {'ell':ells},
                'galaxy_shear_cl': {'ell':ells},
                'cmbkappa_cl': {'ell':ells},
                'galaxy_cmbkappa_cl': {'ell':ells},
                'shear_cmbkappa_cl': {'ell':ells},
                'null_spectra': {'ell':ells}
                }

    # These parameters can probably be read in from a config file
    # Galaxy bias
    b = np.ones_like(z)
    # bz = 0.95/ccl.growth_factor(fid_cosmo,1./(1+z))

    # Intrinsic alignment amplitude
    A_IA = 0.6 * np.ones_like(z)

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

    # Now we need to convert to Pseudo-Cls and then bandpowers
    theory_cls_dir = save_dir + 'inference_chains/'
    obs_noise_cls_dir = save_dir + 'measured_noise_cls/'

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


def log_normal_likelihood_ccl(params, config_dict, mixmats, data_vector, inverse_covariance):

    model_vector = generate_pseudo_bps_model(cosmo_params=params, config_dict=config_dict, mixmats=mixmats)

    # Need to stack data vector
    assert model_vector.shape == data_vector.shape
    n_spec, n_bandpower = model_vector.shape

    n_data = n_spec * n_bandpower
    model_vector = np.reshape(model_vector, n_data)
    data_vector = np.reshape(data_vector, n_data)

    d_vector = model_vector - data_vector

    return -0.5 * d_vector @ inverse_covariance @ d_vector


def test(params, config_dict, mixmats, data_vector):

    model_vector = generate_pseudo_bps_model(cosmo_params=params, config_dict=config_dict, mixmats=mixmats)

    # Need to stack data vector
    assert model_vector.shape == data_vector.shape
    n_spec, n_bandpower = model_vector.shape

    n_data = n_spec * n_bandpower
    model_vector = np.reshape(model_vector, n_data)
    data_vector = np.reshape(data_vector, n_data)

    xs = np.arange(1, n_bandpower + 1)

    for i in range(n_spec):
        fig, ax = plt.subplots()
        ax.plot(xs, model_vector[(i*n_bandpower):(i*n_bandpower)+n_bandpower], color='0')
        ax.plot(xs, data_vector[(i*n_bandpower):(i*n_bandpower)+n_bandpower], marker = 'x', linestyle='None')
        plt.show()


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


def run_nautilus(sampler_config_dict, mixmats, data_vector, inverse_covariance, sampler_checkpoint_file):

    prior = nautilus.Prior()
    prior.add_parameter("w0", dist=(-1.5, -0.5))
    prior.add_parameter("wa", dist=(-0.5, 0.5))
    prior.add_parameter("Omega_c", dist=(0.2, 0.4))
    prior.add_parameter("h", dist=(0.5, 0.8))

    sampler = nautilus.Sampler(
        prior, log_normal_likelihood_ccl, n_live=100,
        likelihood_kwargs={
            "config_dict": sampler_config_dict,
            "mixmats": mixmats,
            "data_vector": data_vector,
            "inverse_covariance": inverse_covariance},
        filepath=sampler_checkpoint_file
    )
    
    start_time = time.time()
    sampler.run(verbose=True)
    
    points, log_w, log_l = sampler.posterior()
    print("--- %s seconds ---" % (time.time() - start_time))
    
    corner.corner(
        points, weights=np.exp(log_w), bins=30, labels=prior.keys
    )
    plt.show()


def execute(pipeline_variables_path):

    sampler_config_dict = sampler_config(pipeline_variables_path=pipeline_variables_path)

    save_dir = sampler_config_dict['save_dir']
    n_bps = sampler_config_dict['n_bandpowers']

    inference_dir = save_dir + 'inference_chains/'
    if not os.path.exists(inference_dir):
        os.makedirs(inference_dir)

    data_vector = np.load(save_dir + 'measured_6x2pt_bps/obs_{}bp.npz'.format(n_bps))['obs_bp']
    sampler_checkpoint_file = inference_dir + "nautilus_inference.h5"

    covariance_matrix = np.load(save_dir + 'cov_fromsim/cov_{}bp.npz'.format(n_bps))['cov']
    inverse_covariance = np.linalg.inv(covariance_matrix)

    # generate_mixmats(sampler_config_dict=sampler_config_dict)

    mix_mats_save_path = inference_dir + 'mixmats.npz'
    mixmats_all = np.load(mix_mats_save_path)

    mixmats = {'mixmat_nn_to_nn': mixmats_all['mixmat_nn_to_nn'],
               'mixmat_ne_to_ne': mixmats_all['mixmat_ne_to_ne'],
               'mixmat_ee_to_ee': mixmats_all['mixmat_ee_to_ee'],
               'mixmat_bb_to_ee': mixmats_all['mixmat_bb_to_ee'],
               'mixmat_kk_to_kk': mixmats_all['mixmat_kk_to_kk'],
               'mixmat_nn_to_kk': mixmats_all['mixmat_nn_to_kk'],
               'mixmat_ke_to_ke': mixmats_all['mixmat_ke_to_ke']
               }

    # log_normal_likelihood_ccl({'w0':-1,'wa':0}, sampler_config_dict, mix_mats, data_vector, inverse_covariance)

    # test({'w0':-1,'wa':0}, sampler_config_dict, mixmats, data_vector)

    run_nautilus(
        sampler_config_dict=sampler_config_dict,
        mixmats=mixmats,
        data_vector=data_vector,
        inverse_covariance=inverse_covariance,
        sampler_checkpoint_file=sampler_checkpoint_file
    )
