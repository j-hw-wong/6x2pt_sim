"""
Collect all 3x2pt observed spectra (pseudo-Cl bandpowers) into one file for input into inference code
"""

import os
import sys
import configparser
import numpy as np


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


def conv_6x2pt_bps(n_zbin, n_bp, save_dir, recov_cat_bps_path, obs_type='obs'):

    """
    Collect all 6x2pt tomographic Pseudo bandpowers and store into single array.

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
    Saves array in .npz format of the combined 6x2pt data vector.
    """

    fiducial_cosmology_dir = save_dir + 'fiducial_cosmology/'

    n_field = 2 * n_zbin
    # n_spec = n_field * (n_field + 1) // 2

    # Form list of power spectra
    fields = [f'{f}{z}' for z in range(1, n_zbin + 1) for f in ['N', 'E']]
    # assert len(fields) == n_field

    spectra = [fields[row] + fields[row + diag] for diag in range(n_field) for row in range(n_field - diag)]

    spec_1 = [fields[row] for diag in range(n_field) for row in range(n_field - diag)]
    spec_2 = [fields[row + diag] for diag in range(n_field) for row in range(n_field - diag)]

    for i in range(n_zbin):
        spectra.append('E{}K1'.format(i+1))
        spectra.append('N{}K1'.format(i+1))

        spec_1.append('E{}'.format(i+1))
        spec_1.append('N{}'.format(i+1))

        spec_2.append('K1')
        spec_2.append('K1')

    spectra.append('K1K1')
    spec_1.append('K1')
    spec_2.append('K1')

    # Identify fields and redshift bin ids used to generate specific power spectrum

    field_1 = [mysplit(spec_1_id)[0] for spec_1_id in spec_1]
    zbin_1 = [mysplit(spec_1_id)[1] for spec_1_id in spec_1]
    field_2 = [mysplit(spec_2_id)[0] for spec_2_id in spec_2]
    zbin_2 = [mysplit(spec_2_id)[1] for spec_2_id in spec_2]

    n_spec = len(spectra)
    obs_bp = np.full((n_spec, n_bp), np.nan)

    for spec_idx in range(len(spectra)):
        f1 = field_1[spec_idx]
        f2 = field_2[spec_idx]
        z1 = int(zbin_1[spec_idx])
        z2 = int(zbin_2[spec_idx])

        if obs_type == 'obs':

            if f1 == 'N' and f2 == 'N':
                measured_bp_file = recov_cat_bps_path + 'galaxy_bp/bin_{}_{}.txt'.format(max(z1, z2), min(z1, z2))

            elif f1 == 'E' and f2 == 'E':
                measured_bp_file = recov_cat_bps_path + 'shear_bp/Cl_EE/bin_{}_{}.txt'.format(max(z1, z2), min(z1, z2))

            elif f1 == 'N' and f2 == 'E':
                measured_bp_file = recov_cat_bps_path + 'galaxy_shear_bp/bin_{}_{}.txt'.format(z1, z2)

            elif f1 == 'E' and f2 == 'N':
                # switch around, i.e. open f2z2f1z1
                measured_bp_file = recov_cat_bps_path + 'galaxy_shear_bp/bin_{}_{}.txt'.format(z2, z1)

            elif f1 == 'E' and f2 == 'K':
                measured_bp_file = recov_cat_bps_path + 'shear_cmbkappa_bp/kCMB_E/bin_{}_{}.txt'.format(z1, z2)

            elif f1 == 'N' and f2 == 'K':
                measured_bp_file = recov_cat_bps_path + 'galaxy_cmbkappa_bp/bin_{}_{}.txt'.format(z1, z2)

            elif f1 == 'K' and f2 == 'K':
                measured_bp_file = recov_cat_bps_path + 'cmbkappa_bp/bin_{}_{}.txt'.format(z1, z2)

            else:
                print('Unexpected Spectra Found - Please check inference pipeline')
                sys.exit()

            measured_bp = np.loadtxt(measured_bp_file)
            obs_bp[spec_idx, :] = measured_bp

        else:

            assert obs_type == 'fid'

            if f1 == 'N' and f2 == 'N':
                measured_bp_file = fiducial_cosmology_dir + 'galaxy_cl/PCl_Bandpowers_gal_gal_bin_{}_{}.txt'. \
                    format(max(z1, z2), min(z1, z2))

            elif f1 == 'E' and f2 == 'E':
                measured_bp_file = fiducial_cosmology_dir + 'shear_cl/PCl_Bandpowers_EE_bin_{}_{}.txt'. \
                    format(max(z1, z2), min(z1, z2))

            elif f1 == 'N' and f2 == 'E':
                measured_bp_file = fiducial_cosmology_dir + 'galaxy_shear_cl/PCl_Bandpowers_gal_E_bin_{}_{}.txt'. \
                    format(z1, z2)

            elif f1 == 'E' and f2 == 'N':
                # switch around, i.e. open f2z2f1z1
                measured_bp_file = fiducial_cosmology_dir + 'galaxy_shear_cl/PCl_Bandpowers_gal_E_bin_{}_{}.txt'. \
                    format(z2, z1)

            elif f1 == 'E' and f2 == 'K':
                measured_bp_file = fiducial_cosmology_dir + \
                                   'shear_cmbkappa_cl/PCl_Bandpowers_kCMB_E_bin_{}_{}.txt'.format(z1, z2)

            elif f1 == 'N' and f2 == 'K':
                measured_bp_file = fiducial_cosmology_dir + \
                                   'galaxy_cmbkappa_cl/PCl_Bandpowers_kCMB_gal_bin_{}_{}.txt'.format(z1, z2)

            elif f1 == 'K' and f2 == 'K':
                measured_bp_file = fiducial_cosmology_dir + \
                                   'cmbkappa_cl/PCl_Bandpowers_kCMB_kCMB_bin_{}_{}.txt'.format(z1, z2)

            else:
                print('Unexpected Spectra Found - Please check inference pipeline')
                sys.exit()

            measured_bp = np.loadtxt(measured_bp_file)
            obs_bp[spec_idx, :] = measured_bp

    obs_bp_path = os.path.join(recov_cat_bps_path, f'obs_{n_bp}bp.npz')
    obs_bp_header = (f'Observed bandpowers for 3x2pt simulation')
    np.savez_compressed(obs_bp_path, obs_bp=obs_bp, header=obs_bp_header)
    return obs_bp


def conv_3x2pt_bps(n_zbin, n_bp, save_dir, recov_cat_bps_path, obs_type='obs'):

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

        if obs_type == 'obs':

            if f1 == 'N' and f2 == 'N':
                measured_bp_file = recov_cat_bps_path + 'galaxy_bp/bin_{}_{}.txt'.format(max(z1, z2), min(z1, z2))

            elif f1 == 'E' and f2 == 'E':
                measured_bp_file = recov_cat_bps_path + 'shear_bp/Cl_EE/bin_{}_{}.txt'.format(max(z1, z2), min(z1, z2))

            elif f1 == 'N' and f2 == 'E':
                measured_bp_file = recov_cat_bps_path + 'galaxy_shear_bp/bin_{}_{}.txt'.format(z1, z2)

            elif f1 == 'E' and f2 == 'N':
                # switch around, i.e. open f2z2f1z1
                measured_bp_file = recov_cat_bps_path + 'galaxy_shear_bp/bin_{}_{}.txt'.format(z2, z1)

            else:
                print('Unexpected Spectra Found - Please check inference pipeline')
                sys.exit()

            measured_bp = np.loadtxt(measured_bp_file)
            obs_bp[spec_idx, :] = measured_bp

        else:

            assert obs_type == 'fid'

            if f1 == 'N' and f2 == 'N':
                measured_bp_file = save_dir + 'theory_cls/galaxy_cl/PCl_Bandpowers_gal_gal_bin_{}_{}.txt'. \
                    format(max(z1, z2), min(z1, z2))

            elif f1 == 'E' and f2 == 'E':
                measured_bp_file = save_dir + 'theory_cls/shear_cl/PCl_Bandpowers_EE_bin_{}_{}.txt'. \
                    format(max(z1, z2), min(z1, z2))

            elif f1 == 'N' and f2 == 'E':
                measured_bp_file = save_dir + 'theory_cls/galaxy_shear_cl/PCl_Bandpowers_gal_E_bin_{}_{}.txt'. \
                    format(z1, z2)

            elif f1 == 'E' and f2 == 'N':
                # switch around, i.e. open f2z2f1z1
                measured_bp_file = save_dir + 'theory_cls/galaxy_shear_cl/PCl_Bandpowers_gal_E_bin_{}_{}.txt'. \
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


def conv_1x2pt_bps(n_zbin, n_bp, save_dir, recov_cat_bps_path, obs_type='obs', field='E'):

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

    # Form list of power spectra
    if field == 'E':
        fields = [f'E{z}' for z in range(1, n_zbin + 1)]
    elif field == 'N':
        fields = [f'N{z}' for z in range(1, n_zbin + 1)]
    elif field == 'EK':
        fields = [f'E{z}K' for z in range(1, n_zbin + 1)]
    elif field == 'NK':
        fields = [f'N{z}K' for z in range(1, n_zbin + 1)]
    else:
        assert field == 'K'
        fields = ['K']

    if field == 'E' or field == 'N':

        n_field = n_zbin
        n_spec = n_field * (n_field + 1) // 2

        assert len(fields) == n_field
        spectra = [fields[row] + fields[row + diag] for diag in range(n_field) for row in range(n_field - diag)]

        spec_1 = [fields[row] for diag in range(n_field) for row in range(n_field - diag)]
        spec_2 = [fields[row + diag] for diag in range(n_field) for row in range(n_field - diag)]

        field_1 = [mysplit(spec_1_id)[0] for spec_1_id in spec_1]
        zbin_1 = [mysplit(spec_1_id)[1] for spec_1_id in spec_1]
        field_2 = [mysplit(spec_2_id)[0] for spec_2_id in spec_2]
        zbin_2 = [mysplit(spec_2_id)[1] for spec_2_id in spec_2]

    elif field == 'EK' or field == 'NK':
        spectra = fields
        n_spec = n_zbin
        zbin_1 = range(1, n_zbin + 1)
        zbin_2 = [1] * n_zbin

    else:
        assert field == 'K'
        spectra = fields
        n_spec = 1
        zbin_1 = [1]
        zbin_2 = [1]

    obs_bp = np.full((n_spec, n_bp), np.nan)

    for spec_idx in range(len(spectra)):
        # f1 = field_1[spec_idx]
        # f2 = field_2[spec_idx]
        z1 = int(zbin_1[spec_idx])
        z2 = int(zbin_2[spec_idx])

        if obs_type == 'obs':

            if field == 'E':
                measured_bp_file = recov_cat_bps_path + 'shear_bp/Cl_EE/bin_{}_{}.txt'.format(max(z1, z2), min(z1, z2))
                measured_bp = np.loadtxt(measured_bp_file)
                obs_bp[spec_idx, :] = measured_bp

            elif field == 'N':
                measured_bp_file = recov_cat_bps_path + 'galaxy_bp/bin_{}_{}.txt'.format(max(z1, z2), min(z1, z2))
                measured_bp = np.loadtxt(measured_bp_file)
                obs_bp[spec_idx, :] = measured_bp

            elif field == 'EK':
                measured_bp_file = recov_cat_bps_path + 'shear_cmbkappa_bp/kCMB_E/bin_{}_{}.txt'.format(z1, z2)
                measured_bp = np.loadtxt(measured_bp_file)
                obs_bp[spec_idx, :] = measured_bp

            elif field == 'NK':
                measured_bp_file = recov_cat_bps_path + 'galaxy_cmbkappa_bp/bin_{}_{}.txt'.format(z1, z2)
                measured_bp = np.loadtxt(measured_bp_file)
                obs_bp[spec_idx, :] = measured_bp

            else:
                assert field == 'K'
                measured_bp_file = recov_cat_bps_path + 'cmbkappa_bp/bin_{}_{}.txt'.format(z1, z2)
                measured_bp = np.loadtxt(measured_bp_file)
                obs_bp[spec_idx, :] = measured_bp
        else:
            assert obs_type == 'fid'
            if field == 'E':
                measured_bp_file = save_dir + 'fiducial_cosmology/shear_cl/PCl_Bandpowers_EE_bin_{}_{}.txt'. \
                    format(max(z1, z2), min(z1, z2))
                measured_bp = np.loadtxt(measured_bp_file)
                obs_bp[spec_idx, :] = measured_bp

            elif field == 'N':
                measured_bp_file = save_dir + 'fiducial_cosmology/galaxy_cl/PCl_Bandpowers_gal_gal_bin_{}_{}.txt'. \
                    format(max(z1, z2), min(z1, z2))
                measured_bp = np.loadtxt(measured_bp_file)
                obs_bp[spec_idx, :] = measured_bp

            elif field == 'EK':
                measured_bp_file = recov_cat_bps_path + 'fiducial_cosmology/PCl_Bandpowers_kCMB_E_bin_{}_{}.txt'.format(z1, z2)
                measured_bp = np.loadtxt(measured_bp_file)
                obs_bp[spec_idx, :] = measured_bp

            elif field == 'NK':
                measured_bp_file = recov_cat_bps_path + 'fiducial_cosmology/PCl_Bandpowers_kCMB_gal_bin_{}_{}.txt'.format(z1, z2)
                measured_bp = np.loadtxt(measured_bp_file)
                obs_bp[spec_idx, :] = measured_bp

            else:
                assert field == 'K'
                measured_bp_file = recov_cat_bps_path + 'fiducial_cosmology/PCl_Bandpowers_kCMB_kCMB_bin_{}_{}.txt'.format(z1, z2)
                # print(measured_bp_file)
                measured_bp = np.loadtxt(measured_bp_file)
                obs_bp[spec_idx, :] = measured_bp

    obs_bp_path = os.path.join(recov_cat_bps_path, f'obs_{n_bp}bp.npz')
    obs_bp_header = (f'Observed bandpowers for 1x2pt simulation')
    np.savez_compressed(obs_bp_path, obs_bp=obs_bp, header=obs_bp_header)
    return obs_bp


def execute(pipeline_variables_path):

    """
    Main function to generate combined 3x2pt or 1x2pt joint tomographic data vector
    """

    # Identify variable used for calculation.
    # pipeline_variables_path = os.environ['PIPELINE_VARIABLES_PATH']

    config = configparser.ConfigParser()
    config.read(pipeline_variables_path)

    save_dir = str(config['simulation_setup']['SIMULATION_SAVE_DIR'])
    recov_cat_bps_path = save_dir + 'measured_6x2pt_bps/'

    n_zbin = int(float(config['redshift_distribution']['N_ZBIN']))
    n_bp = int(float(config['measurement_setup']['N_BANDPOWERS']))

    # Determine observation type, i.e. 3x2pt or 1x2pt and if 1x2pt, which field ('E' or 'N')

    obs_spec = str(config['measurement_setup']['OBS_TYPE'])
    obs_field = str(config['measurement_setup']['FIELD'])

    if obs_spec == '3X2PT':
        conv_3x2pt_bps(
            n_zbin=n_zbin,
            n_bp=n_bp,
            save_dir=save_dir,
            recov_cat_bps_path=recov_cat_bps_path,
            obs_type='obs')

    elif obs_spec == '1X2PT':
        conv_1x2pt_bps(
            n_zbin=n_zbin,
            n_bp=n_bp,
            save_dir=save_dir,
            recov_cat_bps_path=recov_cat_bps_path,
            obs_type='obs',
            field=obs_field)

    elif obs_spec == '6X2PT':
        conv_6x2pt_bps(
            n_zbin=n_zbin,
            n_bp=n_bp,
            save_dir=save_dir,
            recov_cat_bps_path=recov_cat_bps_path,
            obs_type='obs')

# if __name__ == '__main__':
#     main()
