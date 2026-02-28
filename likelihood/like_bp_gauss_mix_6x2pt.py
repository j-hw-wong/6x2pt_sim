"""
Likelihood module to evaluate the joint likelihood of a set of tomographic 3x2pt power spectra, binned into bandpowers,
on the cut sky using a multivariate Gaussian likelihood.

The main functions are setup, which should be called once per analysis, and execute, which is called for every new
point in parameter space.
"""

import numpy as np
import os.path

def is_even(x):
    """
    True if x is even, false otherwise.

    Args:
        x (float): Number to test.

    Returns:
        bool: True if even.
    """
    return x % 2 == 0


def is_odd(x):
    """
    True if x is odd, false otherwise.

    Args:
        x (float): Number to test.

    Returns:
        bool: True if odd.
    """
    return x % 2 == 1


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


def load_cls_dict(n_zbin, cls_dict, lmax=None, lmin=0):
    """
    Given the number of redshift bins and relevant directories, load power spectra (position, shear, cross) in the
    correct order (diagonal / healpy new=True ordering).
    If lmin is supplied, the output will be padded to begin at l=0.

    Args:
        n_zbin (int): Number of redshift bins.
        cls_dict (dict): Dictionary of cls
        lmax (int, optional): Maximum l to load - if not supplied, will load all lines, which requires the individual
                              lmax of each file to be consistent.
        lmin (int, optional): Minimum l supplied. Output will be padded with zeros below this point.

    Returns:
        2D numpy array: All Cls, with different spectra along the first axis and increasing l along the second.
    """

    # Calculate number of fields assuming 1 position field and 1 shear field per redshift bin
    n_field = 2 * n_zbin

    # Form list of power spectra
    fields = [f'{f}{z}' for z in range(1, n_zbin + 1) for f in ['N', 'E']]
    # assert len(fields) == n_field

    spectra_list = [fields[row] + fields[row + diag] for diag in range(n_field) for row in range(n_field - diag)]

    spec_1 = [fields[row] for diag in range(n_field) for row in range(n_field - diag)]
    spec_2 = [fields[row + diag] for diag in range(n_field) for row in range(n_field - diag)]

    for i in range(n_zbin):
        spectra_list.append('E{}K1'.format(i + 1))
        spectra_list.append('N{}K1'.format(i + 1))

        spec_1.append('E{}'.format(i + 1))
        spec_1.append('N{}'.format(i + 1))

        spec_2.append('K1')
        spec_2.append('K1')

    spectra_list.append('K1K1')
    spec_1.append('K1')
    spec_2.append('K1')

    max_rows = None if lmax is None else (lmax - lmin + 1)

    spectra = []
    for spec_id in range(len(spectra_list)):
        spec_1_field = mysplit(spec_1[spec_id])[0]
        spec_1_zbin = mysplit(spec_1[spec_id])[1]

        spec_2_field = mysplit(spec_2[spec_id])[0]
        spec_2_zbin = mysplit(spec_2[spec_id])[1]

        if spec_1_field == 'N' and spec_2_field == 'N':
            spec = np.concatenate((np.zeros(lmin), cls_dict['galaxy_cl']['bin_{}_{}'.format(spec_2_zbin, spec_1_zbin)][0:max_rows]))
            spectra.append(spec)

        elif spec_1_field == 'E' and spec_2_field == 'E':
            spec = np.concatenate((np.zeros(lmin), cls_dict['shear_cl']['bin_{}_{}'.format(spec_2_zbin, spec_1_zbin)][0:max_rows]))
            spectra.append(spec)

        elif spec_1_field == 'N' and spec_2_field == 'E':
            spec = np.concatenate((np.zeros(lmin), cls_dict['galaxy_shear_cl']['bin_{}_{}'.format(spec_1_zbin, spec_2_zbin)][0:max_rows]))
            spectra.append(spec)

        elif spec_1_field == 'E' and spec_2_field == 'N':
            spec = np.concatenate((np.zeros(lmin), cls_dict['galaxy_shear_cl']['bin_{}_{}'.format(spec_2_zbin, spec_1_zbin)][0:max_rows]))
            spectra.append(spec)

        elif spec_1_field == 'E' and spec_2_field == 'K':
            spec = np.concatenate((np.zeros(lmin), cls_dict['shear_cmbkappa_cl']['bin_{}_{}'.format(spec_1_zbin, spec_2_zbin)][0:max_rows]))
            spectra.append(spec)

        elif spec_1_field == 'N' and spec_2_field == 'K':
            spec = np.concatenate((np.zeros(lmin), cls_dict['galaxy_cmbkappa_cl']['bin_{}_{}'.format(spec_1_zbin, spec_2_zbin)][0:max_rows]))
            spectra.append(spec)

        else:
            assert spec_1_field == 'K' and spec_2_field == 'K'
            spec = np.concatenate((np.zeros(lmin), cls_dict['cmbkappa_cl']['bin_{}_{}'.format(spec_1_zbin, spec_2_zbin)][0:max_rows]))
            spectra.append(spec)

    return spectra


def load_cls(n_zbin, cls_dir, lmax=None, lmin=0):
    """
    Given the number of redshift bins and relevant directories, load power spectra (position, shear, cross) in the
    correct order (diagonal / healpy new=True ordering).
    If lmin is supplied, the output will be padded to begin at l=0.

    Args:
        n_zbin (int): Number of redshift bins.
        cls_dict (dict): Dictionary of cls
        lmax (int, optional): Maximum l to load - if not supplied, will load all lines, which requires the individual
                              lmax of each file to be consistent.
        lmin (int, optional): Minimum l supplied. Output will be padded with zeros below this point.

    Returns:
        2D numpy array: All Cls, with different spectra along the first axis and increasing l along the second.
    """

    # Calculate number of fields assuming 1 position field and 1 shear field per redshift bin
    n_field = 2 * n_zbin

    # Form list of power spectra
    fields = [f'{f}{z}' for z in range(1, n_zbin + 1) for f in ['N', 'E']]
    # assert len(fields) == n_field

    spectra_list = [fields[row] + fields[row + diag] for diag in range(n_field) for row in range(n_field - diag)]

    spec_1 = [fields[row] for diag in range(n_field) for row in range(n_field - diag)]
    spec_2 = [fields[row + diag] for diag in range(n_field) for row in range(n_field - diag)]

    for i in range(n_zbin):
        spectra_list.append('E{}K1'.format(i + 1))
        spectra_list.append('N{}K1'.format(i + 1))

        spec_1.append('E{}'.format(i + 1))
        spec_1.append('N{}'.format(i + 1))

        spec_2.append('K1')
        spec_2.append('K1')

    spectra_list.append('K1K1')
    spec_1.append('K1')
    spec_2.append('K1')
    # print(spectra_list)
    max_rows = None if lmax is None else (lmax - lmin + 1)

    spectra = []
    for spec_id in range(len(spectra_list)):
        spec_1_field = mysplit(spec_1[spec_id])[0]
        spec_1_zbin = mysplit(spec_1[spec_id])[1]

        spec_2_field = mysplit(spec_2[spec_id])[0]
        spec_2_zbin = mysplit(spec_2[spec_id])[1]

        if spec_1_field == 'N' and spec_2_field == 'N':
            cl_path = os.path.join(cls_dir, f'galaxy_cl/bin_{spec_2_zbin}_{spec_1_zbin}.txt')
            spec = np.concatenate((np.zeros(lmin), np.loadtxt(cl_path, max_rows=max_rows)))
            spectra.append(spec)

        elif spec_1_field == 'E' and spec_2_field == 'E':
            cl_path = os.path.join(cls_dir, f'shear_cl/bin_{spec_2_zbin}_{spec_1_zbin}.txt')
            spec = np.concatenate((np.zeros(lmin), np.loadtxt(cl_path, max_rows=max_rows)))
            spectra.append(spec)

        elif spec_1_field == 'N' and spec_2_field == 'E':
            cl_path = os.path.join(cls_dir, f'galaxy_shear_cl/bin_{spec_1_zbin}_{spec_2_zbin}.txt')
            spec = np.concatenate((np.zeros(lmin), np.loadtxt(cl_path, max_rows=max_rows)))
            spectra.append(spec)

        elif spec_1_field == 'E' and spec_2_field == 'N':
            cl_path = os.path.join(cls_dir, f'galaxy_shear_cl/bin_{spec_2_zbin}_{spec_1_zbin}.txt')
            spec = np.concatenate((np.zeros(lmin), np.loadtxt(cl_path, max_rows=max_rows)))
            spectra.append(spec)

        elif spec_1_field == 'E' and spec_2_field == 'K':
            cl_path = os.path.join(cls_dir, f'shear_cmbkappa_cl/bin_{spec_1_zbin}_{spec_2_zbin}.txt')
            spec = np.concatenate((np.zeros(lmin), np.loadtxt(cl_path, max_rows=max_rows)))
            spectra.append(spec)

        elif spec_1_field == 'N' and spec_2_field == 'K':
            cl_path = os.path.join(cls_dir, f'galaxy_cmbkappa_cl/bin_{spec_1_zbin}_{spec_2_zbin}.txt')
            spec = np.concatenate((np.zeros(lmin), np.loadtxt(cl_path, max_rows=max_rows)))
            spectra.append(spec)

        else:
            assert spec_1_field == 'K' and spec_2_field == 'K'
            cl_path = os.path.join(cls_dir, f'cmbkappa_cl/bin_{spec_1_zbin}_{spec_2_zbin}.txt')
            spec = np.concatenate((np.zeros(lmin), np.loadtxt(cl_path, max_rows=max_rows)))
            spectra.append(spec)

    return spectra


def setup(mixmats, mix_lmin, input_lmin, input_lmax, n_zbin, n_bandpower):
    """
    Load and precompute everything that is fixed throughout parameter space. This should be called once per analysis,
    prior to any calls to execute.

    Args:
        obs_bp_path (str): Path to the observed bandpowers, in a numpy .npz file with array name obs_bp and shape
                           (n_spectra, n_bandpowers), with spectra in diagonal-major order.
        binmixmat_path (str): Path to combined mixing and binning matrices, in numpy .npz file with array names
                              (binmix_tt_to_tt, binmix_te_to_te, binmix_ee_to_ee, binmix_bb_to_ee), each with shape
                              (n_bandpower, input_lmax - mix_lmin + 1).
        mixmats (list): List containing the mixing matrices in order [nn-nn, ne-ne, ee-ee, bb-ee] for use in coupling
                        theory spectra
        mix_lmin (int): Minimum l for the theory power in the mixing matrices.
        cov_path (str): Path to precomputed covariance, in numpy .npz file with array name cov, with shape
                        (n_data, n_data) where n_data = n_spectra * n_bandpowers.
        pos_nl_path (str): Path to the unbinned position noise power spectrum, in text file.
        she_nl_path (str): Path to the unbinned shear noise power spectrum, in text file.
        noise_lmin (int): Minimum l in noise power spectra.
        input_lmax (int): Maximum l to include in mixing. Theory and noise power will be truncated above this.
        n_zbin (int): Number of redshift bins. It will be assumed that there is one position field and one shear field
                      per redshift bin.

    Returns:
        dict: Config dictionary to pass to execute.
    """

    # n_spec = (2 * n_zbin) * (2 * n_zbin + 1) // 2

    #Specify mixing matrices
    mixmat_nn_to_nn = mixmats[0]    # Dictionary of per bin mixmats
    mixmat_ne_to_ne = mixmats[1]    # Dictionary of per bin mixmats
    mixmat_ee_to_ee = mixmats[2]    # Dictionary of per bin mixmats
    mixmat_bb_to_ee = mixmats[3]    # Dictionary of per bin mixmats
    mixmat_kk_to_kk = mixmats[4]    # Single matrix
    mixmat_nn_to_kk = mixmats[5]    # Dictionary of per bin mixmats
    mixmat_ke_to_ke = mixmats[6]    # Dictionary of per bin mixmats

    n_cl = input_lmax - input_lmin + 1
    mix_lmax = mix_lmin + n_cl - 1

    # Generate a list of spectrum types (NN, EE or NE) in the correct (diagonal) order, so that we know which mixing
    # n_field = 2 * n_zbin

    # Form list of power spectra
    fields = [field for _ in range(n_zbin) for field in ('N', 'E')]
    n_field = len(fields)
    spectra = [fields[row] + fields[row + diag] for diag in range(n_field) for row in range(n_field - diag)]

    for i in range(n_zbin):
        spectra.append('EK')
        spectra.append('NK')

    spectra.append('KK')

    n_spec = len(spectra)

    fields_z = [z for z in range(1, n_zbin + 1) for f in ['N', 'E']]
    spectra_z_1 = [fields_z[row] for diag in range(2 * n_zbin) for row in range((2 * n_zbin) - diag)]
    spectra_z_2 = [fields_z[row + diag] for diag in range(2 * n_zbin) for row in range((2 * n_zbin) - diag)]

    for i in range(n_zbin):
        spectra_z_1.append(i+1)
        spectra_z_2.append(1)
        spectra_z_1.append(i+1)
        spectra_z_2.append(1)

    spectra_z_1.append(1)
    spectra_z_2.append(1)

    # Prepare config dictionary
    config = {
        'mix_lmin': mix_lmin,
        'mix_lmax': mix_lmax,
        'input_lmax': input_lmax,
        'n_spec': n_spec,
        'n_cl': n_cl,
        'n_zbin': n_zbin,
        'spectra': spectra,
        'spectra_z_1': spectra_z_1,
        'spectra_z_2': spectra_z_2,
        'n_bandpower': n_bandpower,
        'mixmat_nn_to_nn': mixmat_nn_to_nn,
        'mixmat_ne_to_ne': mixmat_ne_to_ne,
        'mixmat_ee_to_ee': mixmat_ee_to_ee,
        'mixmat_bb_to_ee': mixmat_bb_to_ee,
        'mixmat_kk_to_kk': mixmat_kk_to_kk,
        'mixmat_nn_to_kk': mixmat_nn_to_kk,
        'mixmat_ke_to_ke': mixmat_ke_to_ke
    }
    return config


def expected_bp(theory_cl, theory_lmin, config, noise_cls, pbl_nn, pbl_ne, pbl_ee, pbl_kk, pbl_ek, pbl_nk):
    """
    Calculate the joint log-likelihood at a particular point in parameter space.

    Args:
        theory_cl (2D numpy array): Theory power spectra, in diagonal ordering, with shape (n_spectra, n_ell).
        theory_lmin (int): Minimum l used in theory_cl.
        config (dict): Config dictionary returned by setup.

    Returns:
        float: Log-likelihood value.
    """

    # Unpack config dictionary
    mix_lmin = config['mix_lmin']
    mix_lmax = config['mix_lmax']
    input_lmax = config['input_lmax']
    n_spec = config['n_spec']
    n_cl = config['n_cl']
    n_zbin = config['n_zbin']
    spectra = config['spectra']
    spectra_z_1 = config['spectra_z_1']
    spectra_z_2 = config['spectra_z_2']
    n_bandpower = config['n_bandpower']
    mixmat_nn_to_nn = config['mixmat_nn_to_nn']
    mixmat_ne_to_ne = config['mixmat_ne_to_ne']
    mixmat_ee_to_ee = config['mixmat_ee_to_ee']
    mixmat_bb_to_ee = config['mixmat_bb_to_ee']
    mixmat_kk_to_kk = config['mixmat_kk_to_kk']
    mixmat_nn_to_kk = config['mixmat_nn_to_kk']
    mixmat_ke_to_ke = config['mixmat_ke_to_ke']

    # pbls are dictionaries indexed by bin number. Need to somehow map this onto the spectra ordering.

    # Trim/pad theory Cls to correct length for input to mixing matrices, truncating power above input_lmax:
    # 1. Trim so power is truncated above input_lmax
    theory_cl = np.asarray(theory_cl)
    theory_cl = theory_cl[:, :(input_lmax - theory_lmin + 1)]
    # 2. Pad so theory power runs from 0 up to max(input_lmax, mix_lmax)
    zeros_lowl = np.zeros((n_spec, theory_lmin))
    zeros_hil = np.zeros((n_spec, max(mix_lmax - input_lmax, 0)))
    theory_cl = np.concatenate((zeros_lowl, theory_cl, zeros_hil), axis=-1)
    # 3. Truncate so it runs from mix_lmin to mix_lmax
    theory_cl = theory_cl[:, mix_lmin:(mix_lmax + 1)]
    assert theory_cl.shape == (n_spec, n_cl), (theory_cl.shape, (n_spec, n_cl))

    exp_bp = np.full((n_spec, n_bandpower), np.nan)
    # Could have a list of spectra plus fields, i.e. N1N1, etc.
    for spec_idx, spec in enumerate(spectra):

        this_cl = theory_cl[spec_idx]
        this_noise_cl = noise_cls[spec_idx]
        # Need to trim noise cls here
        this_noise_cl = this_noise_cl[:(input_lmax - theory_lmin + 1)]
        this_noise_cl = np.concatenate((np.zeros(theory_lmin), this_noise_cl, np.zeros(max(mix_lmax - input_lmax, 0))),
                                       axis=0)
        this_noise_cl = this_noise_cl[mix_lmin:(mix_lmax + 1)]

        spec_z_1 = spectra_z_1[spec_idx]
        spec_z_2 = spectra_z_2[spec_idx]

        if spec == 'NN':

            pbl_nn_spec_1 = pbl_nn['Bin_{}'.format(spec_z_1)]
            pbl_nn_spec_2 = pbl_nn['Bin_{}'.format(spec_z_2)]
            assert pbl_nn_spec_1.shape[0] == pbl_nn_spec_2.shape[0]

            if pbl_nn_spec_1.shape[1] <= pbl_nn_spec_2.shape[1]:
                this_pbl_nn = pbl_nn_spec_1
                this_mixmat_nn_to_nn = mixmat_nn_to_nn['Bin_{}'.format(spec_z_1)]
            else:
                this_pbl_nn = pbl_nn_spec_2
                this_mixmat_nn_to_nn = mixmat_nn_to_nn['Bin_{}'.format(spec_z_2)]

            # print(this_pbl_nn)
            # print(this_pbl_nn.shape)
            this_exp_bp = this_pbl_nn@((this_mixmat_nn_to_nn@(this_cl+this_noise_cl)))

        # elif spec == 'NE':
        #
        #     pbl_ne_spec_1 = pbl_ne['Bin_{}'.format(spec_z_1)]
        #     pbl_ne_spec_2 = pbl_ne['Bin_{}'.format(spec_z_2)]
        #     assert pbl_ne_spec_1.shape[0] == pbl_ne_spec_2.shape[0]
        #
        #     if pbl_ne_spec_1.shape[1] <= pbl_ne_spec_2.shape[1]:
        #         this_pbl_ne = pbl_ne_spec_1
        #     else:
        #         this_pbl_ne = pbl_ne_spec_2
        #
        #     this_exp_bp = this_pbl_ne@((mixmat_ne_to_ne@(this_cl+this_noise_cl)))

        # elif spec == 'EN':
        #
        #     pbl_ne_spec_1 = pbl_ne['Bin_{}'.format(spec_z_2)]
        #     pbl_ne_spec_2 = pbl_ne['Bin_{}'.format(spec_z_1)]
        #     assert pbl_ne_spec_1.shape[0] == pbl_ne_spec_2.shape[0]
        #
        #     if pbl_ne_spec_1.shape[1] <= pbl_ne_spec_2.shape[1]:
        #         this_pbl_ne = pbl_ne_spec_1
        #     else:
        #         this_pbl_ne = pbl_ne_spec_2
        #
        #     this_exp_bp = this_pbl_ne@((mixmat_ne_to_ne@(this_cl+this_noise_cl)))

        elif spec in ('NE', 'EN'):
            if spec == 'NE':
                pbl_ne_spec_1 = pbl_ne['Bin_{}'.format(spec_z_1)]
                pbl_ne_spec_2 = pbl_ne['Bin_{}'.format(spec_z_2)]
                mixmat_spec_1 = mixmat_ne_to_ne['Bin_{}'.format(spec_z_1)]
                mixmat_spec_2 = mixmat_ne_to_ne['Bin_{}'.format(spec_z_2)]
            else:
                pbl_ne_spec_1 = pbl_ne['Bin_{}'.format(spec_z_2)]
                pbl_ne_spec_2 = pbl_ne['Bin_{}'.format(spec_z_1)]
                mixmat_spec_1 = mixmat_ne_to_ne['Bin_{}'.format(spec_z_2)]
                mixmat_spec_2 = mixmat_ne_to_ne['Bin_{}'.format(spec_z_1)]

            assert pbl_ne_spec_1.shape[0] == pbl_ne_spec_2.shape[0]

            if pbl_ne_spec_1.shape[1] <= pbl_ne_spec_2.shape[1]:
                this_pbl_ne = pbl_ne_spec_1
                this_mixmat_ne_to_ne = mixmat_spec_1

            else:
                this_pbl_ne = pbl_ne_spec_2
                this_mixmat_ne_to_ne = mixmat_spec_2

            # print(this_pbl_ne.shape)
            # print(this_mixmat_ne_to_ne.shape)
            # print(this_cl.shape)
            # print(this_noise_cl.shape)
            this_exp_bp = this_pbl_ne@((this_mixmat_ne_to_ne@(this_cl+this_noise_cl)))

        elif spec == 'EE':

            pbl_ee_spec_1 = pbl_ee['Bin_{}'.format(spec_z_1)]
            pbl_ee_spec_2 = pbl_ee['Bin_{}'.format(spec_z_2)]
            assert pbl_ee_spec_1.shape[0] == pbl_ee_spec_2.shape[0]

            if pbl_ee_spec_1.shape[1] <= pbl_ee_spec_2.shape[1]:
                this_pbl_ee = pbl_ee_spec_1
                this_mixmat_ee_to_ee = mixmat_ee_to_ee['Bin_{}'.format(spec_z_1)]
                this_mixmat_bb_to_ee = mixmat_bb_to_ee['Bin_{}'.format(spec_z_1)]

            else:
                this_pbl_ee = pbl_ee_spec_2
                this_mixmat_ee_to_ee = mixmat_ee_to_ee['Bin_{}'.format(spec_z_2)]
                this_mixmat_bb_to_ee = mixmat_bb_to_ee['Bin_{}'.format(spec_z_2)]

            this_exp_bp = this_pbl_ee@((this_mixmat_ee_to_ee@(this_cl+this_noise_cl))+(this_mixmat_bb_to_ee @ (this_noise_cl))) #Add BB noise contribution to auto-spectra - we don't consider this for JW work

        elif spec == 'EK':
            this_pbl_ek = pbl_ek['Bin_{}'.format(spec_z_1)]
            this_mixmat_ke_to_ke = mixmat_ke_to_ke['Bin_{}'.format(spec_z_1)]
            this_exp_bp = this_pbl_ek@((this_mixmat_ke_to_ke@(this_cl+this_noise_cl)))

        elif spec == 'NK':
            this_pbl_nk = pbl_nk['Bin_{}'.format(spec_z_1)]
            this_mixmat_nn_to_kk = mixmat_nn_to_kk['Bin_{}'.format(spec_z_1)]
            this_exp_bp = this_pbl_nk@((this_mixmat_nn_to_kk@(this_cl+this_noise_cl)))

        elif spec == 'KK':
            this_exp_bp = pbl_kk@((mixmat_kk_to_kk@(this_cl+this_noise_cl)))

        else:
            raise ValueError('Unexpected spectrum: ' + spec)
        exp_bp[spec_idx] = this_exp_bp

    return exp_bp

