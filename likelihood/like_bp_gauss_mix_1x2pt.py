"""
Likelihood module to evaluate the joint likelihood of a set of tomographic 3x2pt power spectra, binned into bandpowers,
on the cut sky using a multivariate Gaussian likelihood.

The main functions are setup, which should be called once per analysis, and execute, which is called for every new
point in parameter space.
"""

import os.path
import numpy as np


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


def load_cls_dict(n_zbin, field, cls_dict, lmax=None, lmin=0):
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

    if field == 'E':
        fields = [f'E{z}' for z in range(1, n_zbin+1)]
        n_field = len(fields)
        spectra_list = [fields[row] + fields[row + diag] for diag in range(n_field) for row in range(n_field - diag)]
        spec_1 = [fields[row] for diag in range(n_field) for row in range(n_field - diag)]
        spec_2 = [fields[row + diag] for diag in range(n_field) for row in range(n_field - diag)]

    elif field == 'N':
        fields = [f'N{z}' for z in range(1, n_zbin + 1)]
        n_field = len(fields)
        spectra_list = [fields[row] + fields[row + diag] for diag in range(n_field) for row in range(n_field - diag)]
        spec_1 = [fields[row] for diag in range(n_field) for row in range(n_field - diag)]
        spec_2 = [fields[row + diag] for diag in range(n_field) for row in range(n_field - diag)]

    elif field == 'EK':
        fields = [f'E{z}K1' for z in range(1, n_zbin+1)]
        spectra_list = fields
        spec_1 = [f'E{z}' for z in range(1, n_zbin+1)]
        spec_2 = ['K1'] * n_zbin

    elif field == 'NK':
        fields = [f'N{z}K1' for z in range(1, n_zbin+1)]
        spectra_list = fields
        spec_1 = [f'N{z}' for z in range(1, n_zbin+1)]
        spec_2 = ['K1'] * n_zbin

    else:
        assert field == 'K'
        fields = ['K1K1']
        spectra_list = fields
        spec_1 = ['K1']
        spec_2 = ['K1']

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

        # GGL on its own isn't currently considered but this could be useful to keep if considered in the future
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

    return np.asarray(spectra)


def load_cls(n_zbin, field, cls_dir, lmax=None, lmin=0):
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

    if field == 'E':
        fields = [f'E{z}' for z in range(1, n_zbin+1)]
        n_field = len(fields)
        spectra_list = [fields[row] + fields[row + diag] for diag in range(n_field) for row in range(n_field - diag)]
        spec_1 = [fields[row] for diag in range(n_field) for row in range(n_field - diag)]
        spec_2 = [fields[row + diag] for diag in range(n_field) for row in range(n_field - diag)]

    elif field == 'N':
        fields = [f'N{z}' for z in range(1, n_zbin + 1)]
        n_field = len(fields)
        spectra_list = [fields[row] + fields[row + diag] for diag in range(n_field) for row in range(n_field - diag)]
        spec_1 = [fields[row] for diag in range(n_field) for row in range(n_field - diag)]
        spec_2 = [fields[row + diag] for diag in range(n_field) for row in range(n_field - diag)]

    elif field == 'EK':
        fields = [f'E{z}K1' for z in range(1, n_zbin+1)]
        spectra_list = fields
        spec_1 = [f'E{z}' for z in range(1, n_zbin+1)]
        spec_2 = ['K1'] * n_zbin

    elif field == 'NK':
        fields = [f'N{z}K1' for z in range(1, n_zbin+1)]
        spectra_list = fields
        spec_1 = [f'N{z}' for z in range(1, n_zbin+1)]
        spec_2 = ['K1'] * n_zbin

    else:
        assert field == 'K'
        fields = ['K1K1']
        spectra_list = fields
        spec_1 = ['K1']
        spec_2 = ['K1']

    max_rows = None if lmax is None else (lmax - lmin + 1)
    # print(spectra_list)
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

    return np.asarray(spectra)



'''
def load_cls_dict(n_zbin, cls_dict, field, lmax=None, lmin=0):
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

    # Load power spectra in diagonal-major order
    spectra = []
    for diag in range(n_field):
        for row in range(n_field - diag):
            col = row + diag

            # Extract the bins: for pos-pos and she-she the higher bin index goes first, for pos-she pos goes first
            bins = (row // 2 + 1, col // 2 + 1)

            if is_odd(row) and is_odd(col):
                bin1 = max(bins)
                bin2 = min(bins)

                # Load with appropriate ell range
                max_rows = None if lmax is None else (lmax - lmin + 1)
                if field == 'E':
                    spec = np.concatenate((np.zeros(lmin), cls_dict['shear_cl']['bin_{}_{}'.format(bin1, bin2)][0:max_rows]))
                else:
                    assert field == 'N'
                    spec = np.concatenate((np.zeros(lmin), cls_dict['galaxy_cl']['bin_{}_{}'.format(bin1, bin2)][0:max_rows]))
                spectra.append(spec)

    return spectra


def load_cls(n_zbin, cl_dir, lmax=None, lmin=0):
    """
    Given the number of redshift bins and relevant directories, load power spectra (position, shear, cross) in the
    correct order (diagonal / healpy new=True ordering).
    If lmin is supplied, the output will be padded to begin at l=0.

    Args:
        n_zbin (int): Number of redshift bins.
        she_she_dir (str): Path to directory containing shear-shear power spectra.
        pos_she_dir (str): Path to directory containing position-shear power spectra.
        lmax (int, optional): Maximum l to load - if not supplied, will load all lines, which requires the individual
                              lmax of each file to be consistent.
        lmin (int, optional): Minimum l supplied. Output will be padded with zeros below this point.

    Returns:
        2D numpy array: All Cls, with different spectra along the first axis and increasing l along the second.
    """

    # Calculate number of fields assuming 1 position field and 1 shear field per redshift bin
    n_field = 2 * n_zbin

    # Load power spectra in diagonal-major order
    spectra = []
    for diag in range(n_field):
        for row in range(n_field - diag):
            col = row + diag
            if is_odd(row) and is_odd(col):

                # Extract the bins: for pos-pos and she-she the higher bin index goes first, for pos-she pos goes first
                bins = (row // 2 + 1, col // 2 + 1)
                bin1 = max(bins)
                bin2 = min(bins)

                cl_path = os.path.join(cl_dir, f'bin_{bin1}_{bin2}.txt')

                # Load with appropriate ell range
                max_rows = None if lmax is None else (lmax - lmin + 1)
                spec = np.concatenate((np.zeros(lmin), np.loadtxt(cl_path, max_rows=max_rows)))
                spectra.append(spec)
    # print(spectra)
    return np.asarray(spectra)
'''

def setup(mixmats, field, mix_lmin, input_lmin, input_lmax, n_zbin, n_bandpower):
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


    # Specify mixing matrices
    mixmat_nn_to_nn = mixmats[0]
    mixmat_ne_to_ne = mixmats[1]
    mixmat_ee_to_ee = mixmats[2]
    mixmat_bb_to_ee = mixmats[3]
    mixmat_kk_to_kk = mixmats[4]
    mixmat_nn_to_kk = mixmats[5]
    mixmat_ke_to_ke = mixmats[6]

    # Could e.g assert mixmat shape == binmix shape

    n_cl = input_lmax - input_lmin + 1
    mix_lmax = mix_lmin + n_cl - 1

    # Generate a list of spectrum types (NN, EE or NE) in the correct (diagonal) order, so that we know which mixing
    # matrix/matrices to apply
    if field == 'E':
        fields = [field for _ in range(n_zbin) for field in ('E')]
        n_field = len(fields)
        spectra = [fields[row] + fields[row + diag] for diag in range(n_field) for row in range(n_field - diag)]
        n_spec = (n_zbin) * (n_zbin + 1) // 2

    elif field == 'N':
        fields = [field for _ in range(n_zbin) for field in ('N')]
        n_field = len(fields)
        spectra = [fields[row] + fields[row + diag] for diag in range(n_field) for row in range(n_field - diag)]
        n_spec = (n_zbin) * (n_zbin + 1) // 2

    elif field == 'EK':
        fields = ['EK'] * n_zbin
        spectra = fields
        n_spec = n_zbin

    elif field == 'NK':
        fields = ['NK'] * n_zbin
        spectra = fields
        n_spec = n_zbin

    else:
        assert field == 'K'
        fields = ['K']
        spectra = fields
        n_spec = 1

    assert len(spectra) == n_spec
    n_data = n_spec * n_bandpower

    # Prepare config dictionary
    config = {
        'mix_lmin': mix_lmin,
        'mix_lmax': mix_lmax,
        'input_lmax': input_lmax,
        'n_spec': n_spec,
        'n_cl': n_cl,
        'n_zbin': n_zbin,
        'field': field,
        'spectra': spectra,
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
    field = config['field']
    n_cl = config['n_cl']
    n_zbin = config['n_zbin']
    spectra = config['spectra']
    n_bandpower = config['n_bandpower']
    mixmat_nn_to_nn = config['mixmat_nn_to_nn']
    mixmat_ne_to_ne = config['mixmat_ne_to_ne']
    mixmat_ee_to_ee = config['mixmat_ee_to_ee']
    mixmat_bb_to_ee = config['mixmat_bb_to_ee']
    mixmat_kk_to_kk = config['mixmat_kk_to_kk']
    mixmat_nn_to_kk = config['mixmat_nn_to_kk']
    mixmat_ke_to_ke = config['mixmat_ke_to_ke']

    # Trim/pad theory Cls to correct length for input to mixing matrices, truncating power above input_lmax:
    # 1. Trim so power is truncated above input_lmax
    theory_cl = theory_cl[:, :(input_lmax - theory_lmin + 1)]
    # 2. Pad so theory power runs from 0 up to max(input_lmax, mix_lmax)
    zeros_lowl = np.zeros((n_spec, theory_lmin))
    zeros_hil = np.zeros((n_spec, max(mix_lmax - input_lmax, 0)))
    theory_cl = np.concatenate((zeros_lowl, theory_cl, zeros_hil), axis=-1)
    # 3. Truncate so it runs from mix_lmin to mix_lmax
    theory_cl = theory_cl[:, mix_lmin:(mix_lmax + 1)]
    assert theory_cl.shape == (n_spec, n_cl), (theory_cl.shape, (n_spec, n_cl))

    # Now trim/pad noise Cls as above
    # 1. Trim so power is truncated above input_lmax
    noise_cls = noise_cls[:, :(input_lmax - theory_lmin + 1)]
    # print(len(noise_cls[0]))
    # 2. Pad so theory power runs from 0 up to max(input_lmax, mix_lmax)
    zeros_lowl = np.zeros((n_spec, theory_lmin))
    zeros_hil = np.zeros((n_spec, max(mix_lmax - input_lmax, 0)))
    noise_cls = np.concatenate((zeros_lowl, noise_cls, zeros_hil), axis=-1)
    # 3. Truncate so it runs from mix_lmin to mix_lmax
    noise_cls = noise_cls[:, mix_lmin:(mix_lmax + 1)]
    # assert noise_cls.shape == (n_spec, n_cl), (noise_cls.shape, (n_spec, n_cl))

    exp_bp = np.full((n_spec, n_bandpower), np.nan)
    for spec_idx, spec in enumerate(spectra):
        if field == 'E':
            assert spec == 'EE'
            this_cl = theory_cl[spec_idx]
            this_noise_cl = noise_cls[spec_idx]
            this_exp_bp = pbl_ee @ ((mixmat_ee_to_ee @ (this_cl + this_noise_cl))+(mixmat_bb_to_ee @ (this_noise_cl))) #Need to convolve mixing from noise component in B-mode. B-mode itself is zero
            exp_bp[spec_idx] = this_exp_bp

        elif field == 'N':
            assert spec == 'NN'
            this_cl = theory_cl[spec_idx]
            this_noise_cl = noise_cls[spec_idx]
            this_exp_bp = pbl_nn @ ((mixmat_nn_to_nn @ (this_cl + this_noise_cl)))
            exp_bp[spec_idx] = this_exp_bp

        elif field == 'EK':
            assert spec == 'EK'
            this_cl = theory_cl[spec_idx]
            this_noise_cl = noise_cls[spec_idx]
            this_exp_bp = pbl_ek @ ((mixmat_ke_to_ke @ (this_cl + this_noise_cl)))
            exp_bp[spec_idx] = this_exp_bp

        elif field == 'NK':
            assert spec == 'NK'
            this_cl = theory_cl[spec_idx]
            this_noise_cl = noise_cls[spec_idx]
            this_exp_bp = pbl_nk @ ((mixmat_nn_to_kk @ (this_cl + this_noise_cl)))
            exp_bp[spec_idx] = this_exp_bp

        elif field == 'K':
            assert spec == 'K'
            this_cl = theory_cl[spec_idx]
            this_noise_cl = noise_cls[spec_idx]
            this_exp_bp = pbl_kk @ ((mixmat_kk_to_kk @ (this_cl + this_noise_cl)))
            exp_bp[spec_idx] = this_exp_bp

    assert np.all(np.isfinite(exp_bp))

    return exp_bp
