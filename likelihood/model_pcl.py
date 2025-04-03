import os.path
import numpy as np
import likelihood.mask as mask
import likelihood.like_bp_gauss_mix as like_bp_mix
import likelihood.like_bp_gauss_mix_1x2pt as like_bp_mix_1x2pt


def PCl_bandpowers_3x2pt(cls_dict, n_bp, n_zbin, lmax_like_nn, lmin_like_nn, lmax_like_ne, lmin_like_ne,
                                lmax_like_ee, lmin_like_ee, lmax_in, lmin_in, noise_path, mixmats,
                                bandpower_spacing='log',
                                ):

    """
    Run the like_bp_gauss_mix likelihood module over a CosmoSIS grid repeatedly for different numbers of bandpowers,
    saving a separate likelihood file for each number of bandpowers. This function assumes you are inputting some
    observed set of Pseudo-Cl Bandpowers

    Args:
        grid_dir (str): Path to CosmoSIS grid.
        n_bps (list): List of numbers of bandpowers.
        n_zbin (int): Number of redshift bins.
        lmax_like (int): Maximum l to use in the likelihood.
        lmin_like (int): Minimum l to use in the likelihood.
        lmax_in (int): Maximum l included in mixing.
        lmin_in (int): Minimum l supplied in theory and noise power spectra.
        noise_path (str): Path to directory containing noise power spectra for each of gal, shear, gal_shear Cls.
        mixmats_path (str): Path to mixing matrices in numpy .npz file with four arrays (mixmat_nn_to_nn,
                            mixmat_ne_to_ne, mixmat_ee_to_ee, mixmat_bb_to_ee) each with shape
                            (lmax_like - lmin_in + 1, lmax_in - lmin_in + 1).
        bandpower_spacing (str, optional): Method to divide bandpowers in ell-space. Must be one of 'log' (for log-
                                           spaced bandpowers); 'lin' (for linearly spaced bandpowers); or 'custom' for
                                           a user specified bandpower spacing. Default is 'log'. If 'custom', the
                                           bandpower bin-boundaries must be specified in the bandpower_edges argument.
    """

    # Calculate some useful quantities
    n_field = 2 * n_zbin
    n_spec = n_field * (n_field + 1) // 2
    n_ell_in = lmax_in - lmin_in + 1
    ell_in = np.arange(lmin_in, lmax_in + 1)

    # Form list of power spectra
    fields = [f'{f}{z}' for z in range(1, n_zbin + 1) for f in ['N', 'E']]

    assert len(fields) == n_field
    spectra = [fields[row] + fields[row + diag] for diag in range(n_field) for row in range(n_field - diag)]
    assert len(spectra) == n_spec

    # Load mixing matrices
    # with np.load(mixmats_path) as data:
    mixmat_nn_to_nn = mixmats['mixmat_nn_to_nn'][lmin_like_nn - lmin_in:, :]
    mixmat_ne_to_ne = mixmats['mixmat_ne_to_ne'][lmin_like_ne - lmin_in:, :]
    mixmat_ee_to_ee = mixmats['mixmat_ee_to_ee'][lmin_like_ee - lmin_in:, :]
    mixmat_bb_to_ee = mixmats['mixmat_bb_to_ee'][lmin_like_ee - lmin_in:, :]

    mixmat_shape_nn = (lmax_like_nn - lmin_like_nn + 1, n_ell_in)
    mixmat_shape_ne = (lmax_like_ne - lmin_like_ne + 1, n_ell_in)
    mixmat_shape_ee = (lmax_like_ee - lmin_like_ee + 1, n_ell_in)

    assert mixmat_nn_to_nn.shape == mixmat_shape_nn, (mixmat_nn_to_nn.shape, mixmat_shape_nn)
    assert mixmat_ne_to_ne.shape == mixmat_shape_ne, (mixmat_ne_to_ne.shape, mixmat_shape_ne)
    assert mixmat_ee_to_ee.shape == mixmat_shape_ee, (mixmat_ee_to_ee.shape, mixmat_shape_ee)
    assert mixmat_bb_to_ee.shape == mixmat_shape_ee, (mixmat_bb_to_ee.shape, mixmat_shape_ee)

    assert bandpower_spacing == 'log' or bandpower_spacing == 'lin'

    pbl_ee = mask.get_binning_matrix(
        n_bandpowers=n_bp,
        output_lmin=lmin_like_ee,
        output_lmax=lmax_like_ee,
        bp_spacing=bandpower_spacing)

    pbl_ne = mask.get_binning_matrix(
        n_bandpowers=n_bp,
        output_lmin=lmin_like_ne,
        output_lmax=lmax_like_ne,
        bp_spacing=bandpower_spacing)

    pbl_nn = mask.get_binning_matrix(
        n_bandpowers=n_bp,
        output_lmin=lmin_like_nn,
        output_lmax=lmax_like_nn,
        bp_spacing=bandpower_spacing)

    if pbl_nn.ndim == 1:
        pbl_nn = pbl_nn[np.newaxis, :]

    if pbl_ne.ndim == 1:
        pbl_ne = pbl_ne[np.newaxis, :]

    if pbl_ee.ndim == 1:
        pbl_ee = pbl_ee[np.newaxis, :]

    assert pbl_nn.shape == (n_bp, lmax_like_nn - lmin_like_nn + 1)
    assert pbl_ne.shape == (n_bp, lmax_like_ne - lmin_like_ne + 1)
    assert pbl_ee.shape == (n_bp, lmax_like_ee - lmin_like_ee + 1)

    config = like_bp_mix.setup(
        mixmats=[mixmat_nn_to_nn, mixmat_ne_to_ne, mixmat_ee_to_ee, mixmat_bb_to_ee],
        mix_lmin=lmin_in,
        input_lmin=lmin_in,
        input_lmax=lmax_in,
        n_zbin=n_zbin,
        n_bandpower=n_bp)

    # Check the ells for consistency
    # source_dir = save_dir + 'inference_chains/'

    galaxy_ell = cls_dict['galaxy_cl']['ell']
    shear_ell = cls_dict['shear_cl']['ell']
    galaxy_shear_ell = cls_dict['galaxy_shear_cl']['ell']

    assert np.array_equal(galaxy_ell, ell_in)
    assert np.array_equal(shear_ell, ell_in)
    assert np.array_equal(galaxy_shear_ell, ell_in)

    noise_pos_pos_dir = os.path.join(noise_path, 'galaxy_cl/')
    noise_she_she_dir = os.path.join(noise_path, 'shear_cl/')
    noise_pos_she_dir = os.path.join(noise_path, 'galaxy_shear_cl/')

    theory_cl = like_bp_mix.load_cls_dict(n_zbin, cls_dict, lmax=lmax_in)
    noise_cls = like_bp_mix.load_cls(n_zbin, noise_pos_pos_dir, noise_she_she_dir, noise_pos_she_dir, lmax=lmax_in)

    exp_bps = like_bp_mix.expected_bp(np.asarray(theory_cl), lmin_in, config, noise_cls=noise_cls, pbl_nn=pbl_nn, pbl_ne=pbl_ne, pbl_ee=pbl_ee)

    return exp_bps


def PCl_bandpowers_1x2pt(cls_dict, n_bp, n_zbin, lmax_like, lmin_like, lmax_in, lmin_in,
                                field, noise_path, mixmats, bandpower_spacing='log'):

    """
    Run the like_bp_gauss_mix likelihood module over a CosmoSIS grid repeatedly for different numbers of bandpowers,
    saving a separate likelihood file for each number of bandpowers. This function assumes you are inputting some
    observed set of Pseudo-Cl Bandpowers

    Args:
        grid_dir (str): Path to CosmoSIS grid.
        n_bps (list): List of numbers of bandpowers.
        n_zbin (int): Number of redshift bins.
        lmax_like (int): Maximum l to use in the likelihood.
        lmin_like (int): Minimum l to use in the likelihood.
        lmax_in (int): Maximum l included in mixing.
        lmin_in (int): Minimum l supplied in theory and noise power spectra.
        field (str): Field to use for 1x2pt analysis. Must be 'E' or 'N' for cosmic shear/angular galaxy clustering
        noise_path (str): Path to directory containing noise power spectra for each of gal, shear, gal_shear Cls.
        mixmats_path (str): Path to mixing matrices in numpy .npz file with four arrays (mixmat_nn_to_nn,
                            mixmat_ne_to_ne, mixmat_ee_to_ee, mixmat_bb_to_ee) each with shape
                            (lmax_like - lmin_in + 1, lmax_in - lmin_in + 1).
        bandpower_spacing (str, optional): Method to divide bandpowers in ell-space. Must be one of 'log' (for log-
                                           spaced bandpowers); 'lin' (for linearly spaced bandpowers); or 'custom' for
                                           a user specified bandpower spacing. Default is 'log'. If 'custom', the
                                           bandpower bin-boundaries must be specified in the bandpower_edges argument.
    """

    # Calculate some useful quantities
    n_field = n_zbin
    n_spec = n_field * (n_field + 1) // 2
    n_ell_like = lmax_like - lmin_like + 1
    n_ell_in = lmax_in - lmin_in + 1
    ell_in = np.arange(lmin_in, lmax_in + 1)

    # Form list of power spectra
    if field == 'E':
        fields = [f'E{z}' for z in range(1, n_zbin + 1)]
    else:
        assert field == 'N'
        fields = [f'N{z}' for z in range(1, n_zbin + 1)]

    assert len(fields) == n_field
    spectra = [fields[row] + fields[row + diag] for diag in range(n_field) for row in range(n_field - diag)]
    assert len(spectra) == n_spec

    # Load mixing matrices
    lowl_skip = lmin_like - lmin_in
    # with np.load(mixmats_path) as data:
    mixmat_nn_to_nn = mixmats['mixmat_nn_to_nn'][lowl_skip:, :]
    mixmat_ne_to_ne = mixmats['mixmat_ne_to_ne'][lowl_skip:, :]
    mixmat_ee_to_ee = mixmats['mixmat_ee_to_ee'][lowl_skip:, :]
    mixmat_bb_to_ee = mixmats['mixmat_bb_to_ee'][lowl_skip:, :]
    mixmat_shape = (n_ell_like, n_ell_in)

    assert mixmat_nn_to_nn.shape == mixmat_shape, (mixmat_nn_to_nn.shape, mixmat_shape)
    assert mixmat_ne_to_ne.shape == mixmat_shape, (mixmat_ne_to_ne.shape, mixmat_shape)
    assert mixmat_ee_to_ee.shape == mixmat_shape, (mixmat_ee_to_ee.shape, mixmat_shape)
    assert mixmat_bb_to_ee.shape == mixmat_shape, (mixmat_bb_to_ee.shape, mixmat_shape)

    assert bandpower_spacing == 'log' or bandpower_spacing == 'lin'
    pbl = mask.get_binning_matrix(n_bp, lmin_like, lmax_like, bp_spacing=bandpower_spacing)

    if pbl.ndim == 1:
        pbl = pbl[np.newaxis, :]
    assert pbl.shape == (n_bp, n_ell_like)

    config = like_bp_mix_1x2pt.setup(
        mixmats=[mixmat_nn_to_nn, mixmat_ee_to_ee, mixmat_bb_to_ee],
        field=field,
        mix_lmin=lmin_in,
        input_lmin=lmin_in,
        input_lmax=lmax_in,
        n_zbin=n_zbin,
        n_bandpower=n_bp)

    galaxy_ell = cls_dict['galaxy_cl']['ell']
    shear_ell = cls_dict['shear_cl']['ell']

    assert np.array_equal(galaxy_ell, ell_in)
    assert np.array_equal(shear_ell, ell_in)

    noise_pos_pos_dir = os.path.join(noise_path, 'galaxy_cl/')
    noise_she_she_dir = os.path.join(noise_path, 'shear_cl/')

    if field == 'E':
        theory_cl = like_bp_mix_1x2pt.load_cls_dict(n_zbin, cls_dict, field, lmax=lmax_in)
        noise_cls = like_bp_mix_1x2pt.load_cls(n_zbin, noise_she_she_dir, lmax=lmax_in)

    else:
        assert field == 'N'
        theory_cl = like_bp_mix_1x2pt.load_cls_dict(n_zbin, cls_dict, field, lmax=lmax_in)
        noise_cls = like_bp_mix_1x2pt.load_cls(n_zbin, noise_pos_pos_dir, lmax=lmax_in)

    exp_bps = like_bp_mix_1x2pt.expected_bp(np.asarray(theory_cl), lmin_in, config, noise_cls=noise_cls, pbl=pbl)

    return exp_bps


