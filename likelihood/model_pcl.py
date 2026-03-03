import os.path
import numpy as np
import likelihood.mask as mask
import likelihood.like_bp_gauss_mix_6x2pt as like_bp_mix_6x2pt
import likelihood.like_bp_gauss_mix_3x2pt as like_bp_mix_3x2pt
import likelihood.like_bp_gauss_mix_1x2pt as like_bp_mix_1x2pt


def PCl_bandpowers_6x2pt(cls_dict, n_bp, n_zbin, lmax_like_galaxy, lmin_like_galaxy, lmax_like_galaxy_shear,
                         lmin_like_galaxy_shear, lmax_like_shear, lmin_like_shear, lmax_like_galaxy_kCMB,
                         lmin_like_galaxy_kCMB, lmax_like_shear_kCMB, lmin_like_shear_kCMB, lmax_like_kCMB,
                         lmin_like_kCMB, lmax_in, lmin_in, noise_path, mixmats, bandpower_spacing='log'):

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

    n_field = 2 * n_zbin
    # n_spec = n_field * (n_field + 1) // 2

    # Calculate some useful quantities
    n_ell_in = lmax_in - lmin_in + 1
    ell_in = np.arange(lmin_in, lmax_in + 1)

    # Form list of power spectra
    # fields = [f'{f}{z}' for z in range(1, n_zbin + 1) for f in ['N', 'E']]
    # # assert len(fields) == n_field
    #
    # spectra = [fields[row] + fields[row + diag] for diag in range(n_field) for row in range(n_field - diag)]
    #
    # for i in range(n_zbin):
    #     spectra.append('E{}K1'.format(i+1))
    #     spectra.append('N{}K1'.format(i+1))
    #
    # spectra.append('K1K1')

    # Load mixing matrices
    # with np.load(mixmats_path) as data:
    mixmat_nn_to_nn = mixmats['mixmat_nn_to_nn'][lmin_like_galaxy - lmin_in:, :]
    mixmat_ne_to_ne = mixmats['mixmat_ne_to_ne'][lmin_like_galaxy_shear - lmin_in:, :]
    mixmat_ee_to_ee = mixmats['mixmat_ee_to_ee'][lmin_like_shear - lmin_in:, :]
    mixmat_bb_to_ee = mixmats['mixmat_bb_to_ee'][lmin_like_shear - lmin_in:, :]
    mixmat_kk_to_kk = mixmats['mixmat_kk_to_kk'][lmin_like_kCMB - lmin_in:, :]
    mixmat_nn_to_kk = mixmats['mixmat_nn_to_kk'][lmin_like_galaxy_kCMB - lmin_in:, :]
    mixmat_ke_to_ke = mixmats['mixmat_ke_to_ke'][lmin_like_shear_kCMB - lmin_in:, :]

    mixmat_shape_nn = (lmax_like_galaxy - lmin_like_galaxy + 1, n_ell_in)
    mixmat_shape_ne = (lmax_like_galaxy_shear - lmin_like_galaxy_shear + 1, n_ell_in)
    mixmat_shape_ee = (lmax_like_shear - lmin_like_shear + 1, n_ell_in)
    mixmat_shape_kk = (lmax_like_kCMB - lmin_like_kCMB + 1, n_ell_in)
    mixmat_shape_nnkk = (lmax_like_galaxy_kCMB - lmin_like_galaxy_kCMB + 1, n_ell_in)
    mixmat_shape_ke = (lmax_like_shear_kCMB - lmin_like_shear_kCMB + 1, n_ell_in)

    assert mixmat_nn_to_nn.shape == mixmat_shape_nn, (mixmat_nn_to_nn.shape, mixmat_shape_nn)
    assert mixmat_ne_to_ne.shape == mixmat_shape_ne, (mixmat_ne_to_ne.shape, mixmat_shape_ne)
    assert mixmat_ee_to_ee.shape == mixmat_shape_ee, (mixmat_ee_to_ee.shape, mixmat_shape_ee)
    assert mixmat_bb_to_ee.shape == mixmat_shape_ee, (mixmat_bb_to_ee.shape, mixmat_shape_ee)
    assert mixmat_kk_to_kk.shape == mixmat_shape_kk, (mixmat_kk_to_kk.shape, mixmat_shape_kk)
    assert mixmat_nn_to_kk.shape == mixmat_shape_nnkk, (mixmat_nn_to_kk.shape, mixmat_shape_nnkk)
    assert mixmat_ke_to_ke.shape == mixmat_shape_ke, (mixmat_ke_to_ke.shape, mixmat_shape_ke)

    assert bandpower_spacing == 'log' or bandpower_spacing == 'lin'

    pbl_shear = mask.get_binning_matrix(
        n_bandpowers=n_bp,
        output_lmin=lmin_like_shear,
        output_lmax=lmax_like_shear,
        bp_spacing=bandpower_spacing)

    pbl_galaxy_shear = mask.get_binning_matrix(
        n_bandpowers=n_bp,
        output_lmin=lmin_like_galaxy_shear,
        output_lmax=lmax_like_galaxy_shear,
        bp_spacing=bandpower_spacing)

    pbl_galaxy = mask.get_binning_matrix(
        n_bandpowers=n_bp,
        output_lmin=lmin_like_galaxy,
        output_lmax=lmax_like_galaxy,
        bp_spacing=bandpower_spacing)

    pbl_cmbkk = mask.get_binning_matrix(
        n_bandpowers=n_bp,
        output_lmin=lmin_like_kCMB,
        output_lmax=lmax_like_kCMB,
        bp_spacing=bandpower_spacing)

    pbl_cmbkk_galaxy = mask.get_binning_matrix(
        n_bandpowers=n_bp,
        output_lmin=lmin_like_galaxy_kCMB,
        output_lmax=lmax_like_galaxy_kCMB,
        bp_spacing=bandpower_spacing)

    pbl_cmbkk_shear = mask.get_binning_matrix(
        n_bandpowers=n_bp,
        output_lmin=lmin_like_shear_kCMB,
        output_lmax=lmax_like_shear_kCMB,
        bp_spacing=bandpower_spacing)

    if pbl_shear.ndim == 1:
        pbl_shear = pbl_shear[np.newaxis, :]

    if pbl_galaxy_shear.ndim == 1:
        pbl_galaxy_shear = pbl_galaxy_shear[np.newaxis, :]

    if pbl_galaxy.ndim == 1:
        pbl_galaxy = pbl_galaxy[np.newaxis, :]

    if pbl_cmbkk.ndim == 1:
        pbl_cmbkk = pbl_cmbkk[np.newaxis, :]

    if pbl_cmbkk_galaxy.ndim == 1:
        pbl_cmbkk_galaxy = pbl_cmbkk_galaxy[np.newaxis, :]

    if pbl_cmbkk_shear.ndim == 1:
        pbl_cmbkk_shear = pbl_cmbkk_shear[np.newaxis, :]

    assert pbl_shear.shape == (n_bp, lmax_like_shear - lmin_like_shear + 1)
    assert pbl_galaxy_shear.shape == (n_bp, lmax_like_galaxy_shear - lmin_like_galaxy_shear + 1)
    assert pbl_galaxy.shape == (n_bp, lmax_like_galaxy - lmin_like_galaxy + 1)
    assert pbl_cmbkk_shear.shape == (n_bp, lmax_like_shear_kCMB - lmin_like_shear_kCMB + 1)
    assert pbl_cmbkk_galaxy.shape == (n_bp, lmax_like_galaxy_kCMB - lmin_like_galaxy_kCMB + 1)
    assert pbl_cmbkk.shape == (n_bp, lmax_like_kCMB - lmin_like_kCMB + 1)

    config = like_bp_mix_6x2pt.setup(
        mixmats=[mixmat_nn_to_nn, mixmat_ne_to_ne, mixmat_ee_to_ee, mixmat_bb_to_ee, mixmat_kk_to_kk, mixmat_nn_to_kk,
                 mixmat_ke_to_ke],
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
    galaxy_cmbkappa_ell = cls_dict['galaxy_cmbkappa_cl']['ell']
    shear_cmbkappa_ell = cls_dict['shear_cmbkappa_cl']['ell']
    cmbkappa_ell = cls_dict['cmbkappa_cl']['ell']

    assert np.array_equal(galaxy_ell, ell_in)
    assert np.array_equal(shear_ell, ell_in)
    assert np.array_equal(galaxy_shear_ell, ell_in)
    assert np.array_equal(galaxy_cmbkappa_ell, ell_in)
    assert np.array_equal(shear_cmbkappa_ell, ell_in)
    assert np.array_equal(cmbkappa_ell, ell_in)
    #
    # noise_pos_pos_dir = os.path.join(noise_path, 'galaxy_cl/')
    # noise_she_she_dir = os.path.join(noise_path, 'shear_cl/')
    # noise_pos_she_dir = os.path.join(noise_path, 'galaxy_shear_cl/')
    # noise_kCMB_kCMB_dir = os.path.join(noise_path, 'cmbkappa_cl/')
    # noise_she_kCMB_dir = os.path.join(noise_path, 'shear_cmbkappa_cl/')
    # noise_pos_kCMB_dir = os.path.join(noise_path, 'galaxy_cmbkappa_cl/')

    theory_cl = like_bp_mix_6x2pt.load_cls_dict(n_zbin, cls_dict, lmax=lmax_in)
    noise_cls = like_bp_mix_6x2pt.load_cls(n_zbin, noise_path, lmax=lmax_in)

    exp_bps = like_bp_mix_6x2pt.expected_bp(np.asarray(theory_cl), lmin_in, config, noise_cls=noise_cls,
                                            pbl_nn=pbl_galaxy, pbl_ne=pbl_galaxy_shear, pbl_ee=pbl_shear,
                                            pbl_kk=pbl_cmbkk, pbl_ek=pbl_cmbkk_shear, pbl_nk=pbl_cmbkk_galaxy)

    return exp_bps


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

    config = like_bp_mix_3x2pt.setup(
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

    theory_cl = like_bp_mix_3x2pt.load_cls_dict(n_zbin, cls_dict, lmax=lmax_in)
    noise_cls = like_bp_mix_3x2pt.load_cls(n_zbin, noise_pos_pos_dir, noise_she_she_dir, noise_pos_she_dir, lmax=lmax_in)

    exp_bps = like_bp_mix_3x2pt.expected_bp(np.asarray(theory_cl), lmin_in, config, noise_cls=noise_cls, pbl_nn=pbl_nn, pbl_ne=pbl_ne, pbl_ee=pbl_ee)

    return exp_bps


def PCl_bandpowers_1x2pt(cls_dict, n_bp, n_zbin, lmax_like_galaxy, lmin_like_galaxy, lmax_like_galaxy_shear,
                         lmin_like_galaxy_shear, lmax_like_shear, lmin_like_shear, lmax_like_galaxy_kCMB,
                         lmin_like_galaxy_kCMB, lmax_like_shear_kCMB, lmin_like_shear_kCMB, lmax_like_kCMB,
                         lmin_like_kCMB, lmax_in, lmin_in, field, noise_path, mixmats, bandpower_spacing='log'):

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
    # n_ell_like = lmax_like - lmin_like + 1
    n_ell_in = lmax_in - lmin_in + 1
    ell_in = np.arange(lmin_in, lmax_in + 1)

    # # Form list of power spectra
    # if field == 'E':
    #     fields = [f'E{z}' for z in range(1, n_zbin + 1)]
    #     assert len(fields) == n_field
    #     spectra = [fields[row] + fields[row + diag] for diag in range(n_field) for row in range(n_field - diag)]
    #     assert len(spectra) == n_spec
    #
    # elif field == 'N':
    #     fields = [f'N{z}' for z in range(1, n_zbin + 1)]
    #     assert len(fields) == n_field
    #     spectra = [fields[row] + fields[row + diag] for diag in range(n_field) for row in range(n_field - diag)]
    #     assert len(spectra) == n_spec
    #
    # elif field == 'EK':
    #     pass
    #
    # elif field == 'NK':
    #     pass
    #
    # else:
    #     assert field == 'K'

    mixmat_nn_to_nn = mixmats['mixmat_nn_to_nn'][lmin_like_galaxy - lmin_in:, :]
    mixmat_ne_to_ne = mixmats['mixmat_ne_to_ne'][lmin_like_galaxy_shear - lmin_in:, :]
    mixmat_ee_to_ee = mixmats['mixmat_ee_to_ee'][lmin_like_shear - lmin_in:, :]
    mixmat_bb_to_ee = mixmats['mixmat_bb_to_ee'][lmin_like_shear - lmin_in:, :]
    mixmat_kk_to_kk = mixmats['mixmat_kk_to_kk'][lmin_like_kCMB - lmin_in:, :]
    mixmat_nn_to_kk = mixmats['mixmat_nn_to_kk'][lmin_like_galaxy_kCMB - lmin_in:, :]
    mixmat_ke_to_ke = mixmats['mixmat_ke_to_ke'][lmin_like_shear_kCMB - lmin_in:, :]

    mixmat_shape_nn = (lmax_like_galaxy - lmin_like_galaxy + 1, n_ell_in)
    mixmat_shape_ne = (lmax_like_galaxy_shear - lmin_like_galaxy_shear + 1, n_ell_in)
    mixmat_shape_ee = (lmax_like_shear - lmin_like_shear + 1, n_ell_in)
    mixmat_shape_kk = (lmax_like_kCMB - lmin_like_kCMB + 1, n_ell_in)
    mixmat_shape_nnkk = (lmax_like_galaxy_kCMB - lmin_like_galaxy_kCMB + 1, n_ell_in)
    mixmat_shape_ke = (lmax_like_shear_kCMB - lmin_like_shear_kCMB + 1, n_ell_in)

    assert mixmat_nn_to_nn.shape == mixmat_shape_nn, (mixmat_nn_to_nn.shape, mixmat_shape_nn)
    assert mixmat_ne_to_ne.shape == mixmat_shape_ne, (mixmat_ne_to_ne.shape, mixmat_shape_ne)
    assert mixmat_ee_to_ee.shape == mixmat_shape_ee, (mixmat_ee_to_ee.shape, mixmat_shape_ee)
    assert mixmat_bb_to_ee.shape == mixmat_shape_ee, (mixmat_bb_to_ee.shape, mixmat_shape_ee)
    assert mixmat_kk_to_kk.shape == mixmat_shape_kk, (mixmat_kk_to_kk.shape, mixmat_shape_kk)
    assert mixmat_nn_to_kk.shape == mixmat_shape_nnkk, (mixmat_nn_to_kk.shape, mixmat_shape_nnkk)
    assert mixmat_ke_to_ke.shape == mixmat_shape_ke, (mixmat_ke_to_ke.shape, mixmat_shape_ke)

    assert bandpower_spacing == 'log' or bandpower_spacing == 'lin'

    pbl_shear = mask.get_binning_matrix(
        n_bandpowers=n_bp,
        output_lmin=lmin_like_shear,
        output_lmax=lmax_like_shear,
        bp_spacing=bandpower_spacing)

    pbl_galaxy_shear = mask.get_binning_matrix(
        n_bandpowers=n_bp,
        output_lmin=lmin_like_galaxy_shear,
        output_lmax=lmax_like_galaxy_shear,
        bp_spacing=bandpower_spacing)

    pbl_galaxy = mask.get_binning_matrix(
        n_bandpowers=n_bp,
        output_lmin=lmin_like_galaxy,
        output_lmax=lmax_like_galaxy,
        bp_spacing=bandpower_spacing)

    pbl_cmbkk = mask.get_binning_matrix(
        n_bandpowers=n_bp,
        output_lmin=lmin_like_kCMB,
        output_lmax=lmax_like_kCMB,
        bp_spacing=bandpower_spacing)

    pbl_cmbkk_galaxy = mask.get_binning_matrix(
        n_bandpowers=n_bp,
        output_lmin=lmin_like_galaxy_kCMB,
        output_lmax=lmax_like_galaxy_kCMB,
        bp_spacing=bandpower_spacing)

    pbl_cmbkk_shear = mask.get_binning_matrix(
        n_bandpowers=n_bp,
        output_lmin=lmin_like_shear_kCMB,
        output_lmax=lmax_like_shear_kCMB,
        bp_spacing=bandpower_spacing)

    if pbl_shear.ndim == 1:
        pbl_shear = pbl_shear[np.newaxis, :]

    if pbl_galaxy_shear.ndim == 1:
        pbl_galaxy_shear = pbl_galaxy_shear[np.newaxis, :]

    if pbl_galaxy.ndim == 1:
        pbl_galaxy = pbl_galaxy[np.newaxis, :]

    if pbl_cmbkk.ndim == 1:
        pbl_cmbkk = pbl_cmbkk[np.newaxis, :]

    if pbl_cmbkk_galaxy.ndim == 1:
        pbl_cmbkk_galaxy = pbl_cmbkk_galaxy[np.newaxis, :]

    if pbl_cmbkk_shear.ndim == 1:
        pbl_cmbkk_shear = pbl_cmbkk_shear[np.newaxis, :]

    assert pbl_shear.shape == (n_bp, lmax_like_shear - lmin_like_shear + 1)
    assert pbl_galaxy_shear.shape == (n_bp, lmax_like_galaxy_shear - lmin_like_galaxy_shear + 1)
    assert pbl_galaxy.shape == (n_bp, lmax_like_galaxy - lmin_like_galaxy + 1)
    assert pbl_cmbkk_shear.shape == (n_bp, lmax_like_shear_kCMB - lmin_like_shear_kCMB + 1)
    assert pbl_cmbkk_galaxy.shape == (n_bp, lmax_like_galaxy_kCMB - lmin_like_galaxy_kCMB + 1)
    assert pbl_cmbkk.shape == (n_bp, lmax_like_kCMB - lmin_like_kCMB + 1)


    '''
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
    '''
    config = like_bp_mix_1x2pt.setup(
        mixmats=[mixmat_nn_to_nn, mixmat_ne_to_ne, mixmat_ee_to_ee, mixmat_bb_to_ee, mixmat_kk_to_kk, mixmat_nn_to_kk,
                 mixmat_ke_to_ke],
        field=field,
        mix_lmin=lmin_in,
        input_lmin=lmin_in,
        input_lmax=lmax_in,
        n_zbin=n_zbin,
        n_bandpower=n_bp)

    galaxy_ell = cls_dict['galaxy_cl']['ell']
    shear_ell = cls_dict['shear_cl']['ell']
    galaxy_cmbkappa_ell = cls_dict['galaxy_cmbkappa_cl']['ell']
    shear_cmbkappa_ell = cls_dict['shear_cmbkappa_cl']['ell']
    cmbkappa_ell = cls_dict['cmbkappa_cl']['ell']

    assert np.array_equal(galaxy_ell, ell_in)
    assert np.array_equal(shear_ell, ell_in)
    assert np.array_equal(galaxy_cmbkappa_ell, ell_in)
    assert np.array_equal(shear_cmbkappa_ell, ell_in)
    assert np.array_equal(cmbkappa_ell, ell_in)

    # noise_pos_pos_dir = os.path.join(noise_path, 'galaxy_cl/')
    # noise_she_she_dir = os.path.join(noise_path, 'shear_cl/')
    # noise_pos_kCMB_dir = os.path.join(noise_path, 'galaxy_cmbkappa_cl/')
    # noise_she_kCMB_dir = os.path.join(noise_path, 'shear_cmbkappa_cl/')
    # noise_kCMB_kCMB_dir = os.path.join(noise_path, 'cmbkappa_cl/')

    theory_cl = like_bp_mix_1x2pt.load_cls_dict(n_zbin, field, cls_dict, lmax=lmax_in)
    noise_cls = like_bp_mix_1x2pt.load_cls(n_zbin, field, noise_path, lmax=lmax_in)

    # if field == 'E':
    #     noise_cls = like_bp_mix_1x2pt.load_cls(n_zbin, field, noise_path, lmax=lmax_in)
    #
    # elif field == 'N':
    #     noise_cls = like_bp_mix_1x2pt.load_cls(n_zbin, noise_pos_pos_dir, lmax=lmax_in)
    #
    # elif field == 'EK':
    #     noise_cls = like_bp_mix_1x2pt.load_cls(n_zbin, noise_she_kCMB_dir, lmax=lmax_in)
    #
    # elif field == 'NK':
    #     noise_cls = like_bp_mix_1x2pt.load_cls(n_zbin, noise_pos_kCMB_dir, lmax=lmax_in)
    #
    # else:
    #     assert field == 'K'
    #     noise_cls = like_bp_mix_1x2pt.load_cls(n_zbin, noise_kCMB_kCMB_dir, lmax=lmax_in)

    exp_bps = like_bp_mix_1x2pt.expected_bp(np.asarray(theory_cl), lmin_in, config, noise_cls=noise_cls,
                                            pbl_nn=pbl_galaxy, pbl_ne=pbl_galaxy_shear, pbl_ee=pbl_shear,
                                            pbl_kk=pbl_cmbkk, pbl_ek=pbl_cmbkk_shear, pbl_nk=pbl_cmbkk_galaxy)

    return exp_bps


