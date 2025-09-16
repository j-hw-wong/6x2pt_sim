"""
Functions to do with masks and mixing matrices.
"""

import time
import sys
import numpy as np
import healpy as hp
import pymaster as nmt


def get_6x2pt_mixmats(mask_path, mask_path_cmb, nside, lmin, input_lmax, lmax_out_nn, lmax_out_ne, lmax_out_ee,
                      lmax_out_ek, lmax_out_nk, lmax_out_kk, save_path):
    """
    Calculate all 3x2pt mixing matrices from a mask using NaMaster, and save to disk in a single file.

    Args:
        mask_path (str): Path to mask FITS file. If None, full sky is assumed, in which case the mixing matrices should
                         be diagonal.
        nside (int): HEALPix resolution to use.
        lmin (int): Minimum l to include in mixing matrices.
        lmax_mix (int): Maximum l to include in input to mixing matrices.
        lmax_out (int): Maximum l to include in output from mixing matrices.
        save_path (str): Path to save output, as a single numpy .npz file containing all mixing matrices.
    """

    # Load and rescale mask, and calculate fsky
    if mask_path == 'None':
        print('Full sky')
        mask = np.ones(hp.pixelfunc.nside2npix(nside))
    else:
        print('Loading and rescaling mask')
        mask = hp.pixelfunc.ud_grade(hp.read_map(mask_path, dtype=float), nside)
        assert np.amin(mask) == 0
        assert np.amax(mask) == 1
    assert np.all(np.isfinite(mask))

    if mask_path_cmb == 'None':
        print('Full sky')
        mask_cmb = np.ones(hp.pixelfunc.nside2npix(nside))
    else:
        print('Loading and rescaling mask')
        mask_cmb = hp.pixelfunc.ud_grade(hp.read_map(mask_path_cmb, dtype=float), nside)
        assert np.amin(mask_cmb) == 0
        assert np.amax(mask_cmb) == 1
    assert np.all(np.isfinite(mask_cmb))

    # fsky = np.mean(mask)
    # print(f'fsky = {fsky:.3f}')

    # Create NaMaster binning scheme as individual Cls
    print('Creating binning scheme')
    bins = nmt.NmtBin.from_lmax_linear(input_lmax, 1)

    # Calculate mixing matrices for spin 0-0, 0-2 (equivalent to 2-0), and 2-2
    field_spin0 = nmt.NmtField(mask, None, spin=0, lite=True, lmax_sht=input_lmax)
    field_spin2 = nmt.NmtField(mask, None, spin=2, lite=True, lmax_sht=input_lmax)

    field_spin0_cmb = nmt.NmtField(mask_cmb, None, spin=0, lite=True, lmax_sht=input_lmax)

    workspace_spin00 = nmt.NmtWorkspace()
    print(f'Calculating mixing matrix 1 / 6 at {time.strftime("%c")}')
    workspace_spin00.compute_coupling_matrix(field_spin0, field_spin0, bins)
    workspace_spin02 = nmt.NmtWorkspace()
    print(f'Calculating mixing matrix 2 / 6 at {time.strftime("%c")}')
    workspace_spin02.compute_coupling_matrix(field_spin0, field_spin2, bins)
    workspace_spin22 = nmt.NmtWorkspace()
    print(f'Calculating mixing matrix 3 / 6 at {time.strftime("%c")}')
    workspace_spin22.compute_coupling_matrix(field_spin2, field_spin2, bins)

    workspace_spin00_cmb = nmt.NmtWorkspace()
    print(f'Calculating mixing matrix 4 / 6 at {time.strftime("%c")}')
    workspace_spin00_cmb.compute_coupling_matrix(field_spin0_cmb, field_spin0_cmb, bins)

    workspace_spin00_galaxy_cmb = nmt.NmtWorkspace()
    print(f'Calculating mixing matrix 5 / 6 at {time.strftime("%c")}')
    workspace_spin00_galaxy_cmb.compute_coupling_matrix(field_spin0, field_spin0_cmb, bins)

    workspace_spin02_shear_cmb = nmt.NmtWorkspace()
    print(f'Calculating mixing matrix 6 / 6 at {time.strftime("%c")}')
    workspace_spin02_shear_cmb.compute_coupling_matrix(field_spin0_cmb, field_spin2, bins)
    # workspace_spin22_cmb = nmt.NmtWorkspace()
    # print(f'Calculating mixing matrix 6 / 6 at {time.strftime("%c")}')
    # workspace_spin22_cmb.compute_coupling_matrix(field_spin2, field_spin2, bins)

    # Extract the relevant mixing matrices
    print('Extracting mixing matrices')
    # For 0-0 there is only a single mixing matrix
    mixmats_spin00 = workspace_spin00.get_coupling_matrix()
    mixmat_nn_to_nn = mixmats_spin00
    # For 0-2 they are arranged NE->NE, NB->NE // NE->NB NB->NB, per l, so select every other row and column
    mixmats_spin02 = workspace_spin02.get_coupling_matrix()
    mixmat_ne_to_ne = mixmats_spin02[::2, ::2]
    # For 2-2 there are 4x4 elements per l, ordered EE, EB, BE, BB. We only need EE->EE and BB->EE,
    # so select every 4th row and the 1st and 4th columns from each block
    mixmats_spin22 = workspace_spin22.get_coupling_matrix()
    mixmat_ee_to_ee = mixmats_spin22[::4, ::4]
    mixmat_bb_to_ee = mixmats_spin22[::4, 3::4]

    # For 0-0 there is only a single mixing matrix
    mixmats_spin00_cmb = workspace_spin00_cmb.get_coupling_matrix()
    mixmat_kk_to_kk = mixmats_spin00_cmb

    # For 0-0 there is only a single mixing matrix
    mixmats_spin00_galaxy_cmb = workspace_spin00_galaxy_cmb.get_coupling_matrix()
    mixmat_nn_to_kk = mixmats_spin00_galaxy_cmb

    # For 0-2 they are arranged NE->NE, NB->NE // NE->NB NB->NB, per l, so select every other row and column
    mixmats_spin02_shear_cmb = workspace_spin02_shear_cmb.get_coupling_matrix()
    mixmat_ke_to_ke = mixmats_spin02_shear_cmb[::2, ::2]

    # Check everything has the correct shape
    mixmat_shape = (input_lmax + 1, input_lmax + 1)
    assert mixmat_nn_to_nn.shape == mixmat_shape
    assert mixmat_ne_to_ne.shape == mixmat_shape
    assert mixmat_ee_to_ee.shape == mixmat_shape
    assert mixmat_bb_to_ee.shape == mixmat_shape
    assert mixmat_kk_to_kk.shape == mixmat_shape
    assert mixmat_nn_to_kk.shape == mixmat_shape
    assert mixmat_ke_to_ke.shape == mixmat_shape

    # Trim to required output range
    mixmat_nn_to_nn = mixmat_nn_to_nn[lmin:(lmax_out_nn + 1), lmin:(input_lmax+1)]
    mixmat_ne_to_ne = mixmat_ne_to_ne[lmin:(lmax_out_ne + 1), lmin:(input_lmax+1)]
    mixmat_ee_to_ee = mixmat_ee_to_ee[lmin:(lmax_out_ee + 1), lmin:(input_lmax+1)]
    mixmat_bb_to_ee = mixmat_bb_to_ee[lmin:(lmax_out_ee + 1), lmin:(input_lmax+1)]
    mixmat_kk_to_kk = mixmat_kk_to_kk[lmin:(lmax_out_kk + 1), lmin:(input_lmax+1)]
    mixmat_nn_to_kk = mixmat_nn_to_kk[lmin:(lmax_out_nk + 1), lmin:(input_lmax+1)]
    mixmat_ke_to_ke = mixmat_ke_to_ke[lmin:(lmax_out_ek + 1), lmin:(input_lmax+1)]

    # Do some final checks

    assert mixmat_nn_to_nn.shape == (lmax_out_nn - lmin + 1, input_lmax - lmin + 1)
    assert mixmat_ne_to_ne.shape == (lmax_out_ne - lmin + 1, input_lmax - lmin + 1)
    assert mixmat_ee_to_ee.shape == (lmax_out_ee - lmin + 1, input_lmax - lmin + 1)
    assert mixmat_bb_to_ee.shape == (lmax_out_ee - lmin + 1, input_lmax - lmin + 1)
    assert mixmat_kk_to_kk.shape == (lmax_out_kk - lmin + 1, input_lmax - lmin + 1)
    assert mixmat_nn_to_kk.shape == (lmax_out_nk - lmin + 1, input_lmax - lmin + 1)
    assert mixmat_ke_to_ke.shape == (lmax_out_ek - lmin + 1, input_lmax - lmin + 1)

    assert np.all(np.isfinite(mixmat_nn_to_nn))
    assert np.all(np.isfinite(mixmat_ne_to_ne))
    assert np.all(np.isfinite(mixmat_ee_to_ee))
    assert np.all(np.isfinite(mixmat_bb_to_ee))
    assert np.all(np.isfinite(mixmat_kk_to_kk))
    assert np.all(np.isfinite(mixmat_nn_to_kk))
    assert np.all(np.isfinite(mixmat_ke_to_ke))

    # Save to disk
    header = (f'Mixing matrices. Output from {__file__}.get_3x2pt_mixmats for mask_path = {mask_path}, '
              f'mask_path_cmb = {mask_path_cmb}, lmin = {lmin}, lmax_out_nn = {lmax_out_nn}, '
              f'lmax_out_ne = {lmax_out_ne}, lmax_out_ee = {lmax_out_ee}, lmax_out_kk = {lmax_out_kk},'
              f'lmax_out_nk = {lmax_out_nk}, lmax_out_ek = {lmax_out_ek} at {time.strftime("%c")}')

    np.savez_compressed(save_path, mixmat_nn_to_nn=mixmat_nn_to_nn, mixmat_ne_to_ne=mixmat_ne_to_ne,
                        mixmat_ee_to_ee=mixmat_ee_to_ee, mixmat_bb_to_ee=mixmat_bb_to_ee,
                        mixmat_kk_to_kk=mixmat_kk_to_kk, mixmat_nn_to_kk=mixmat_nn_to_kk,
                        mixmat_ke_to_ke=mixmat_ke_to_ke, header=header)
    print('Saved ' + save_path)


'''
def get_3x2pt_mixmats(mask_path, nside, lmin, input_lmax, lmax_out_nn, lmax_out_ne, lmax_out_ee, save_path):
    """
    Calculate all 3x2pt mixing matrices from a mask using NaMaster, and save to disk in a single file.

    Args:
        mask_path (str): Path to mask FITS file. If None, full sky is assumed, in which case the mixing matrices should
                         be diagonal.
        nside (int): HEALPix resolution to use.
        lmin (int): Minimum l to include in mixing matrices.
        lmax_mix (int): Maximum l to include in input to mixing matrices.
        lmax_out (int): Maximum l to include in output from mixing matrices.
        save_path (str): Path to save output, as a single numpy .npz file containing all mixing matrices.
    """

    # Load and rescale mask, and calculate fsky
    if mask_path is not None:
        print('Loading and rescaling mask')
        mask = hp.pixelfunc.ud_grade(hp.read_map(mask_path, dtype=float), nside)
        assert np.amin(mask) == 0
        assert np.amax(mask) == 1
    else:
        print('Full sky')
        mask = np.ones(hp.pixelfunc.nside2npix(nside))
    assert np.all(np.isfinite(mask))
    fsky = np.mean(mask)
    print(f'fsky = {fsky:.3f}')

    # Create NaMaster binning scheme as individual Cls
    print('Creating binning scheme')
    bins = nmt.NmtBin.from_lmax_linear(input_lmax, 1)

    # Calculate mixing matrices for spin 0-0, 0-2 (equivalent to 2-0), and 2-2
    field_spin0 = nmt.NmtField(mask, None, spin=0, lite=True, lmax_sht=input_lmax)
    field_spin2 = nmt.NmtField(mask, None, spin=2, lite=True, lmax_sht=input_lmax)
    # field_spin0 = nmt.NmtField(mask, None, spin=0, lite=True)
    # field_spin2 = nmt.NmtField(mask, None, spin=2, lite=True)
    workspace_spin00 = nmt.NmtWorkspace()
    print(f'Calculating mixing matrix 1 / 3 at {time.strftime("%c")}')
    workspace_spin00.compute_coupling_matrix(field_spin0, field_spin0, bins)
    workspace_spin02 = nmt.NmtWorkspace()
    print(f'Calculating mixing matrix 2 / 3 at {time.strftime("%c")}')
    workspace_spin02.compute_coupling_matrix(field_spin0, field_spin2, bins)
    workspace_spin22 = nmt.NmtWorkspace()
    print(f'Calculating mixing matrix 3 / 3 at {time.strftime("%c")}')
    workspace_spin22.compute_coupling_matrix(field_spin2, field_spin2, bins)

    # Extract the relevant mixing matrices
    print('Extracting mixing matrices')
    # For 0-0 there is only a single mixing matrix
    mixmats_spin00 = workspace_spin00.get_coupling_matrix()
    mixmat_nn_to_nn = mixmats_spin00
    #print(mixmat_nn_to_nn)
    #print(mixmat_nn_to_nn.shape)
    # For 0-2 they are arranged NE->NE, NB->NE // NE->NB NB->NB, per l, so select every other row and column
    mixmats_spin02 = workspace_spin02.get_coupling_matrix()
    mixmat_ne_to_ne = mixmats_spin02[::2, ::2]
    # For 2-2 there are 4x4 elements per l, ordered EE, EB, BE, BB. We only need EE->EE and BB->EE,
    # so select every 4th row and the 1st and 4th columns from each block
    mixmats_spin22 = workspace_spin22.get_coupling_matrix()
    mixmat_ee_to_ee = mixmats_spin22[::4, ::4]
    mixmat_bb_to_ee = mixmats_spin22[::4, 3::4]

    # Check everything has the correct shape
    mixmat_shape = (input_lmax + 1, input_lmax + 1)
    assert mixmat_nn_to_nn.shape == mixmat_shape
    assert mixmat_ne_to_ne.shape == mixmat_shape
    assert mixmat_ee_to_ee.shape == mixmat_shape
    assert mixmat_bb_to_ee.shape == mixmat_shape

    # Trim to required output range
    mixmat_nn_to_nn = mixmat_nn_to_nn[lmin:(lmax_out_nn + 1), lmin:(input_lmax+1)]
    mixmat_ne_to_ne = mixmat_ne_to_ne[lmin:(lmax_out_ne + 1), lmin:(input_lmax+1)]
    mixmat_ee_to_ee = mixmat_ee_to_ee[lmin:(lmax_out_ee + 1), lmin:(input_lmax+1)]
    mixmat_bb_to_ee = mixmat_bb_to_ee[lmin:(lmax_out_ee + 1), lmin:(input_lmax+1)]
    # Do some final checks

    assert mixmat_nn_to_nn.shape == (lmax_out_nn - lmin + 1, input_lmax - lmin + 1)
    assert mixmat_ne_to_ne.shape == (lmax_out_ne - lmin + 1, input_lmax - lmin + 1)
    assert mixmat_ee_to_ee.shape == (lmax_out_ee - lmin + 1, input_lmax - lmin + 1)
    assert mixmat_bb_to_ee.shape == (lmax_out_ee - lmin + 1, input_lmax - lmin + 1)
    assert np.all(np.isfinite(mixmat_nn_to_nn))
    assert np.all(np.isfinite(mixmat_ne_to_ne))
    assert np.all(np.isfinite(mixmat_ee_to_ee))
    assert np.all(np.isfinite(mixmat_bb_to_ee))

    # Save to disk
    header = (f'Mixing matrices. Output from {__file__}.get_3x2pt_mixmats for mask_path = {mask_path}, '
              f'nside = {nside}, lmin = {lmin}, lmax_out_nn = {lmax_out_nn}, lmax_out_ne = {lmax_out_ne}, '
              f'lmax_out_ee = {lmax_out_ee} at {time.strftime("%c")}')
    np.savez_compressed(save_path, mixmat_nn_to_nn=mixmat_nn_to_nn, mixmat_ne_to_ne=mixmat_ne_to_ne,
                        mixmat_ee_to_ee=mixmat_ee_to_ee, mixmat_bb_to_ee=mixmat_bb_to_ee, header=header)
    print('Saved ' + save_path)
'''

def get_binning_matrix(n_bandpowers, output_lmin, output_lmax, input_lmin=None, input_lmax=None,
                       bp_spacing='lin', bp_edges=None):
    """
    Returns the binning matrix to convert Cls to log-spaced bandpowers, following Eqn. 20 of Hivon et al. 2002.

    Input ell range defaults to match output ell range - note this behaviour is not suitable if this matrix is to be
    used to multiply the raw output from healpy anafast/alm2cl, which returns all ells from l=0. In that case,
    explicitly set input_lmin=0.

    Args:
        n_bandpowers (int): Number of bandpowers required.
        output_lmin (int): Minimum l to include in the binning.
        output_lmax (int): Maximum l to include in the binning.
        input_lmin (int, optional): Minimum l in the input (defaults to output_lmin).
        input_lmax (int, optional): Maximum l in the input (defaults to output_lmax).
        bp_spacing (str, optional): Type of bandpower spacing to use. Default is 'log' for logspaced, can also be 'lin'
                                    for linear spacing or 'custom' if user wants to specify a custom binning
        bp_edges (arr, optional): Used if bp_spacing is 'custom'. Bandpower boundaries in l-space. Must be specified as
                                  [bp_lmin, bp_lmax], i.e. includes both the lower boundary of lowest band and upper
                                  boundary of highest band.

    Returns:
        2D numpy array: Binning matrix P_bl, shape (n_bandpowers, n_input_ell),
                        with n_input_ell = input_lmax - input_lmin + 1.
    """

    # Calculate bin boundaries (add small fraction to lmax to include it in the end bin)
    # Might be a good idea to change this into an argument for the function, i.e.
    # type of bandpower separation (log-space, lin-space or some other custom) can be set
    # by the user

    accepted_bp_spacings = {'log', 'lin', 'custom'}
    if bp_spacing not in accepted_bp_spacings:
        print('Error! Bandpower Spacing Not Recognised - Exiting...')
        sys.exit()
    elif bp_spacing == 'log':
        edges = np.logspace(np.log10(output_lmin), np.log10(output_lmax + 1e-5), n_bandpowers + 1)
    elif bp_spacing == 'lin':
        edges = np.linspace(output_lmin, output_lmax + 1e-5, n_bandpowers + 1)
    else:
        assert bp_spacing == 'custom'
        if bp_edges is None:
            print('WARNING - Must supply custom bandpower edges')
            sys.exit()

        assert len(bp_edges) == n_bandpowers + 1
        assert int(min(bp_edges)) == int(output_lmin)
        assert int(max(bp_edges)) == int(output_lmax)
        edges = bp_edges

    # edges = np.logspace(np.log10(output_lmin), np.log10(output_lmax + 1e-5), n_bandpowers + 1)

    # Calculate input ell range and create broadcasted views for convenience
    if input_lmin is None:
        input_lmin = output_lmin
    if input_lmax is None:
        input_lmax = output_lmax
    ell = np.arange(input_lmin, input_lmax + 1)[None, ...]

    lower_edges = edges[:-1, None]
    upper_edges = edges[1:, None]
    # print(edges)
    # First calculate a boolean matrix of whether each ell is included in each bandpower,
    # then apply the l(l+1)/2π / n_l factor where n_l is the number of ells in the bin
    in_bin = (ell >= lower_edges) & (ell < upper_edges)
    # print(np.floor(upper_edges))
    # print(np.ceil(lower_edges))
    n_ell = np.floor(upper_edges) - np.ceil(lower_edges) + 1
    # print(n_ell)
    pbl = in_bin * ell * (ell + 1) / (2 * np.pi * n_ell)
    # print(pbl)
    return pbl

