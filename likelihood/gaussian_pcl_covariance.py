"""
Functions to calculate 3x2pt Gaussian covariance.
"""

import sys
import time
import warnings

import likelihood.mask as mask
import math as maths
import healpy as hp
import numpy as np
import pymaster as nmt

warnings.filterwarnings('error')  # terminate on warning


def spin_from_field(field):

    if field == 'N':
        spin = 0
    elif field == 'E':
        spin = 2
    elif field == 'K':
        spin = 0
    else:
        print('Unknown field type')
        sys.exit()
    return spin


def spectrum_from_fields_1x2pt(field_a, zbin_a, field_b, zbin_b):

    if field_a == 'K' and (field_b == 'E' or field_b == 'N'):
        spec = f"{field_b}{zbin_b}{field_a}{zbin_a}"

    elif field_b == 'K' and (field_a == 'E' or field_a == 'N'):
        spec = f"{field_a}{zbin_a}{field_b}{zbin_b}"

    else:
        if zbin_a < zbin_b:
            spec = f"{field_a}{zbin_a}{field_b}{zbin_b}"

        else:
            spec = f"{field_b}{zbin_b}{field_a}{zbin_a}"

    return spec


def spectrum_from_fields_3x2pt(field_a, zbin_a, field_b, zbin_b):

    if zbin_a < zbin_b or (zbin_a == zbin_b and field_a == 'N'):
        spec = f"{field_a}{zbin_a}{field_b}{zbin_b}"
    else:
        spec = f"{field_b}{zbin_b}{field_a}{zbin_a}"

    return spec


def spectrum_from_fields_6x2pt(field_a, zbin_a, field_b, zbin_b):

    if field_a == 'K' and (field_b == 'E' or field_b == 'N'):
        spec = f"{field_b}{zbin_b}{field_a}{zbin_a}"

    elif field_b == 'K' and (field_a == 'E' or field_a == 'N'):
        spec = f"{field_a}{zbin_a}{field_b}{zbin_b}"

    else:

        if zbin_a < zbin_b or (zbin_a == zbin_b and field_a == 'N'):
            spec = f"{field_a}{zbin_a}{field_b}{zbin_b}"
        else:
            spec = f"{field_b}{zbin_b}{field_a}{zbin_a}"

        # if zbin_a < zbin_b:
        #     spec = f"{field_a}{zbin_a}{field_b}{zbin_b}"
        #
        # else:
        #     spec = f"{field_b}{zbin_b}{field_a}{zbin_a}"

    return spec


def workspace_from_spins(spin_a, spin_b, workspace_spin00, workspace_spin02, workspace_spin22):
    """
    Returns the appropriate NmtWorkspace object for the two supplied spins.

    Args:
        spin_a (int): Spin of one field.
        spin_b (int): Spin of the other field.
        workspace_spin00 (NmtWorkspace): Workspace for two spin-0 fields.
        workspace_spin02 (NmtWorkspace): Workspace for one spin-0 and one spin-2 field (in either order).
        workspace_spin22 (NmtWorkspace): Workspace for two spin-2 fields.

    Returns:
        NmtWorkspace: The workspace object corresponding to the two supplied spins.
    """

    spins = (spin_a, spin_b)
    if spins == (0, 0):
        return workspace_spin00
    if spins in [(0, 2), (2, 0)]:
        return workspace_spin02
    if spins == (2, 2):
        return workspace_spin22
    raise ValueError(f'Unexpected combination of spins {spins}')


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


def load_cls(signal_paths, noise_paths, lmax_in, lmax_out, signal_lmin, noise_lmin):
    """
    Load a list of Cls with appropriate noise, given lists of path to signal and noise Cls. If any entry in either list
    is None, then zeros are used. Signal and noise are padded with zeros below ``signal_lmin`` and ``noise_lmin`` and
    above ``lmax_in`` (if less than ``lmax_out``).

    Args:
        signal_paths (list): List of paths to signal Cls. If any entry is None, it is taken to be zero.
        noise_paths (list): List of paths to noise Cls. If any entry is None, it is taken to be zero.
        lmax_in (int): Maximum l to load. If this is less than ``lmax_out`` then padded with zeros above this l.
        lmax_out (int): Maximum l to return.
        signal_lmin (int): First l of signal power spectra. Signal Cls will be padded with zeros below this.
        noise_lmin (int): First l of noise power spectra. Noise Cls will be padded with zeros below this. If None,
        assumes no noise spectra to be combined with signal (e.g. if calculating signal and noise covariance separately)
        ^NOTE: Might need to write a guard for this...

    Returns:
        list: List of numpy arrays, each of which is a signal + noise power spectrum in the supplied order.
    """

    if noise_lmin is None:
        assert all(n is None for n in noise_paths)

    # If a signal or noise path is None then just use zeros
    zero_cl = np.zeros(lmax_out + 1)
    zero_pad = np.zeros(lmax_out - lmax_in) if lmax_out > lmax_in else []

    if lmax_in > lmax_out:
        lmax_in = lmax_out

    # Load Cls with appropriate padding and add signal and noise
    combined_cls = []
    for signal_path, noise_path in zip(signal_paths, noise_paths):

        signal_cl = (np.concatenate((np.zeros(signal_lmin),
                                     np.loadtxt(signal_path, max_rows=(lmax_in - signal_lmin + 1)), zero_pad))
                     if signal_path else zero_cl)
        noise_cl = (np.concatenate((np.zeros(noise_lmin),
                                    np.loadtxt(noise_path, max_rows=(lmax_in - noise_lmin + 1)), zero_pad))
                    if noise_path else zero_cl)
        combined_cls.append(signal_cl + noise_cl)

    return combined_cls


def get_1x2pt_cov_pcl(n_zbin, signal_path, noise_path, field, lmax_in, lmin_in, lmax_out, lmin_out, noise_lmin, mask_path,
                      nside, save_block_filemask, n_bp, bandpower_spacing, save_cov_filemask):
    """
    Calculate 3x2pt Gaussian covariance using NaMaster, saving each block separately to disk. This function computes
    the covariance associated with a Pseudo-Cl in the absence of noise - i.e. we apply the coupling coefficients of
    a mask to a theoretical (full-sky) Cl (using the coupled=True argument in NaMaster gaussian_covariance). The
    complete covariance for a noisy Pseudo-Cl observation can be calculated by combining this function with
    get_3x2pt_cov_noise - which calculates the Gaussian covariance of a predicted noise spectrum. NOTE - this
    theoretical noise spectrum must predict the noise associated with the region of sky observed (unlike the full-sky
    Cl) since it is NOT combined with the mask coupling coefficients.

    Args:
        n_zbin (int): Number of redshift bins, assuming 1 position field and 1 shear field per redshift bin.
        pos_pos_filemask (str): Path to text file containing a position-position power spectrum with ``{hi_zbin}`` and
                                ``{lo_zbin}`` placeholders.
        pos_she_filemask (str): Path to text file containing a position-shear power spectrum with ``{pos_zbin}`` and
                                ``{she_zbin}`` placeholders.
        she_she_filemask (str): Path to text file containing a shear-shear power spectrum with ``{hi_zbin}`` and
                                ``{lo_zbin}`` placeholders.
        lmax_in (int): Maximum l to including in mixing.
        lmin_in (int): Minimum l in input power spectra.
        lmax_out (int): Maximum l to include in covariance.
        lmin_out (int): Minimum l to include in covariance.
        noise_lmin (int): Minimum l in noise power spectra.
        mask_path (str): Path to mask FITS file. If None, full sky is assumed, in which case the covariance will be
                         diagonal.
        nside (int): HEALPix resolution nside parameter.
        save_filemask (str): Path to save each covariance block to disk, with ``{spec1_idx}`` and ``{spec2_idx}``
                             placeholders.
    """

    # Load and rescale mask, and calculate fsky
    if mask_path is not None:
        print('Loading and rescaling mask')
        obs_mask = hp.pixelfunc.ud_grade(hp.read_map(mask_path, dtype=float), nside)
        assert np.amin(obs_mask) == 0
        assert np.amax(obs_mask) == 1
    else:
        print('Full sky')
        obs_mask = np.ones(hp.pixelfunc.nside2npix(nside))
    assert np.all(np.isfinite(obs_mask))
    fsky = np.mean(obs_mask)
    print(f'fsky = {fsky:.3f}')

    # Create NaMaster binning scheme as individual Cls
    print('Creating binning scheme')
    bins = nmt.NmtBin.from_lmax_linear(lmax_in, 1)

    # Calculate mixing matrices for spin 0-0, 0-2 (equivalent to 2-0), and 2-2
    field_spin0 = nmt.NmtField(obs_mask, None, spin=0, lite=True, lmax_sht=lmax_in)
    field_spin2 = nmt.NmtField(obs_mask, None, spin=2, lite=True, lmax_sht=lmax_in)

    workspace_spin00 = nmt.NmtWorkspace()
    print(f'Calculating mixing matrix 1 / 3 at {time.strftime("%c")}')
    workspace_spin00.compute_coupling_matrix(field_spin0, field_spin0, bins)
    workspace_spin02 = nmt.NmtWorkspace()
    print(f'Calculating mixing matrix 2 / 3 at {time.strftime("%c")}')
    workspace_spin02.compute_coupling_matrix(field_spin0, field_spin2, bins)
    workspace_spin22 = nmt.NmtWorkspace()
    print(f'Calculating mixing matrix 3 / 3 at {time.strftime("%c")}')
    workspace_spin22.compute_coupling_matrix(field_spin2, field_spin2, bins)
    assert np.all(np.isfinite(workspace_spin00.get_coupling_matrix()))
    assert np.all(np.isfinite(workspace_spin02.get_coupling_matrix()))
    assert np.all(np.isfinite(workspace_spin22.get_coupling_matrix()))

    # Generate list of fields
    print('Generating list of fields')

    if field == 'E':
        fields = [f'E{z}' for z in range(1, n_zbin + 1)]

    elif field == 'N':
        fields = [f'N{z}' for z in range(1, n_zbin + 1)]

    elif field == 'EK':
        fields = [f'E{z}' for z in range(1, n_zbin + 1)]
        fields.append('K1')

    elif field == 'NK':
        fields = [f'N{z}' for z in range(1, n_zbin + 1)]
        fields.append('K1')
    else:
        assert field == 'K'
        fields = ['K1']

    n_field = len(fields)
    if field == 'E' or field == 'N' or field == 'EK' or field == 'NK':
        spectra = [fields[row] + fields[row + diag] for diag in range(n_field) for row in range(n_field - diag)]
        # print(spectra)
        spec_1 = [fields[row] for diag in range(n_field) for row in range(n_field - diag)]
        spec_2 = [fields[row + diag] for diag in range(n_field) for row in range(n_field - diag)]

    else:
        assert field == 'K'
        spectra = ['K1K1']
        spec_1 = ['K1']
        spec_2 = ['K1']

    # Generate list of target spectra (the ones we want the covariance for) in the correct (diagonal) order
    print('Generating list of spectra')

    field_1 = [mysplit(spec_1_id)[0] for spec_1_id in spec_1]
    zbin_1 = [mysplit(spec_1_id)[1] for spec_1_id in spec_1]
    field_2 = [mysplit(spec_2_id)[0] for spec_2_id in spec_2]
    zbin_2 = [mysplit(spec_2_id)[1] for spec_2_id in spec_2]

    # Generate list of sets of mode-coupled theory Cls corresponding to the target spectra
    coupled_theory_cls = []
    for spec_idx, spec in enumerate(spectra):
        print(f'Loading and coupling input Cls {spec_idx + 1} / {len(spectra)}')

        field_types = [field_1[spec_idx], field_2[spec_idx]]
        zbins = [zbin_1[spec_idx], zbin_2[spec_idx]]

        # Get paths of signal and noise spectra to load
        if field_types == ['N', 'N']:
            # NN only
            signal_paths = [f"{signal_path}/galaxy_cl/bin_{max(zbins)}_{min(zbins)}.txt"]
            noise_paths = [f"{noise_path}/galaxy_cl/bin_{max(zbins)}_{min(zbins)}.txt"]
            workspace = workspace_spin00

        elif field_types == ['E', 'E']:
            # EE, EB, BE, BB
            signal_paths = [f"{signal_path}/shear_cl/bin_{max(zbins)}_{min(zbins)}.txt", None, None, None]
            noise_paths = [f"{noise_path}/shear_cl/bin_{max(zbins)}_{min(zbins)}.txt",
                           None,
                           None,
                           f"{noise_path}/shear_cl/bin_{max(zbins)}_{min(zbins)}.txt"]
            workspace = workspace_spin22

        elif field_types == ['K', 'K']:
            # KCMB
            signal_paths = [f"{signal_path}/cmbkappa_cl/bin_1_1.txt"]
            noise_paths = [f"{noise_path}/cmbkappa_cl/bin_1_1.txt"]
            workspace = workspace_spin00

        elif field_types == ['K', 'N']:
            # KN
            signal_paths = [f"{signal_path}/galaxy_cmbkappa_cl/bin_{zbins[1]}_{zbins[0]}.txt"]
            noise_paths = [f"{noise_path}/galaxy_cmbkappa_cl/bin_{zbins[1]}_{zbins[0]}.txt"]
            workspace = workspace_spin00

        elif field_types == ['N', 'K']:
            # KN
            signal_paths = [f"{signal_path}/galaxy_cmbkappa_cl/bin_{zbins[0]}_{zbins[1]}.txt"]
            noise_paths = [f"{noise_path}/galaxy_cmbkappa_cl/bin_{zbins[0]}_{zbins[1]}.txt"]
            workspace = workspace_spin00

        elif field_types == ['K', 'E']:
            # KE, KB
            signal_paths = [f"{signal_path}/shear_cmbkappa_cl/bin_{zbins[1]}_{zbins[0]}.txt", None]
            noise_paths = [f"{noise_path}/shear_cmbkappa_cl/bin_{zbins[1]}_{zbins[0]}.txt", None]
            workspace = workspace_spin02

        elif field_types == ['E', 'K']:
            # EK, BK
            signal_paths = [f"{signal_path}/shear_cmbkappa_cl/bin_{zbins[0]}_{zbins[1]}.txt", None]
            noise_paths = [f"{noise_path}/shear_cmbkappa_cl/bin_{zbins[0]}_{zbins[1]}.txt", None]
            workspace = workspace_spin02

        # Shouldn't need GGL only because we don't consider this as a standalone probe. But this should be the code if
        # it is used in the future
        # elif field_types == ['N', 'E']:
        #     # NE, NB
        #     signal_paths = [f"{signal_path}/galaxy_shear_cl/bin_{zbins[0]}_{zbins[1]}.txt", None]
        #     noise_paths = [f"{signal_path}/galaxy_shear_cl/bin_{zbins[0]}_{zbins[1]}.txt", None]
        #     workspace = workspace_spin02
        #
        # elif field_types == ['E', 'N']:
        #     # EN, BN
        #     signal_paths = [f"{signal_path}/galaxy_shear_cl/bin_{zbins[1]}_{zbins[0]}.txt", None]
        #     noise_paths = [f"{signal_path}/galaxy_shear_cl/bin_{zbins[1]}_{zbins[0]}.txt", None]
        #     workspace = workspace_spin02

        else:
            raise ValueError(f'Unexpected field type: {field}')
        # Load in the signal + noise Cls
        print(f'Loading and coupling input Cls {spec_idx + 1} / {len(spectra)}: Loading...')
        uncoupled_theory_cls = load_cls(signal_paths, noise_paths, lmax_in, lmax_in, lmin_in, noise_lmin)

        # Apply the "improved NKA" method: couple the theory Cls,
        # then divide by fsky to avoid double-counting the reduction in power
        print(f'Loading and coupling input Cls {spec_idx + 1} / {len(spectra)}: Coupling...')

        coupled_cls = workspace.couple_cell(uncoupled_theory_cls)
        assert np.all(np.isfinite(coupled_cls))
        assert len(coupled_cls) == len(uncoupled_theory_cls)
        coupled_theory_cls.append(np.divide(coupled_cls, fsky))

    # Calculate additional covariance coupling coefficients (independent of spin)
    print(f'Computing covariance coupling coefficients at {time.strftime("%c")}')
    cov_workspace = nmt.NmtCovarianceWorkspace()
    cov_workspace.compute_coupling_coefficients(field_spin2, field_spin2, lmax=lmax_in)

    if field == 'EK':
        cov_spectra = [f'E{z}K1' for z in range(1, n_zbin + 1)]
        cov_spec_1 = [f'E{z}' for z in range(1, n_zbin + 1)]
        cov_spec_2 = ['K1'] * n_zbin

    elif field == 'NK':
        cov_spectra = [f'N{z}K1' for z in range(1, n_zbin + 1)]
        cov_spec_1 = [f'N{z}' for z in range(1, n_zbin + 1)]
        cov_spec_2 = ['K1'] * n_zbin

    else:
        cov_spectra = spectra
        cov_spec_1 = spec_1
        cov_spec_2 = spec_2

    cov_field_1 = [mysplit(spec_1_id)[0] for spec_1_id in cov_spec_1]
    cov_zbin_1 = [mysplit(spec_1_id)[1] for spec_1_id in cov_spec_1]
    cov_field_2 = [mysplit(spec_2_id)[0] for spec_2_id in cov_spec_2]
    cov_zbin_2 = [mysplit(spec_2_id)[1] for spec_2_id in cov_spec_2]

    # Iterate over unique pairs of spectra
    for spec_a_idx, spec_a in enumerate(cov_spectra):

        # Obtain the spins and workspace for the first spectrum
        spin_a1 = spin_from_field(cov_field_1[spec_a_idx])
        spin_a2 = spin_from_field(cov_field_2[spec_a_idx])

        workspace_a = workspace_from_spins(spin_a1, spin_a2, workspace_spin00, workspace_spin02, workspace_spin22)

        for spec_b_idx, spec_b in enumerate(cov_spectra[:(spec_a_idx + 1)]):
            print(f'Calculating covariance block row {spec_a_idx} column {spec_b_idx} at {time.strftime("%c")}')

            # Obtain the spins and workspace for the second spectrum
            spin_b1 = spin_from_field(cov_field_1[spec_b_idx])
            spin_b2 = spin_from_field(cov_field_2[spec_b_idx])
            workspace_b = workspace_from_spins(spin_b1, spin_b2, workspace_spin00, workspace_spin02, workspace_spin22)

            # Identify the four power spectra we need to calculate this covariance
            a1b1 = spectrum_from_fields_1x2pt(cov_field_1[spec_a_idx], cov_zbin_1[spec_a_idx], cov_field_1[spec_b_idx], cov_zbin_1[spec_b_idx])
            a1b2 = spectrum_from_fields_1x2pt(cov_field_1[spec_a_idx], cov_zbin_1[spec_a_idx], cov_field_2[spec_b_idx], cov_zbin_2[spec_b_idx])
            a2b1 = spectrum_from_fields_1x2pt(cov_field_2[spec_a_idx], cov_zbin_2[spec_a_idx], cov_field_1[spec_b_idx], cov_zbin_1[spec_b_idx])
            a2b2 = spectrum_from_fields_1x2pt(cov_field_2[spec_a_idx], cov_zbin_2[spec_a_idx], cov_field_2[spec_b_idx], cov_zbin_2[spec_b_idx])

            # Obtain the corresponding theory Cls
            cl_a1b1 = coupled_theory_cls[spectra.index(a1b1)]
            cl_a1b2 = coupled_theory_cls[spectra.index(a1b2)]
            cl_a2b1 = coupled_theory_cls[spectra.index(a2b1)]
            cl_a2b2 = coupled_theory_cls[spectra.index(a2b2)]
            assert np.all(np.isfinite(cl_a1b1))
            assert np.all(np.isfinite(cl_a1b2))
            assert np.all(np.isfinite(cl_a2b1))
            assert np.all(np.isfinite(cl_a2b2))

            # Evaluate the covariance
            cl_cov = nmt.gaussian_covariance(cov_workspace, spin_a1, spin_a2, spin_b1, spin_b2, cl_a1b1, cl_a1b2,
                                             cl_a2b1, cl_a2b2, workspace_a, workspace_b, coupled=True)
            # print(cl_cov.shape)
            # Extract the part of the covariance we want, which is conveniently always the [..., 0, ..., 0] block,
            # since all other blocks relate to B-modes
            # print(coupled_theory_cls[spec_a_idx])
            # cl_cov = cl_cov.reshape((lmax_in + 1, len(coupled_theory_cls[spec_a_idx]),
            #                          lmax_in + 1, len(coupled_theory_cls[spec_b_idx])))
            cl_cov = cl_cov.reshape((lmax_in + 1, maths.factorial(spin_a1) * maths.factorial(spin_a2),
                                     lmax_in + 1, maths.factorial(spin_b1) * maths.factorial(spin_b2)))
            # print(cl_cov.shape)
            cl_cov = cl_cov[:, 0, :, 0]
            cl_cov = cl_cov[lmin_out:(lmax_out + 1), lmin_out:(lmax_out + 1)]

            # Do some checks and save to disk
            assert np.all(np.isfinite(cl_cov))
            n_ell_out = lmax_out - lmin_out + 1
            assert cl_cov.shape == (n_ell_out, n_ell_out)
            if spec_a_idx == spec_b_idx:
                assert np.allclose(cl_cov, cl_cov.T)
            save_path = save_block_filemask.format(spec1_idx=spec_a_idx, spec2_idx=spec_b_idx)
            header = (f'Output from {__file__}.get_3x2pt_cov for spectra ({spec_a}, {spec_b}), with parameters '
                      f'n_zbin = {n_zbin}, lmax_in = {lmax_in}, lmin_in = {lmin_in}, '
                      f'lmax_out = {lmax_out}, lmin_out = {lmin_out}, '
                      f'mask_path = {mask_path}, '
                      f'nside = {nside}; time {time.strftime("%c")}')
            print(f'Saving block at {time.strftime("%c")}')
            np.savez_compressed(save_path, cov_block=cl_cov, spec1_idx=spec_a_idx, spec2_idx=spec_b_idx, header=header)
            print(f'Saving {save_path} at {time.strftime("%c")}')

    print(f'Binning at {time.strftime("%c")}')

    if field not in {'E', 'N', 'K', 'EK', 'NK'}:
        print(f'Unknown field type - {field}')
        sys.exit()

    if field == 'K':
        n_fields = 1
        n_spec = 1

    elif field == 'EK' or field == 'NK':
        n_fields = n_zbin
        n_spec = n_zbin

    else:
        assert field == 'E' or field == 'N'
        n_fields = n_zbin
        n_spec = n_fields * (n_fields + 1) // 2

    n_ell_cov = lmax_out - lmin_out + 1

    print(f'Calculating binning matrix at {time.strftime("%c")}')

    pbl = mask.get_binning_matrix(
        n_bandpowers=n_bp,
        output_lmin=lmin_out,
        output_lmax=lmax_out,
        bp_spacing=bandpower_spacing)

    assert pbl.shape == (n_bp, n_ell_cov)

    print(f'Preallocating full covariance at {time.strftime("%c")}')
    n_data = n_spec * n_bp
    cov = np.full((n_data, n_data), np.nan)

    # Loop over all blocks
    for spec1 in range(n_spec):
        for spec2 in range(spec1 + 1):
            print(f'spec1 = {spec1}, spec2 = {spec2} at {time.strftime("%c")}')

            print('Loading block')
            block_path = save_block_filemask.format(spec1_idx=spec1, spec2_idx=spec2)
            with np.load(block_path) as data:
                assert data['spec1_idx'] == spec1
                assert data['spec2_idx'] == spec2
                block_unbinned = data['cov_block']
            assert np.all(np.isfinite(block_unbinned))
            assert block_unbinned.shape == (n_ell_cov, n_ell_cov)
            # lowl_skip = lmin_out - lmin_in
            # block_unbinned = block_unbinned[lowl_skip:, lowl_skip:]
            # assert block_unbi/nned.shape == (n_ell_out, n_ell_out)

            print('Binning block')
            block_binned = pbl @ block_unbinned @ pbl.T
            assert np.all(np.isfinite(block_binned))
            assert block_binned.shape == (n_bp, n_bp)

            print('Inserting block')
            cov[(spec1 * n_bp):((spec1 + 1) * n_bp), (spec2 * n_bp):((spec2 + 1) * n_bp)] = block_binned

    # Reflect to fill remaining elements, and check symmetric
    cov = np.where(np.isnan(cov), cov.T, cov)
    assert np.all(np.isfinite(cov))
    assert np.allclose(cov, cov.T, atol=0)

    # Save to disk
    save_path = save_cov_filemask.format(n_bp=n_bp)
    header = (f'Full binned covariance matrix. Output from {__file__}.bin_combine_cov for n_bp = {n_bp}, '
              f'n_zbin = {n_zbin}, lmin_in = {lmin_in}, lmin_out = {lmin_out}, lmax_out = {lmax_out}, '
              f'save_block_filemask = {save_block_filemask},'
              f'save_cov_filemask = {save_cov_filemask}, at {time.strftime("%c")}')
    np.savez_compressed(save_path, cov=cov, n_bp=n_bp, header=header)
    print(f'Saved {save_path} at {time.strftime("%c")}')
    print()

    print(f'Done at {time.strftime("%c")}')


def get_1x2pt_cov_pcl_diff_masks(n_zbin, signal_path, noise_path, field, lmax_in, lmin_in, lmax_out, lmin_out, noise_lmin,
                      mask_path, cmb_mask_path, nside, save_block_filemask, n_bp, pbl, save_cov_filemask):
    """
    Calculate 3x2pt Gaussian covariance using NaMaster, saving each block separately to disk. This function computes
    the covariance associated with a Pseudo-Cl in the absence of noise - i.e. we apply the coupling coefficients of
    a mask to a theoretical (full-sky) Cl (using the coupled=True argument in NaMaster gaussian_covariance). The
    complete covariance for a noisy Pseudo-Cl observation can be calculated by combining this function with
    get_3x2pt_cov_noise - which calculates the Gaussian covariance of a predicted noise spectrum. NOTE - this
    theoretical noise spectrum must predict the noise associated with the region of sky observed (unlike the full-sky
    Cl) since it is NOT combined with the mask coupling coefficients.

    Args:
        n_zbin (int): Number of redshift bins, assuming 1 position field and 1 shear field per redshift bin.
        pos_pos_filemask (str): Path to text file containing a position-position power spectrum with ``{hi_zbin}`` and
                                ``{lo_zbin}`` placeholders.
        pos_she_filemask (str): Path to text file containing a position-shear power spectrum with ``{pos_zbin}`` and
                                ``{she_zbin}`` placeholders.
        she_she_filemask (str): Path to text file containing a shear-shear power spectrum with ``{hi_zbin}`` and
                                ``{lo_zbin}`` placeholders.
        lmax_in (int): Maximum l to including in mixing.
        lmin_in (int): Minimum l in input power spectra.
        lmax_out (int): Maximum l to include in covariance.
        lmin_out (int): Minimum l to include in covariance.
        noise_lmin (int): Minimum l in noise power spectra.
        mask_path (str): Path to mask FITS file. If None, full sky is assumed, in which case the covariance will be
                         diagonal.
        nside (int): HEALPix resolution nside parameter.
        save_filemask (str): Path to save each covariance block to disk, with ``{spec1_idx}`` and ``{spec2_idx}``
                             placeholders.
    """

    # Load and rescale mask, and calculate fsky
    if mask_path is not None:
        print('Loading and rescaling mask')
        lss_mask = hp.pixelfunc.ud_grade(hp.read_map(mask_path, dtype=float), nside)
        assert np.amin(lss_mask) == 0
        assert np.amax(lss_mask) == 1
    else:
        print('Full sky')
        lss_mask = np.ones(hp.pixelfunc.nside2npix(nside))
    assert np.all(np.isfinite(lss_mask))
    fsky = np.mean(lss_mask)
    print(f'LSS fsky = {fsky:.3f}')

    if cmb_mask_path is not None:
        print('Loading and rescaling mask')
        cmb_mask = hp.pixelfunc.ud_grade(hp.read_map(cmb_mask_path, dtype=float), nside)
        assert np.amin(cmb_mask) == 0
        assert np.amax(cmb_mask) == 1
    else:
        print('Full sky')
        cmb_mask = np.ones(hp.pixelfunc.nside2npix(nside))
    assert np.all(np.isfinite(cmb_mask))
    fsky_cmb = np.mean(cmb_mask)
    print(f'CMB fsky = {fsky_cmb:.3f}')

    # Create NaMaster binning scheme as individual Cls
    print('Creating binning scheme')
    bins = nmt.NmtBin.from_lmax_linear(lmax_in, 1)

    if field == 'EK' or field == 'NK':
        # Calculate mixing matrices for spin 0-0, 0-2 (equivalent to 2-0), and 2-2
        field_spin0_lss = nmt.NmtField(lss_mask, None, spin=0, lite=True, lmax_sht=lmax_in)
        field_spin2_lss = nmt.NmtField(lss_mask, None, spin=2, lite=True, lmax_sht=lmax_in)

        field_spin0_cmb = nmt.NmtField(cmb_mask, None, spin=0, lite=True, lmax_sht=lmax_in)
        field_spin2_cmb = nmt.NmtField(cmb_mask, None, spin=2, lite=True, lmax_sht=lmax_in)

    elif field == 'KK':
        obs_mask = cmb_mask
        # Calculate mixing matrices for spin 0-0, 0-2 (equivalent to 2-0), and 2-2
        field_spin0 = nmt.NmtField(obs_mask, None, spin=0, lite=True, lmax_sht=lmax_in)
        field_spin2 = nmt.NmtField(obs_mask, None, spin=2, lite=True, lmax_sht=lmax_in)
    else:
        assert field == 'E' or field == 'N'
        obs_mask = lss_mask
        # Calculate mixing matrices for spin 0-0, 0-2 (equivalent to 2-0), and 2-2
        field_spin0 = nmt.NmtField(obs_mask, None, spin=0, lite=True, lmax_sht=lmax_in)
        field_spin2 = nmt.NmtField(obs_mask, None, spin=2, lite=True, lmax_sht=lmax_in)

    workspace_spin00 = nmt.NmtWorkspace()
    print(f'Calculating mixing matrix 1 / 3 at {time.strftime("%c")}')
    workspace_spin00.compute_coupling_matrix(field_spin0, field_spin0, bins)
    workspace_spin02 = nmt.NmtWorkspace()
    print(f'Calculating mixing matrix 2 / 3 at {time.strftime("%c")}')
    workspace_spin02.compute_coupling_matrix(field_spin0, field_spin2, bins)
    workspace_spin22 = nmt.NmtWorkspace()
    print(f'Calculating mixing matrix 3 / 3 at {time.strftime("%c")}')
    workspace_spin22.compute_coupling_matrix(field_spin2, field_spin2, bins)
    assert np.all(np.isfinite(workspace_spin00.get_coupling_matrix()))
    assert np.all(np.isfinite(workspace_spin02.get_coupling_matrix()))
    assert np.all(np.isfinite(workspace_spin22.get_coupling_matrix()))

    # Generate list of fields
    print('Generating list of fields')

    if field == 'E':
        fields = [f'E{z}' for z in range(1, n_zbin + 1)]

    elif field == 'N':
        fields = [f'N{z}' for z in range(1, n_zbin + 1)]

    elif field == 'EK':
        fields = [f'E{z}' for z in range(1, n_zbin + 1)]
        fields.append('K1')

    elif field == 'NK':
        fields = [f'N{z}' for z in range(1, n_zbin + 1)]
        fields.append('K1')
    else:
        assert field == 'K'
        fields = ['K1']

    n_field = len(fields)
    if field == 'E' or field == 'N' or field == 'EK' or field == 'NK':
        spectra = [fields[row] + fields[row + diag] for diag in range(n_field) for row in range(n_field - diag)]
        # print(spectra)
        spec_1 = [fields[row] for diag in range(n_field) for row in range(n_field - diag)]
        spec_2 = [fields[row + diag] for diag in range(n_field) for row in range(n_field - diag)]

    else:
        assert field == 'K'
        spectra = ['K1K1']
        spec_1 = ['K1']
        spec_2 = ['K1']

    # Generate list of target spectra (the ones we want the covariance for) in the correct (diagonal) order
    print('Generating list of spectra')

    field_1 = [mysplit(spec_1_id)[0] for spec_1_id in spec_1]
    zbin_1 = [mysplit(spec_1_id)[1] for spec_1_id in spec_1]
    field_2 = [mysplit(spec_2_id)[0] for spec_2_id in spec_2]
    zbin_2 = [mysplit(spec_2_id)[1] for spec_2_id in spec_2]

    # Generate list of sets of mode-coupled theory Cls corresponding to the target spectra
    coupled_theory_cls = []
    for spec_idx, spec in enumerate(spectra):
        print(f'Loading and coupling input Cls {spec_idx + 1} / {len(spectra)}')

        field_types = [field_1[spec_idx], field_2[spec_idx]]
        zbins = [zbin_1[spec_idx], zbin_2[spec_idx]]

        # Get paths of signal and noise spectra to load
        if field_types == ['N', 'N']:
            # NN only
            signal_paths = [f"{signal_path}/galaxy_cl/bin_{max(zbins)}_{min(zbins)}.txt"]
            noise_paths = [f"{noise_path}/galaxy_cl/bin_{max(zbins)}_{min(zbins)}.txt"]
            workspace = workspace_spin00

        elif field_types == ['E', 'E']:
            # EE, EB, BE, BB
            signal_paths = [f"{signal_path}/shear_cl/bin_{max(zbins)}_{min(zbins)}.txt", None, None, None]
            noise_paths = [f"{noise_path}/shear_cl/bin_{max(zbins)}_{min(zbins)}.txt",
                           None,
                           None,
                           f"{noise_path}/shear_cl/bin_{max(zbins)}_{min(zbins)}.txt"]
            workspace = workspace_spin22

        elif field_types == ['K', 'K']:
            # KCMB
            signal_paths = [f"{signal_path}/cmbkappa_cl/bin_1_1.txt"]
            noise_paths = [f"{noise_path}/cmbkappa_cl/bin_1_1.txt"]
            workspace = workspace_spin00

        elif field_types == ['K', 'N']:
            # KN
            signal_paths = [f"{signal_path}/galaxy_cmbkappa_cl/bin_{zbins[1]}_{zbins[0]}.txt"]
            noise_paths = [f"{noise_path}/galaxy_cmbkappa_cl/bin_{zbins[1]}_{zbins[0]}.txt"]
            workspace = workspace_spin00

        elif field_types == ['N', 'K']:
            # KN
            signal_paths = [f"{signal_path}/galaxy_cmbkappa_cl/bin_{zbins[0]}_{zbins[1]}.txt"]
            noise_paths = [f"{noise_path}/galaxy_cmbkappa_cl/bin_{zbins[0]}_{zbins[1]}.txt"]
            workspace = workspace_spin00

        elif field_types == ['K', 'E']:
            # KE, KB
            signal_paths = [f"{signal_path}/shear_cmbkappa_cl/bin_{zbins[1]}_{zbins[0]}.txt", None]
            noise_paths = [f"{noise_path}/shear_cmbkappa_cl/bin_{zbins[1]}_{zbins[0]}.txt", None]
            workspace = workspace_spin02

        elif field_types == ['E', 'K']:
            # EK, BK
            signal_paths = [f"{signal_path}/shear_cmbkappa_cl/bin_{zbins[0]}_{zbins[1]}.txt", None]
            noise_paths = [f"{noise_path}/shear_cmbkappa_cl/bin_{zbins[0]}_{zbins[1]}.txt", None]
            workspace = workspace_spin02

        # Shouldn't need GGL only because we don't consider this as a standalone probe. But this should be the code if
        # it is used in the future
        # elif field_types == ['N', 'E']:
        #     # NE, NB
        #     signal_paths = [f"{signal_path}/galaxy_shear_cl/bin_{zbins[0]}_{zbins[1]}.txt", None]
        #     noise_paths = [f"{signal_path}/galaxy_shear_cl/bin_{zbins[0]}_{zbins[1]}.txt", None]
        #     workspace = workspace_spin02
        #
        # elif field_types == ['E', 'N']:
        #     # EN, BN
        #     signal_paths = [f"{signal_path}/galaxy_shear_cl/bin_{zbins[1]}_{zbins[0]}.txt", None]
        #     noise_paths = [f"{signal_path}/galaxy_shear_cl/bin_{zbins[1]}_{zbins[0]}.txt", None]
        #     workspace = workspace_spin02

        else:
            raise ValueError(f'Unexpected field type: {field}')
        # Load in the signal + noise Cls
        print(f'Loading and coupling input Cls {spec_idx + 1} / {len(spectra)}: Loading...')
        uncoupled_theory_cls = load_cls(signal_paths, noise_paths, lmax_in, lmax_in, lmin_in, noise_lmin)

        # Apply the "improved NKA" method: couple the theory Cls,
        # then divide by fsky to avoid double-counting the reduction in power
        print(f'Loading and coupling input Cls {spec_idx + 1} / {len(spectra)}: Coupling...')

        coupled_cls = workspace.couple_cell(uncoupled_theory_cls)
        assert np.all(np.isfinite(coupled_cls))
        assert len(coupled_cls) == len(uncoupled_theory_cls)
        coupled_theory_cls.append(np.divide(coupled_cls, fsky))

    # Calculate additional covariance coupling coefficients (independent of spin)
    print(f'Computing covariance coupling coefficients at {time.strftime("%c")}')
    cov_workspace = nmt.NmtCovarianceWorkspace()
    cov_workspace.compute_coupling_coefficients(field_spin2, field_spin2, lmax=lmax_in)

    if field == 'EK':
        cov_spectra = [f'E{z}K1' for z in range(1, n_zbin + 1)]
        cov_spec_1 = [f'E{z}' for z in range(1, n_zbin + 1)]
        cov_spec_2 = ['K1'] * n_zbin

    elif field == 'NK':
        cov_spectra = [f'N{z}K1' for z in range(1, n_zbin + 1)]
        cov_spec_1 = [f'N{z}' for z in range(1, n_zbin + 1)]
        cov_spec_2 = ['K1'] * n_zbin

    else:
        cov_spectra = spectra
        cov_spec_1 = spec_1
        cov_spec_2 = spec_2

    cov_field_1 = [mysplit(spec_1_id)[0] for spec_1_id in cov_spec_1]
    cov_zbin_1 = [mysplit(spec_1_id)[1] for spec_1_id in cov_spec_1]
    cov_field_2 = [mysplit(spec_2_id)[0] for spec_2_id in cov_spec_2]
    cov_zbin_2 = [mysplit(spec_2_id)[1] for spec_2_id in cov_spec_2]

    # Iterate over unique pairs of spectra
    for spec_a_idx, spec_a in enumerate(cov_spectra):

        # Obtain the spins and workspace for the first spectrum
        spin_a1 = spin_from_field(cov_field_1[spec_a_idx])
        spin_a2 = spin_from_field(cov_field_2[spec_a_idx])

        workspace_a = workspace_from_spins(spin_a1, spin_a2, workspace_spin00, workspace_spin02, workspace_spin22)

        for spec_b_idx, spec_b in enumerate(cov_spectra[:(spec_a_idx + 1)]):
            print(f'Calculating covariance block row {spec_a_idx} column {spec_b_idx} at {time.strftime("%c")}')

            # Obtain the spins and workspace for the second spectrum
            spin_b1 = spin_from_field(cov_field_1[spec_b_idx])
            spin_b2 = spin_from_field(cov_field_2[spec_b_idx])
            workspace_b = workspace_from_spins(spin_b1, spin_b2, workspace_spin00, workspace_spin02, workspace_spin22)

            # Identify the four power spectra we need to calculate this covariance
            a1b1 = spectrum_from_fields_1x2pt(cov_field_1[spec_a_idx], cov_zbin_1[spec_a_idx], cov_field_1[spec_b_idx], cov_zbin_1[spec_b_idx])
            a1b2 = spectrum_from_fields_1x2pt(cov_field_1[spec_a_idx], cov_zbin_1[spec_a_idx], cov_field_2[spec_b_idx], cov_zbin_2[spec_b_idx])
            a2b1 = spectrum_from_fields_1x2pt(cov_field_2[spec_a_idx], cov_zbin_2[spec_a_idx], cov_field_1[spec_b_idx], cov_zbin_1[spec_b_idx])
            a2b2 = spectrum_from_fields_1x2pt(cov_field_2[spec_a_idx], cov_zbin_2[spec_a_idx], cov_field_2[spec_b_idx], cov_zbin_2[spec_b_idx])

            # Obtain the corresponding theory Cls
            cl_a1b1 = coupled_theory_cls[spectra.index(a1b1)]
            cl_a1b2 = coupled_theory_cls[spectra.index(a1b2)]
            cl_a2b1 = coupled_theory_cls[spectra.index(a2b1)]
            cl_a2b2 = coupled_theory_cls[spectra.index(a2b2)]
            assert np.all(np.isfinite(cl_a1b1))
            assert np.all(np.isfinite(cl_a1b2))
            assert np.all(np.isfinite(cl_a2b1))
            assert np.all(np.isfinite(cl_a2b2))

            # Evaluate the covariance
            cl_cov = nmt.gaussian_covariance(cov_workspace, spin_a1, spin_a2, spin_b1, spin_b2, cl_a1b1, cl_a1b2,
                                             cl_a2b1, cl_a2b2, workspace_a, workspace_b, coupled=True)
            # print(cl_cov.shape)
            # Extract the part of the covariance we want, which is conveniently always the [..., 0, ..., 0] block,
            # since all other blocks relate to B-modes
            # print(coupled_theory_cls[spec_a_idx])
            # cl_cov = cl_cov.reshape((lmax_in + 1, len(coupled_theory_cls[spec_a_idx]),
            #                          lmax_in + 1, len(coupled_theory_cls[spec_b_idx])))
            cl_cov = cl_cov.reshape((lmax_in + 1, maths.factorial(spin_a1) * maths.factorial(spin_a2),
                                     lmax_in + 1, maths.factorial(spin_b1) * maths.factorial(spin_b2)))
            # print(cl_cov.shape)
            cl_cov = cl_cov[:, 0, :, 0]
            cl_cov = cl_cov[lmin_out:(lmax_out + 1), lmin_out:(lmax_out + 1)]

            # Do some checks and save to disk
            assert np.all(np.isfinite(cl_cov))
            n_ell_out = lmax_out - lmin_out + 1
            assert cl_cov.shape == (n_ell_out, n_ell_out)
            if spec_a_idx == spec_b_idx:
                assert np.allclose(cl_cov, cl_cov.T)
            save_path = save_block_filemask.format(spec1_idx=spec_a_idx, spec2_idx=spec_b_idx)
            header = (f'Output from {__file__}.get_3x2pt_cov for spectra ({spec_a}, {spec_b}), with parameters '
                      f'n_zbin = {n_zbin}, lmax_in = {lmax_in}, lmin_in = {lmin_in}, '
                      f'lmax_out = {lmax_out}, lmin_out = {lmin_out}, '
                      f'mask_path = {mask_path}, '
                      f'nside = {nside}; time {time.strftime("%c")}')
            print(f'Saving block at {time.strftime("%c")}')
            np.savez_compressed(save_path, cov_block=cl_cov, spec1_idx=spec_a_idx, spec2_idx=spec_b_idx, header=header)
            print(f'Saving {save_path} at {time.strftime("%c")}')

    print(f'Binning at {time.strftime("%c")}')

    if field not in {'E', 'N', 'K', 'EK', 'NK'}:
        print(f'Unknown field type - {field}')
        sys.exit()

    if field == 'K':
        n_fields = 1
        n_spec = 1

    elif field == 'EK' or field == 'NK':
        n_fields = n_zbin
        n_spec = n_zbin

    else:
        assert field == 'E' or field == 'N'
        n_fields = n_zbin
        n_spec = n_fields * (n_fields + 1) // 2

    n_ell_cov = lmax_out - lmin_out + 1

    print(f'Calculating binning matrix at {time.strftime("%c")}')

    assert pbl.shape == (n_bp, n_ell_cov)

    print(f'Preallocating full covariance at {time.strftime("%c")}')
    n_data = n_spec * n_bp
    cov = np.full((n_data, n_data), np.nan)

    # Loop over all blocks
    for spec1 in range(n_spec):
        for spec2 in range(spec1 + 1):
            print(f'spec1 = {spec1}, spec2 = {spec2} at {time.strftime("%c")}')

            print('Loading block')
            block_path = save_block_filemask.format(spec1_idx=spec1, spec2_idx=spec2)
            with np.load(block_path) as data:
                assert data['spec1_idx'] == spec1
                assert data['spec2_idx'] == spec2
                block_unbinned = data['cov_block']
            assert np.all(np.isfinite(block_unbinned))
            assert block_unbinned.shape == (n_ell_cov, n_ell_cov)
            # lowl_skip = lmin_out - lmin_in
            # block_unbinned = block_unbinned[lowl_skip:, lowl_skip:]
            # assert block_unbi/nned.shape == (n_ell_out, n_ell_out)

            print('Binning block')
            block_binned = pbl @ block_unbinned @ pbl.T
            assert np.all(np.isfinite(block_binned))
            assert block_binned.shape == (n_bp, n_bp)

            print('Inserting block')
            cov[(spec1 * n_bp):((spec1 + 1) * n_bp), (spec2 * n_bp):((spec2 + 1) * n_bp)] = block_binned

    # Reflect to fill remaining elements, and check symmetric
    cov = np.where(np.isnan(cov), cov.T, cov)
    assert np.all(np.isfinite(cov))
    assert np.allclose(cov, cov.T, atol=0)

    # Save to disk
    save_path = save_cov_filemask.format(n_bp=n_bp)
    header = (f'Full binned covariance matrix. Output from {__file__}.bin_combine_cov for n_bp = {n_bp}, '
              f'n_zbin = {n_zbin}, lmin_in = {lmin_in}, lmin_out = {lmin_out}, lmax_out = {lmax_out}, '
              f'save_block_filemask = {save_block_filemask},'
              f'save_cov_filemask = {save_cov_filemask}, at {time.strftime("%c")}')
    np.savez_compressed(save_path, cov=cov, n_bp=n_bp, header=header)
    print(f'Saved {save_path} at {time.strftime("%c")}')
    print()

    print(f'Done at {time.strftime("%c")}')

























'''
def bin_combine_cov_1x2pt(n_zbin, field, lmin_in, lmin_out, lmax, n_bp, pbl, input_filemask, save_filemask):

    """
    Loop over numbers of bandpowers, and for each one, bin all blocks of unbinned covariance matrix and combine into
    a single matrix, saved to disk.

    Args:
        n_zbin (int): Number of redshift bins, assuming 1 position field and 1 shear field per redshift bin.
        lmin_in (int): Minimum l in the unbinned covariance.
        lmin_out (int): Minimum l to include in the binned covariance.
        lmax (int): Maximum l in the unbinned and binned covariance.
        n_bp_min (int): Minimum number of bandpowers to loop over (inclusive).
        n_bp_max (int): Maximum number of bandpowers to loop over (inclusive).
        input_filemask (str): Path to unbinned covariance blocks output by ``get_3x2pt_cov``, with ``{spec1_idx}`` and
                              ``{spec2_idx}`` placeholders.
        save_filemask (str): Path to save each binned covariance matrix, with ``{n_bp}`` placeholder.
    """

    # Calculate number of spectra and ells

    if field not in {'E', 'N', 'K', 'EK', 'NK'}:
        print(f'Unknown field type - {field}')
        sys.exit()

    if field == 'K':
        n_fields = 1
        n_spec = 1

    elif field == 'EK' or field == 'NK':
        n_fields = n_zbin
        n_spec = n_zbin

    else:
        assert field == 'E' or field == 'N'
        n_fields = n_zbin
        n_spec = n_fields * (n_fields + 1) // 2

    n_ell_in = lmax - lmin_in + 1
    n_ell_out = lmax - lmin_out + 1

    print(f'Calculating binning matrix at {time.strftime("%c")}')

    assert pbl.shape == (n_bp, n_ell_out)

    print(f'Preallocating full covariance at {time.strftime("%c")}')
    n_data = n_spec * n_bp
    cov = np.full((n_data, n_data), np.nan)

    # Loop over all blocks
    for spec1 in range(n_spec):
        for spec2 in range(spec1 + 1):
            print(f'spec1 = {spec1}, spec2 = {spec2} at {time.strftime("%c")}')

            print('Loading block')
            block_path = input_filemask.format(spec1_idx=spec1, spec2_idx=spec2)
            with np.load(block_path) as data:
                assert data['spec1_idx'] == spec1
                assert data['spec2_idx'] == spec2
                block_unbinned = data['cov_block']
            assert np.all(np.isfinite(block_unbinned))
            assert block_unbinned.shape == (n_ell_in, n_ell_in)
            lowl_skip = lmin_out - lmin_in
            block_unbinned = block_unbinned[lowl_skip:, lowl_skip:]
            assert block_unbinned.shape == (n_ell_out, n_ell_out)

            print('Binning block')
            block_binned = pbl @ block_unbinned @ pbl.T
            assert np.all(np.isfinite(block_binned))
            assert block_binned.shape == (n_bp, n_bp)

            print('Inserting block')
            cov[(spec1 * n_bp):((spec1 + 1) * n_bp), (spec2 * n_bp):((spec2 + 1) * n_bp)] = block_binned

    # Reflect to fill remaining elements, and check symmetric
    cov = np.where(np.isnan(cov), cov.T, cov)
    assert np.all(np.isfinite(cov))
    assert np.allclose(cov, cov.T, atol=0)

    # Save to disk
    save_path = save_filemask.format(n_bp=n_bp)
    header = (f'Full binned covariance matrix. Output from {__file__}.bin_combine_cov for n_bp = {n_bp}, '
              f'n_zbin = {n_zbin}, lmin_in = {lmin_in}, lmin_out = {lmin_out}, lmax = {lmax}, '
              f'input_filemask = {input_filemask}, at {time.strftime("%c")}')
    np.savez_compressed(save_path, cov=cov, n_bp=n_bp, header=header)
    print(f'Saved {save_path} at {time.strftime("%c")}')
    print()

    print(f'Done at {time.strftime("%c")}')
'''


def get_3x2pt_cov_pcl(n_zbin, signal_path, noise_path, lmax_in, lmin_in, lmax_out_nn,lmin_out_nn,lmax_out_ne,lmin_out_ne,
                      lmax_out_ee,lmin_out_ee, noise_lmin, mask_path, nside, save_filemask, n_bp, bandpower_spacing,
                      cov_filemask):

    """
    Calculate 3x2pt Gaussian covariance using NaMaster, saving each block separately to disk. This function computes
    the covariance associated with a Pseudo-Cl in the absence of noise - i.e. we apply the coupling coefficients of
    a mask to a theoretical (full-sky) Cl (using the coupled=True argument in NaMaster gaussian_covariance). The
    complete covariance for a noisy Pseudo-Cl observation can be calculated by combining this function with
    get_3x2pt_cov_noise - which calculates the Gaussian covariance of a predicted noise spectrum. NOTE - this
    theoretical noise spectrum must predict the noise associated with the region of sky observed (unlike the full-sky
    Cl) since it is NOT combined with the mask coupling coefficients.

    Args:
        n_zbin (int): Number of redshift bins, assuming 1 position field and 1 shear field per redshift bin.
        pos_pos_filemask (str): Path to text file containing a position-position power spectrum with ``{hi_zbin}`` and
                                ``{lo_zbin}`` placeholders.
        pos_she_filemask (str): Path to text file containing a position-shear power spectrum with ``{pos_zbin}`` and
                                ``{she_zbin}`` placeholders.
        she_she_filemask (str): Path to text file containing a shear-shear power spectrum with ``{hi_zbin}`` and
                                ``{lo_zbin}`` placeholders.
        lmax_in (int): Maximum l to including in mixing.
        lmin_in (int): Minimum l in input power spectra.
        lmax_out (int): Maximum l to include in covariance.
        lmin_out (int): Minimum l to include in covariance.
        noise_lmin (int): Minimum l in noise power spectra.
        mask_path (str): Path to mask FITS file. If None, full sky is assumed, in which case the covariance will be
                         diagonal.
        nside (int): HEALPix resolution nside parameter.
        save_filemask (str): Path to save each covariance block to disk, with ``{spec1_idx}`` and ``{spec2_idx}``
                             placeholders.
    """

    pos_pos_filemask = signal_path + 'galaxy_cl/bin_{hi_zbin}_{lo_zbin}.txt'
    pos_she_filemask = signal_path + 'galaxy_shear_cl/bin_{pos_zbin}_{she_zbin}.txt'
    she_she_filemask = signal_path + 'shear_cl/bin_{hi_zbin}_{lo_zbin}.txt'

    pos_nl_path = noise_path + 'galaxy_cl/bin_{hi_zbin}_{lo_zbin}.txt'
    pos_she_nl_path = noise_path + 'galaxy_shear_cl/bin_{pos_zbin}_{she_zbin}.txt'
    she_nl_path = noise_path + 'shear_cl/bin_{hi_zbin}_{lo_zbin}.txt'

    assert lmin_out_ne == lmin_out_nn
    assert lmax_out_ne == lmax_out_nn

    # Load and rescale mask, and calculate fsky
    if mask_path is not None:
        print('Loading and rescaling mask')
        obs_mask = hp.pixelfunc.ud_grade(hp.read_map(mask_path, dtype=float), nside)
        assert np.amin(obs_mask) == 0
        assert np.amax(obs_mask) == 1
    else:
        print('Full sky')
        obs_mask = np.ones(hp.pixelfunc.nside2npix(nside))
    assert np.all(np.isfinite(obs_mask))
    fsky = np.mean(obs_mask)
    print(f'fsky = {fsky:.3f}')

    # Create NaMaster binning scheme as individual Cls
    print('Creating binning scheme')
    bins = nmt.NmtBin.from_lmax_linear(lmax_in, 1)

    # Calculate mixing matrices for spin 0-0, 0-2 (equivalent to 2-0), and 2-2
    field_spin0 = nmt.NmtField(obs_mask, None, spin=0, lite=True, lmax_sht=lmax_in)
    field_spin2 = nmt.NmtField(obs_mask, None, spin=2, lite=True, lmax_sht=lmax_in)
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
    assert np.all(np.isfinite(workspace_spin00.get_coupling_matrix()))
    assert np.all(np.isfinite(workspace_spin02.get_coupling_matrix()))
    assert np.all(np.isfinite(workspace_spin22.get_coupling_matrix()))

    # Generate list of fields
    print('Generating list of fields')
    #
    n_field = 2 * n_zbin
    # n_spec = n_field * (n_field + 1) // 2
    fields = [f'{f}{z}' for z in range(1, n_zbin + 1) for f in ['N', 'E']]

    spectra = [fields[row] + fields[row + diag] for diag in range(n_field) for row in range(n_field - diag)]
    # print(fields)
    # print(spectra)
    spec_1 = [fields[row] for diag in range(n_field) for row in range(n_field - diag)]
    spec_2 = [fields[row + diag] for diag in range(n_field) for row in range(n_field - diag)]

    field_1 = [mysplit(spec_1_id)[0] for spec_1_id in spec_1]
    zbin_1 = [mysplit(spec_1_id)[1] for spec_1_id in spec_1]
    field_2 = [mysplit(spec_2_id)[0] for spec_2_id in spec_2]
    zbin_2 = [mysplit(spec_2_id)[1] for spec_2_id in spec_2]

    coupled_theory_cls = []
    for spec_idx, spec in enumerate(spectra):
        print(f'Loading and coupling input Cls {spec_idx + 1} / {len(spectra)}')

        field_types = [field_1[spec_idx], field_2[spec_idx]]
        zbins = [zbin_1[spec_idx], zbin_2[spec_idx]]

        # Get paths of signal and noise spectra to load
        if field_types == ['N', 'N']:
            # NN only
            signal_paths = [pos_pos_filemask.format(hi_zbin=max(zbins), lo_zbin=min(zbins))]
            noise_paths = [pos_nl_path.format(hi_zbin=max(zbins), lo_zbin=min(zbins))]
            workspace = workspace_spin00

        elif field_types == ['N', 'E']:
            # NE, NB
            signal_paths = [pos_she_filemask.format(pos_zbin=zbins[0], she_zbin=zbins[1]), None]
            noise_paths = [pos_she_nl_path.format(pos_zbin=zbins[0], she_zbin=zbins[1]), None]
            # print(signal_paths)
            # print(noise_paths)
            workspace = workspace_spin02

        elif field_types == ['E', 'N']:
            # EN, BN
            signal_paths = [pos_she_filemask.format(pos_zbin=zbins[1], she_zbin=zbins[0]), None]
            noise_paths = [pos_she_nl_path.format(pos_zbin=zbins[1], she_zbin=zbins[0]), None]
            workspace = workspace_spin02

        elif field_types == ['E', 'E']:
            # EE, EB, BE, BB
            signal_paths = [she_she_filemask.format(hi_zbin=max(zbins), lo_zbin=min(zbins)), None, None, None]
            noise_paths = [she_nl_path.format(hi_zbin=max(zbins), lo_zbin=min(zbins)),
                           None,
                           None,
                           she_nl_path.format(hi_zbin=max(zbins), lo_zbin=min(zbins))]
            workspace = workspace_spin22

        else:
            raise ValueError(f'Unexpected field type combination: {field_types}')
        # Load in the signal + noise Cls
        print(f'Loading and coupling input Cls {spec_idx + 1} / {len(spectra)}: Loading...')
        uncoupled_theory_cls = load_cls(signal_paths, noise_paths, lmax_in, lmax_in, lmin_in, noise_lmin)
        # if field_types == ['E', 'N']:
        #     print(uncoupled_theory_cls)
        # Apply the "improved NKA" method: couple the theory Cls,
        # then divide by fsky to avoid double-counting the reduction in power
        print(f'Loading and coupling input Cls {spec_idx + 1} / {len(spectra)}: Coupling...')

        coupled_cls = workspace.couple_cell(uncoupled_theory_cls)
        assert np.all(np.isfinite(coupled_cls))
        assert len(coupled_cls) == len(uncoupled_theory_cls)
        coupled_theory_cls.append(np.divide(coupled_cls, fsky))

    # Calculate additional covariance coupling coefficients (independent of spin)
    print(f'Computing covariance coupling coefficients at {time.strftime("%c")}')
    cov_workspace = nmt.NmtCovarianceWorkspace()
    cov_workspace.compute_coupling_coefficients(field_spin2, field_spin2, lmax=lmax_in)

    # Iterate over unique pairs of spectra
    for spec_a_idx, spec_a in enumerate(spectra):
        # Obtain the spins and workspace for the first spectrum
        spin_a1 = spin_from_field(field_1[spec_a_idx])
        spin_a2 = spin_from_field(field_2[spec_a_idx])

        workspace_a = workspace_from_spins(spin_a1, spin_a2, workspace_spin00, workspace_spin02, workspace_spin22)

        for spec_b_idx, spec_b in enumerate(spectra[:(spec_a_idx + 1)]):
            print(f'Calculating covariance block row {spec_a_idx} column {spec_b_idx} at {time.strftime("%c")}')

            # Obtain the spins and workspace for the second spectrum
            spin_b1 = spin_from_field(field_1[spec_b_idx])
            spin_b2 = spin_from_field(field_2[spec_b_idx])
            workspace_b = workspace_from_spins(spin_b1, spin_b2, workspace_spin00, workspace_spin02, workspace_spin22)

            # Identify the four power spectra we need to calculate this covariance

            a1b1 = spectrum_from_fields_3x2pt(field_1[spec_a_idx], zbin_1[spec_a_idx], field_1[spec_b_idx], zbin_1[spec_b_idx])
            a1b2 = spectrum_from_fields_3x2pt(field_1[spec_a_idx], zbin_1[spec_a_idx], field_2[spec_b_idx], zbin_2[spec_b_idx])
            a2b1 = spectrum_from_fields_3x2pt(field_2[spec_a_idx], zbin_2[spec_a_idx], field_1[spec_b_idx], zbin_1[spec_b_idx])
            a2b2 = spectrum_from_fields_3x2pt(field_2[spec_a_idx], zbin_2[spec_a_idx], field_2[spec_b_idx], zbin_2[spec_b_idx])

            # Obtain the corresponding theory Cls
            cl_a1b1 = coupled_theory_cls[spectra.index(a1b1)]
            cl_a1b2 = coupled_theory_cls[spectra.index(a1b2)]
            cl_a2b1 = coupled_theory_cls[spectra.index(a2b1)]
            cl_a2b2 = coupled_theory_cls[spectra.index(a2b2)]
            assert np.all(np.isfinite(cl_a1b1))
            assert np.all(np.isfinite(cl_a1b2))
            assert np.all(np.isfinite(cl_a2b1))
            assert np.all(np.isfinite(cl_a2b2))

            # Evaluate the covariance
            cl_cov = nmt.gaussian_covariance(cov_workspace, spin_a1, spin_a2, spin_b1, spin_b2, cl_a1b1, cl_a1b2,
                                             cl_a2b1, cl_a2b2, workspace_a, workspace_b, coupled=True)

            # Extract the part of the covariance we want, which is conveniently always the [..., 0, ..., 0] block,
            # since all other blocks relate to B-modes
            cl_cov = cl_cov.reshape((lmax_in + 1, len(coupled_theory_cls[spec_a_idx]),
                                     lmax_in + 1, len(coupled_theory_cls[spec_b_idx])))
            cl_cov = cl_cov[:, 0, :, 0]

            if field_1[spec_a_idx] == 'N' and field_2[spec_a_idx] == 'N':
                if field_1[spec_b_idx] == 'E' and field_2[spec_b_idx] == 'E':
                    # NN <-> EE (needs to be cut asymmetrically)
                    lmin_row = lmin_out_nn
                    lmax_row = lmax_out_nn
                    lmin_column = lmin_out_ee
                    lmax_column = lmax_out_ee
                else:
                    # NN <-> NN ; NN <-> NE ; NN <-> EN
                    lmin_row = lmin_out_nn
                    lmax_row = lmax_out_nn
                    lmin_column = lmin_out_nn
                    lmax_column = lmax_out_nn

            elif field_1[spec_a_idx] == 'E' and field_2[spec_a_idx] == 'E':
                if field_1[spec_b_idx] == 'E' and field_2[spec_b_idx] == 'E':
                    # EE <-> EE
                    lmin_row = lmin_out_ee
                    lmax_row = lmax_out_ee
                    lmin_column = lmin_out_ee
                    lmax_column = lmax_out_ee

                else:
                    # EE <-> NN ; EE <-> NE ; EE <-> EN
                    lmin_row = lmin_out_ee
                    lmax_row = lmax_out_ee
                    lmin_column = lmin_out_nn
                    lmax_column = lmax_out_nn

            elif field_1[spec_a_idx] == 'N' and field_2[spec_a_idx] == 'E':
                if field_1[spec_b_idx] == 'E' and field_2[spec_b_idx] == 'E':
                    # NE <-> EE
                    lmin_row = lmin_out_ne
                    lmax_row = lmax_out_ne
                    lmin_column = lmin_out_ee
                    lmax_column = lmax_out_ee
                else:
                    # NE <-> NN ; NE <-> NE
                    lmin_row = lmin_out_ne
                    lmax_row = lmax_out_ne
                    lmin_column = lmin_out_ne
                    lmax_column = lmax_out_ne

            elif field_1[spec_a_idx] == 'E' and field_2[spec_a_idx] == 'N':
                if field_1[spec_b_idx] == 'E' and field_2[spec_b_idx] == 'E':
                    # NE <-> EE
                    lmin_row = lmin_out_ne
                    lmax_row = lmax_out_ne
                    lmin_column = lmin_out_ee
                    lmax_column = lmax_out_ee
                else:
                    # NE <-> NN ; NE <-> NE
                    lmin_row = lmin_out_ne
                    lmax_row = lmax_out_ne
                    lmin_column = lmin_out_ne
                    lmax_column = lmax_out_ne

            else:
                print('Uncollected field types')
                print(field_1[spec_a_idx], field_2[spec_a_idx])
                print(field_1[spec_b_idx], field_2[spec_b_idx])
                sys.exit()

            cl_cov = cl_cov[lmin_row:(lmax_row + 1), lmin_column:(lmax_column + 1)]

            # Do some checks and save to disk
            assert np.all(np.isfinite(cl_cov))
            # n_ell_out = lmax_out - lmin_out + 1
            # assert cl_cov.shape == (n_ell_out, n_ell_out)
            if spec_a_idx == spec_b_idx:
                assert np.allclose(cl_cov, cl_cov.T)

            save_path = save_filemask.format(spec1_idx=spec_a_idx, spec2_idx=spec_b_idx)
            header = (f'Output from {__file__}.get_3x2pt_cov for spectra ({spec_a}, {spec_b}), with parameters '
                      f'n_zbin = {n_zbin}, pos_pos_filemask {pos_pos_filemask}, pos_she_filemask {pos_she_filemask}, '
                      f'she_she_filemask {she_she_filemask}, pos_nl_path = {pos_nl_path},  pos_she_nl_path = {pos_she_nl_path},'
                      f'she_nl_path = {she_nl_path}, lmax_in = {lmax_in}, lmin_in = {lmin_in}, '
                      f'mask_path = {mask_path}, '
                      f'nside = {nside}; time {time.strftime("%c")}')
            print(f'Saving block at {time.strftime("%c")}')
            np.savez_compressed(save_path, cov_block=cl_cov, spec1_idx=spec_a_idx, spec2_idx=spec_b_idx, header=header)
            print(f'Saving {save_path} at {time.strftime("%c")}')

    print(f'Done at {time.strftime("%c")}')


    # Calculate number of spectra and ells
    n_fields = 2 * n_zbin

    n_spec = n_fields * (n_fields + 1) // 2

    # Loop over all numbers of bandpowers

    print(f'Starting n_bp = {n_bp} at {time.strftime("%c")}')

    print(f'Calculating binning matrix at {time.strftime("%c")}')

    assert bandpower_spacing == 'log' or bandpower_spacing == 'lin'

    pbl_ee = mask.get_binning_matrix(
        n_bandpowers=n_bp,
        output_lmin=lmin_out_ee,
        output_lmax=lmax_out_ee,
        bp_spacing=bandpower_spacing)

    pbl_ne = mask.get_binning_matrix(
        n_bandpowers=n_bp,
        output_lmin=lmin_out_ne,
        output_lmax=lmax_out_ne,
        bp_spacing=bandpower_spacing)

    pbl_nn = mask.get_binning_matrix(
        n_bandpowers=n_bp,
        output_lmin=lmin_out_nn,
        output_lmax=lmax_out_nn,
        bp_spacing=bandpower_spacing)

    assert np.alltrue(pbl_nn == pbl_ne)

    print(f'Preallocating full covariance at {time.strftime("%c")}')
    n_data = n_spec * n_bp
    cov = np.full((n_data, n_data), np.nan)

    for spec_a_idx, spec_a in enumerate(spectra):

        for spec_b_idx, spec_b in enumerate(spectra[:(spec_a_idx + 1)]):

            print(f'spec1 = {spec_a_idx}, spec2 = {spec_b_idx} at {time.strftime("%c")}')

            print('Loading block')
            block_path = save_filemask.format(spec1_idx=spec_a_idx, spec2_idx=spec_b_idx)
            with np.load(block_path) as data:
                assert data['spec1_idx'] == spec_a_idx
                assert data['spec2_idx'] == spec_b_idx
                block_unbinned = data['cov_block']
            assert np.all(np.isfinite(block_unbinned))

            print('Binning block')

            if field_1[spec_a_idx] == 'N' and field_2[spec_a_idx] == 'N':
                if field_1[spec_b_idx] == 'E' and field_2[spec_b_idx] == 'E':
                    pbl_a = pbl_nn
                    pbl_b = pbl_ee
                else:
                    pbl_a = pbl_nn
                    pbl_b = pbl_nn

            elif field_1[spec_a_idx] == 'E' and field_2[spec_a_idx] == 'E':
                if field_1[spec_b_idx] == 'E' and field_2[spec_b_idx] == 'E':
                    pbl_a = pbl_ee
                    pbl_b = pbl_ee
                else:
                    pbl_a = pbl_ee
                    pbl_b = pbl_nn

            elif field_1[spec_a_idx] == 'N' and field_2[spec_a_idx] == 'E':
                if field_1[spec_b_idx] == 'E' and field_2[spec_b_idx] == 'E':
                    pbl_a = pbl_nn
                    pbl_b = pbl_ee
                else:
                    pbl_a = pbl_nn
                    pbl_b = pbl_nn

            elif field_1[spec_a_idx] == 'E' and field_2[spec_a_idx] == 'N':
                if field_1[spec_b_idx] == 'E' and field_2[spec_b_idx] == 'E':
                    pbl_a = pbl_nn
                    pbl_b = pbl_ee
                else:
                    pbl_a = pbl_nn
                    pbl_b = pbl_nn

            else:
                print('Error in bandpower binning')
                sys.exit()

            block_binned = pbl_a @ block_unbinned @ pbl_b.T
            assert np.all(np.isfinite(block_binned))
            assert block_binned.shape == (n_bp, n_bp)

            print('Inserting block')
            cov[(spec_a_idx * n_bp):((spec_a_idx + 1) * n_bp), (spec_b_idx * n_bp):((spec_b_idx + 1) * n_bp)] = block_binned

    # Reflect to fill remaining elements, and check symmetric
    cov = np.where(np.isnan(cov), cov.T, cov)
    assert np.all(np.isfinite(cov))
    assert np.allclose(cov, cov.T, atol=0)

    # Save to disk
    save_path = cov_filemask.format(n_bp=n_bp)
    header = (f'Full binned covariance matrix. Output from {__file__}.bin_combine_cov for n_bp = {n_bp}, '
              f'n_zbin = {n_zbin}, lmin_in = {lmin_in}'
              f', at {time.strftime("%c")}')
    np.savez_compressed(save_path, cov=cov, n_bp=n_bp, header=header)
    print(f'Saved {save_path} at {time.strftime("%c")}')
    print()

    print(f'Done at {time.strftime("%c")}')


def get_6x2pt_cov_pcl(n_zbin, signal_path, noise_path, lmax_in, lmin_in, lmax_out_nn, lmin_out_nn, lmax_out_ne,
                       lmin_out_ne, lmax_out_ee,lmin_out_ee, lmax_out_kk, lmin_out_kk, lmax_out_kn, lmin_out_kn,
                       lmax_out_ke, lmin_out_ke, noise_lmin, mask_path, cmb_mask_path, nside, save_filemask, n_bp, bandpower_spacing,
                       cov_filemask):

    """
    Calculate 6x2pt Gaussian covariance using NaMaster, saving each block separately to disk. This function computes
    the covariance associated with a Pseudo-Cl in the absence of noise - i.e. we apply the coupling coefficients of
    a mask to a theoretical (full-sky) Cl (using the coupled=True argument in NaMaster gaussian_covariance). The
    complete covariance for a noisy Pseudo-Cl observation can be calculated by combining this function with
    get_3x2pt_cov_noise - which calculates the Gaussian covariance of a predicted noise spectrum. NOTE - this
    theoretical noise spectrum must predict the noise associated with the region of sky observed (unlike the full-sky
    Cl) since it is NOT combined with the mask coupling coefficients.

    Args:
        n_zbin (int): Number of redshift bins, assuming 1 position field and 1 shear field per redshift bin.
        pos_pos_filemask (str): Path to text file containing a position-position power spectrum with ``{hi_zbin}`` and
                                ``{lo_zbin}`` placeholders.
        pos_she_filemask (str): Path to text file containing a position-shear power spectrum with ``{pos_zbin}`` and
                                ``{she_zbin}`` placeholders.
        she_she_filemask (str): Path to text file containing a shear-shear power spectrum with ``{hi_zbin}`` and
                                ``{lo_zbin}`` placeholders.
        lmax_in (int): Maximum l to including in mixing.
        lmin_in (int): Minimum l in input power spectra.
        lmax_out (int): Maximum l to include in covariance.
        lmin_out (int): Minimum l to include in covariance.
        noise_lmin (int): Minimum l in noise power spectra.
        mask_path (str): Path to mask FITS file. If None, full sky is assumed, in which case the covariance will be
                         diagonal.
        nside (int): HEALPix resolution nside parameter.
        save_filemask (str): Path to save each covariance block to disk, with ``{spec1_idx}`` and ``{spec2_idx}``
                             placeholders.
    """

    # Load and rescale mask, and calculate fsky
    if mask_path is not None:
        print('Loading and rescaling mask')
        lss_mask = hp.pixelfunc.ud_grade(hp.read_map(mask_path, dtype=float), nside)
        assert np.amin(lss_mask) == 0
        assert np.amax(lss_mask) == 1
    else:
        print('Full sky')
        lss_mask = np.ones(hp.pixelfunc.nside2npix(nside))
    assert np.all(np.isfinite(lss_mask))
    fsky = np.mean(lss_mask)
    print(f'fsky = {fsky:.3f}')

    # Load and rescale mask, and calculate fsky
    if cmb_mask_path is not None:
        print('Loading and rescaling mask')
        cmb_mask = hp.pixelfunc.ud_grade(hp.read_map(cmb_mask_path, dtype=float), nside)
        assert np.amin(cmb_mask) == 0
        assert np.amax(cmb_mask) == 1
    else:
        print('Full sky')
        cmb_mask = np.ones(hp.pixelfunc.nside2npix(nside))
    assert np.all(np.isfinite(cmb_mask))
    fsky_cmb = np.mean(cmb_mask)
    print(f'fsky = {fsky_cmb:.3f}')

    assert (lss_mask == cmb_mask).all()
    obs_mask = lss_mask

    # Create NaMaster binning scheme as individual Cls
    print('Creating binning scheme')
    bins = nmt.NmtBin.from_lmax_linear(lmax_in, 1)

    # Calculate mixing matrices for spin 0-0, 0-2 (equivalent to 2-0), and 2-2
    field_spin0 = nmt.NmtField(obs_mask, None, spin=0, lite=True, lmax_sht=lmax_in)
    field_spin2 = nmt.NmtField(obs_mask, None, spin=2, lite=True, lmax_sht=lmax_in)

    workspace_spin00 = nmt.NmtWorkspace()
    print(f'Calculating mixing matrix 1 / 3 at {time.strftime("%c")}')
    workspace_spin00.compute_coupling_matrix(field_spin0, field_spin0, bins)
    workspace_spin02 = nmt.NmtWorkspace()
    print(f'Calculating mixing matrix 2 / 3 at {time.strftime("%c")}')
    workspace_spin02.compute_coupling_matrix(field_spin0, field_spin2, bins)
    workspace_spin22 = nmt.NmtWorkspace()
    print(f'Calculating mixing matrix 3 / 3 at {time.strftime("%c")}')
    workspace_spin22.compute_coupling_matrix(field_spin2, field_spin2, bins)
    assert np.all(np.isfinite(workspace_spin00.get_coupling_matrix()))
    assert np.all(np.isfinite(workspace_spin02.get_coupling_matrix()))
    assert np.all(np.isfinite(workspace_spin22.get_coupling_matrix()))

    # Generate list of fields
    print('Generating list of fields')

    # fields = [f'{f}{z}' for z in range(1, n_zbin + 1) for f in ['N', 'E']]
    # fields.append('K1')
    #
    # spectra = [fields[row] + fields[row + diag] for diag in range(len(fields)) for row in range(len(fields) - diag)]
    # spec_1 = [fields[row] for diag in range(len(fields)) for row in range(len(fields) - diag)]
    # spec_2 = [fields[row + diag] for diag in range(len(fields)) for row in range(len(fields) - diag)]

    # Form list of power spectra
    fields = [f'{f}{z}' for z in range(1, n_zbin + 1) for f in ['N', 'E']]

    spectra = [fields[row] + fields[row + diag] for diag in range((2*n_zbin)) for row in range((2*n_zbin) - diag)]

    spec_1 = [fields[row] for diag in range((2*n_zbin)) for row in range((2*n_zbin) - diag)]
    spec_2 = [fields[row + diag] for diag in range((2*n_zbin)) for row in range((2*n_zbin) - diag)]

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

    # Generate list of target spectra (the ones we want the covariance for) in the correct (diagonal) order
    print('Generating list of spectra')

    field_1 = [mysplit(spec_1_id)[0] for spec_1_id in spec_1]
    zbin_1 = [mysplit(spec_1_id)[1] for spec_1_id in spec_1]
    field_2 = [mysplit(spec_2_id)[0] for spec_2_id in spec_2]
    zbin_2 = [mysplit(spec_2_id)[1] for spec_2_id in spec_2]

    # Generate list of sets of mode-coupled theory Cls corresponding to the target spectra
    coupled_theory_cls = []
    for spec_idx, spec in enumerate(spectra):
        print(f'Loading and coupling input Cls {spec_idx + 1} / {len(spectra)}')

        field_types = [field_1[spec_idx], field_2[spec_idx]]
        zbins = [zbin_1[spec_idx], zbin_2[spec_idx]]

        # Get paths of signal and noise spectra to load
        if field_types == ['N', 'N']:
            # NN only
            signal_paths = [f"{signal_path}/galaxy_cl/bin_{max(zbins)}_{min(zbins)}.txt"]
            noise_paths = [f"{noise_path}/galaxy_cl/bin_{max(zbins)}_{min(zbins)}.txt"]
            workspace = workspace_spin00

        elif field_types == ['E', 'E']:
            # EE, EB, BE, BB
            signal_paths = [f"{signal_path}/shear_cl/bin_{max(zbins)}_{min(zbins)}.txt", None, None, None]
            noise_paths = [f"{noise_path}/shear_cl/bin_{max(zbins)}_{min(zbins)}.txt",
                           None,
                           None,
                           f"{noise_path}/shear_cl/bin_{max(zbins)}_{min(zbins)}.txt"]
            workspace = workspace_spin22

        elif field_types == ['K', 'K']:
            # KCMB
            signal_paths = [f"{signal_path}/cmbkappa_cl/bin_1_1.txt"]
            noise_paths = [f"{noise_path}/cmbkappa_cl/bin_1_1.txt"]
            workspace = workspace_spin00

        elif field_types == ['K', 'N']:
            # KN
            signal_paths = [f"{signal_path}/galaxy_cmbkappa_cl/bin_{zbins[1]}_{zbins[0]}.txt"]
            noise_paths = [f"{noise_path}/galaxy_cmbkappa_cl/bin_{zbins[1]}_{zbins[0]}.txt"]
            workspace = workspace_spin00

        elif field_types == ['N', 'K']:
            # KN
            signal_paths = [f"{signal_path}/galaxy_cmbkappa_cl/bin_{zbins[0]}_{zbins[1]}.txt"]
            noise_paths = [f"{noise_path}/galaxy_cmbkappa_cl/bin_{zbins[0]}_{zbins[1]}.txt"]
            workspace = workspace_spin00

        elif field_types == ['K', 'E']:
            # KE, KB
            signal_paths = [f"{signal_path}/shear_cmbkappa_cl/bin_{zbins[1]}_{zbins[0]}.txt", None]
            noise_paths = [f"{noise_path}/shear_cmbkappa_cl/bin_{zbins[1]}_{zbins[0]}.txt", None]
            workspace = workspace_spin02

        elif field_types == ['E', 'K']:
            # EK, BK
            signal_paths = [f"{signal_path}/shear_cmbkappa_cl/bin_{zbins[0]}_{zbins[1]}.txt", None]
            noise_paths = [f"{noise_path}/shear_cmbkappa_cl/bin_{zbins[0]}_{zbins[1]}.txt", None]
            workspace = workspace_spin02

        elif field_types == ['N', 'E']:
            # NE, NB
            signal_paths = [f"{signal_path}/galaxy_shear_cl/bin_{zbins[0]}_{zbins[1]}.txt", None]
            noise_paths = [f"{noise_path}/galaxy_shear_cl/bin_{zbins[0]}_{zbins[1]}.txt", None]
            workspace = workspace_spin02

        elif field_types == ['E', 'N']:
            # EN, BN
            signal_paths = [f"{signal_path}/galaxy_shear_cl/bin_{zbins[1]}_{zbins[0]}.txt", None]
            noise_paths = [f"{noise_path}/galaxy_shear_cl/bin_{zbins[1]}_{zbins[0]}.txt", None]
            workspace = workspace_spin02

        else:
            raise ValueError(f'Unexpected field combination: {field_types}')
        # Load in the signal + noise Cls
        print(f'Loading and coupling input Cls {spec_idx + 1} / {len(spectra)}: Loading...')
        uncoupled_theory_cls = load_cls(signal_paths, noise_paths, lmax_in, lmax_in, lmin_in, noise_lmin)

        # Apply the "improved NKA" method: couple the theory Cls,
        # then divide by fsky to avoid double-counting the reduction in power
        print(f'Loading and coupling input Cls {spec_idx + 1} / {len(spectra)}: Coupling...')

        coupled_cls = workspace.couple_cell(uncoupled_theory_cls)
        assert np.all(np.isfinite(coupled_cls))
        assert len(coupled_cls) == len(uncoupled_theory_cls)
        coupled_theory_cls.append(np.divide(coupled_cls, fsky))

    # Calculate additional covariance coupling coefficients (independent of spin)
    print(f'Computing covariance coupling coefficients at {time.strftime("%c")}')
    cov_workspace = nmt.NmtCovarianceWorkspace()
    cov_workspace.compute_coupling_coefficients(field_spin2, field_spin2, lmax=lmax_in)

    # Iterate over unique pairs of spectra
    for spec_a_idx, spec_a in enumerate(spectra):

        # Obtain the spins and workspace for the first spectrum
        spin_a1 = spin_from_field(field_1[spec_a_idx])
        spin_a2 = spin_from_field(field_2[spec_a_idx])

        workspace_a = workspace_from_spins(spin_a1, spin_a2, workspace_spin00, workspace_spin02, workspace_spin22)

        for spec_b_idx, spec_b in enumerate(spectra[:(spec_a_idx + 1)]):
            print(f'Calculating covariance block row {spec_a_idx} column {spec_b_idx} at {time.strftime("%c")}')

            # Obtain the spins and workspace for the second spectrum
            spin_b1 = spin_from_field(field_1[spec_b_idx])
            spin_b2 = spin_from_field(field_2[spec_b_idx])
            workspace_b = workspace_from_spins(spin_b1, spin_b2, workspace_spin00, workspace_spin02, workspace_spin22)

            # Identify the four power spectra we need to calculate this covariance
            a1b1 = spectrum_from_fields_6x2pt(field_1[spec_a_idx], zbin_1[spec_a_idx], field_1[spec_b_idx], zbin_1[spec_b_idx])
            a1b2 = spectrum_from_fields_6x2pt(field_1[spec_a_idx], zbin_1[spec_a_idx], field_2[spec_b_idx], zbin_2[spec_b_idx])
            a2b1 = spectrum_from_fields_6x2pt(field_2[spec_a_idx], zbin_2[spec_a_idx], field_1[spec_b_idx], zbin_1[spec_b_idx])
            a2b2 = spectrum_from_fields_6x2pt(field_2[spec_a_idx], zbin_2[spec_a_idx], field_2[spec_b_idx], zbin_2[spec_b_idx])

            # print(spectra)

            # Obtain the corresponding theory Cls
            cl_a1b1 = coupled_theory_cls[spectra.index(a1b1)]
            cl_a1b2 = coupled_theory_cls[spectra.index(a1b2)]
            cl_a2b1 = coupled_theory_cls[spectra.index(a2b1)]
            cl_a2b2 = coupled_theory_cls[spectra.index(a2b2)]

            assert np.all(np.isfinite(cl_a1b1))
            assert np.all(np.isfinite(cl_a1b2))
            assert np.all(np.isfinite(cl_a2b1))
            assert np.all(np.isfinite(cl_a2b2))

            # Evaluate the covariance
            cl_cov = nmt.gaussian_covariance(cov_workspace, spin_a1, spin_a2, spin_b1, spin_b2, cl_a1b1, cl_a1b2,
                                             cl_a2b1, cl_a2b2, workspace_a, workspace_b, coupled=True)
            # print(cl_cov.shape)
            # Extract the part of the covariance we want, which is conveniently always the [..., 0, ..., 0] block,
            # since all other blocks relate to B-modes
            # print(coupled_theory_cls[spec_a_idx])
            # cl_cov = cl_cov.reshape((lmax_in + 1, len(coupled_theory_cls[spec_a_idx]),
            #                          lmax_in + 1, len(coupled_theory_cls[spec_b_idx])))
            cl_cov = cl_cov.reshape((lmax_in + 1, maths.factorial(spin_a1) * maths.factorial(spin_a2),
                                     lmax_in + 1, maths.factorial(spin_b1) * maths.factorial(spin_b2)))
            # print(cl_cov.shape)
            cl_cov = cl_cov[:, 0, :, 0]

            # def lcut_from_fields(field_1_a, field_2_a, field_1_b, field_2_b):

            fa = f"{field_1[spec_a_idx]}{field_2[spec_a_idx]}"
            fb = f"{field_1[spec_b_idx]}{field_2[spec_b_idx]}"

            lcut_dict = {'NN': [lmin_out_nn, lmax_out_nn],
                         'NE': [lmin_out_ne, lmax_out_ne],
                         'NK': [lmin_out_kn, lmax_out_kn],
                         'EE': [lmin_out_ee, lmax_out_ee],
                         'EN': [lmin_out_ne, lmax_out_ne],
                         'EK': [lmin_out_ke, lmax_out_ke],
                         'KK': [lmin_out_kk, lmax_out_kk],
                         'KN': [lmin_out_kn, lmax_out_kn],
                         'KE': [lmin_out_ke, lmax_out_ke]
                         }

            lmin_row, lmax_row = lcut_dict[fa]
            lmin_column, lmax_column = lcut_dict[fb]

            cl_cov = cl_cov[lmin_row:(lmax_row + 1), lmin_column:(lmax_column + 1)]

            # Do some checks and save to disk
            assert np.all(np.isfinite(cl_cov))
            # n_ell_out = lmax_out - lmin_out + 1
            # assert cl_cov.shape == (n_ell_out, n_ell_out)
            if spec_a_idx == spec_b_idx:
                assert np.allclose(cl_cov, cl_cov.T)

            save_path = save_filemask.format(spec1_idx=spec_a_idx, spec2_idx=spec_b_idx)
            header = (f'Output from {__file__}.get_3x2pt_cov for spectra ({spec_a}, {spec_b}), with parameters '
                      f'n_zbin = {n_zbin}, lmax_in = {lmax_in}, lmin_in = {lmin_in}, '
                      f'lmax_out_row = {lmax_row}, lmin_out_row = {lmin_row}, '
                      f'lmax_out_column = {lmax_column}, lmin_out_column = {lmin_column}, '
                      f'mask_path = {mask_path}, '
                      f'nside = {nside}; time {time.strftime("%c")}')
            print(f'Saving block at {time.strftime("%c")}')
            np.savez_compressed(save_path, cov_block=cl_cov, spec1_idx=spec_a_idx, spec2_idx=spec_b_idx,
                                header=header)
            print(f'Saving {save_path} at {time.strftime("%c")}')

        print(f'Binning at {time.strftime("%c")}')

    # Calculate number of spectra and ells
    # n_fields = 2 * n_zbin

    n_spec = (2 * (n_zbin**2)) + (3*n_zbin) + 1

    # Loop over all numbers of bandpowers

    print(f'Starting n_bp = {n_bp} at {time.strftime("%c")}')

    print(f'Calculating binning matrix at {time.strftime("%c")}')

    assert bandpower_spacing == 'log' or bandpower_spacing == 'lin'

    pbl_ee = mask.get_binning_matrix(
        n_bandpowers=n_bp,
        output_lmin=lmin_out_ee,
        output_lmax=lmax_out_ee,
        bp_spacing=bandpower_spacing)

    pbl_ne = mask.get_binning_matrix(
        n_bandpowers=n_bp,
        output_lmin=lmin_out_ne,
        output_lmax=lmax_out_ne,
        bp_spacing=bandpower_spacing)

    pbl_nn = mask.get_binning_matrix(
        n_bandpowers=n_bp,
        output_lmin=lmin_out_nn,
        output_lmax=lmax_out_nn,
        bp_spacing=bandpower_spacing)

    pbl_kk = mask.get_binning_matrix(
        n_bandpowers=n_bp,
        output_lmin=lmin_out_kk,
        output_lmax=lmax_out_kk,
        bp_spacing=bandpower_spacing)

    pbl_ke = mask.get_binning_matrix(
        n_bandpowers=n_bp,
        output_lmin=lmin_out_ke,
        output_lmax=lmax_out_ke,
        bp_spacing=bandpower_spacing)

    pbl_kn = mask.get_binning_matrix(
        n_bandpowers=n_bp,
        output_lmin=lmin_out_kn,
        output_lmax=lmax_out_kn,
        bp_spacing=bandpower_spacing)

    assert np.alltrue(pbl_nn == pbl_ne)

    print(f'Preallocating full covariance at {time.strftime("%c")}')
    n_data = n_spec * n_bp
    cov = np.full((n_data, n_data), np.nan)

    for spec_a_idx, spec_a in enumerate(spectra):

        for spec_b_idx, spec_b in enumerate(spectra[:(spec_a_idx + 1)]):

            print(f'spec1 = {spec_a_idx}, spec2 = {spec_b_idx} at {time.strftime("%c")}')

            print('Loading block')
            block_path = save_filemask.format(spec1_idx=spec_a_idx, spec2_idx=spec_b_idx)
            with np.load(block_path) as data:
                assert data['spec1_idx'] == spec_a_idx
                assert data['spec2_idx'] == spec_b_idx
                block_unbinned = data['cov_block']
            assert np.all(np.isfinite(block_unbinned))

            print('Binning block')

            pbl_fa = f"{field_1[spec_a_idx]}{field_2[spec_a_idx]}"
            pbl_fb = f"{field_1[spec_b_idx]}{field_2[spec_b_idx]}"

            pbl_dict = {'NN': pbl_nn,
                         'NE': pbl_ne,
                         'NK': pbl_kn,
                         'EE': pbl_ee,
                         'EN': pbl_ne,
                         'EK': pbl_ke,
                         'KK': pbl_kk,
                         'KN': pbl_kn,
                         'KE': pbl_ke
                         }

            pbl_a = pbl_dict[pbl_fa]
            pbl_b = pbl_dict[pbl_fb]

            # if field_1[spec_a_idx] == 'N' and field_2[spec_a_idx] == 'N':
            #     if field_1[spec_b_idx] == 'E' and field_2[spec_b_idx] == 'E':
            #         pbl_a = pbl_nn
            #         pbl_b = pbl_ee
            #     else:
            #         pbl_a = pbl_nn
            #         pbl_b = pbl_nn
            #
            # elif field_1[spec_a_idx] == 'E' and field_2[spec_a_idx] == 'E':
            #     if field_1[spec_b_idx] == 'E' and field_2[spec_b_idx] == 'E':
            #         pbl_a = pbl_ee
            #         pbl_b = pbl_ee
            #     else:
            #         pbl_a = pbl_ee
            #         pbl_b = pbl_nn
            #
            # elif field_1[spec_a_idx] == 'N' and field_2[spec_a_idx] == 'E':
            #     if field_1[spec_b_idx] == 'E' and field_2[spec_b_idx] == 'E':
            #         pbl_a = pbl_nn
            #         pbl_b = pbl_ee
            #     else:
            #         pbl_a = pbl_nn
            #         pbl_b = pbl_nn
            #
            # elif field_1[spec_a_idx] == 'E' and field_2[spec_a_idx] == 'N':
            #     if field_1[spec_b_idx] == 'E' and field_2[spec_b_idx] == 'E':
            #         pbl_a = pbl_nn
            #         pbl_b = pbl_ee
            #     else:
            #         pbl_a = pbl_nn
            #         pbl_b = pbl_nn
            #
            # else:
            #     print('error here')
            #     sys.exit()

            block_binned = pbl_a @ block_unbinned @ pbl_b.T
            assert np.all(np.isfinite(block_binned))
            assert block_binned.shape == (n_bp, n_bp)

            print('Inserting block')
            cov[(spec_a_idx * n_bp):((spec_a_idx + 1) * n_bp), (spec_b_idx * n_bp):((spec_b_idx + 1) * n_bp)] = block_binned

    # Reflect to fill remaining elements, and check symmetric
    cov = np.where(np.isnan(cov), cov.T, cov)
    assert np.all(np.isfinite(cov))
    assert np.allclose(cov, cov.T, atol=0)

    # Save to disk
    save_path = cov_filemask.format(n_bp=n_bp)
    header = (f'Full binned covariance matrix. Output from {__file__}.bin_combine_cov for n_bp = {n_bp}, '
              f'n_zbin = {n_zbin}, lmin_in = {lmin_in}'
              f', at {time.strftime("%c")}')
    np.savez_compressed(save_path, cov=cov, n_bp=n_bp, header=header)
    print(f'Saved {save_path} at {time.strftime("%c")}')
    print()

    print(f'Done at {time.strftime("%c")}')




    #
    # if field not in {'E', 'N', 'K', 'EK', 'NK'}:
    #     print(f'Unknown field type - {field}')
    #     sys.exit()
    #
    # if field == 'K':
    #     n_fields = 1
    #     n_spec = 1
    #
    # elif field == 'EK' or field == 'NK':
    #     n_fields = n_zbin
    #     n_spec = n_zbin
    #
    # else:
    #     assert field == 'E' or field == 'N'
    #     n_fields = n_zbin
    #     n_spec = n_fields * (n_fields + 1) // 2
    #
    # n_ell_cov = lmax_out - lmin_out + 1
    #
    # print(f'Calculating binning matrix at {time.strftime("%c")}')
    #
    # assert pbl.shape == (n_bp, n_ell_cov)
    #
    # print(f'Preallocating full covariance at {time.strftime("%c")}')
    # n_data = n_spec * n_bp
    # cov = np.full((n_data, n_data), np.nan)
    #
    # # Loop over all blocks
    # for spec1 in range(n_spec):
    #     for spec2 in range(spec1 + 1):
    #         print(f'spec1 = {spec1}, spec2 = {spec2} at {time.strftime("%c")}')
    #
    #         print('Loading block')
    #         block_path = save_block_filemask.format(spec1_idx=spec1, spec2_idx=spec2)
    #         with np.load(block_path) as data:
    #             assert data['spec1_idx'] == spec1
    #             assert data['spec2_idx'] == spec2
    #             block_unbinned = data['cov_block']
    #         assert np.all(np.isfinite(block_unbinned))
    #         assert block_unbinned.shape == (n_ell_cov, n_ell_cov)
    #         # lowl_skip = lmin_out - lmin_in
    #         # block_unbinned = block_unbinned[lowl_skip:, lowl_skip:]
    #         # assert block_unbi/nned.shape == (n_ell_out, n_ell_out)
    #
    #         print('Binning block')
    #         block_binned = pbl @ block_unbinned @ pbl.T
    #         assert np.all(np.isfinite(block_binned))
    #         assert block_binned.shape == (n_bp, n_bp)
    #
    #         print('Inserting block')
    #         cov[(spec1 * n_bp):((spec1 + 1) * n_bp), (spec2 * n_bp):((spec2 + 1) * n_bp)] = block_binned
    #
    # # Reflect to fill remaining elements, and check symmetric
    # cov = np.where(np.isnan(cov), cov.T, cov)
    # assert np.all(np.isfinite(cov))
    # assert np.allclose(cov, cov.T, atol=0)
    #
    # # Save to disk
    # save_path = save_cov_filemask.format(n_bp=n_bp)
    # header = (f'Full binned covariance matrix. Output from {__file__}.bin_combine_cov for n_bp = {n_bp}, '
    #           f'n_zbin = {n_zbin}, lmin_in = {lmin_in}, lmin_out = {lmin_out}, lmax_out = {lmax_out}, '
    #           f'save_block_filemask = {save_block_filemask},'
    #           f'save_cov_filemask = {save_cov_filemask}, at {time.strftime("%c")}')
    # np.savez_compressed(save_path, cov=cov, n_bp=n_bp, header=header)
    # print(f'Saved {save_path} at {time.strftime("%c")}')
    # print()
    #
    # print(f'Done at {time.strftime("%c")}')

