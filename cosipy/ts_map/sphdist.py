# sphdist.py
#
# Small, self-contained utilities for spherical geometry in the HEALPix / healpy
# convention:
#   - theta is colatitude in [0, pi] (0 at North pole, pi at South pole)
#   - phi   is longitude  in [0, 2*pi)
#
# The “distance” computed here is the great-circle angular separation on the unit
# sphere, returned in radians.

import numpy as np
import mhealpy as hp

def angdist_vec(v1, v2):
    """
    Great-circle angular distance between two directions represented as vectors.
    v1 is a single vector of shape (3,); v2 may be a single vector of shape (3,)
    or a collection of N vectors of shape (3,N)

    Assumes v1 and v2 are unit vectors. The angular separation d is
    computed via the numerically robust identity:
        d = atan2(||v1 × v2||, v1 · v2)

    This is preferable to arccos(v1·v2) because arccos is ill-conditioned when
    d is very small (near 0) or very close to pi; atan2 remains stable in both
    regimes. 
    
    I found this formula at the following matlab forum:
    https://uk.mathworks.com/matlabcentral/answers/101590-how-can-i-determine-the-angle-between-two-vectors-in-matlab#answer_185622

    Parameters
    ----------
    v1 : array-like, shape (3,)
    v2 : array-like, shape (3,) or (3,N)
        3D vectors representing directions on the sphere (ideally unit length).

    Returns
    -------
    array of float, length N
        Angular separation in radians, in [0, pi].
    """

    # Dot product gives cos(γ) for unit vectors.
    dot = v1 @ v2

    # Numerical safety
    dot = np.clip(dot, -1.0, 1.0)

    # Norm of cross product gives sin(γ) for unit vectors.
    cross_norm = np.linalg.norm(np.cross(v1, v2.T), axis=1)

    # atan2(sin(γ), cos(γ)) returns γ in a stable way over the full range.
    return np.arctan2(cross_norm, dot)


def angdist_pix_tap(theta0, phi0, pix_ID, NSIDE, nest: bool = False):
    """
    Angular distance A_i(o) between a TAP origin o and the centre direction of pixel h_i.

    Implements the piecewise definition:
        A_i(o) = 0            if o ∈ h_i
               = d(o, c_i)    otherwise
    where:
      - o is the unit vector corresponding to (theta0, phi0),
      - c_i is the unit vector of the centre of pixel pix_ID,
      - d(·,·) is the great-circle angular distance (via angdist_vec).

    We interpret "o ∈ h_i" via healpy's unique pixel assignment:
        containing = hp.ang2pix(NSIDE, theta0, phi0, nest=nest)

    Parameters
    ----------
    theta0, phi0 : float
        TAP coordinates in healpy convention: theta0=colatitude, phi0=longitude (radians).
    pix_ID : int or array of int
        HEALPix pixel indices
    NSIDE : int or array of int
        HEALPix NSIDE parameter (resolution); if not scalar, one value per pixel index
    nest : bool, optional
        Pixel ordering flag (False=RING, True=NEST). Must match how pix_ID is interpreted.

    Returns
    -------
    array of float
        A_i(o) in radians per pix_ID, in [0, pi].
    """

    # Convert TAP angles to a unit vector [1x3]
    v_tap = hp.ang2vec(theta0, phi0, lonlat=False)

    # Get the unit vector of the centre of pixel pix_ID
    # (tuple of x, y, z vectors)
    v_pix = hp.pix2vec(NSIDE, pix_ID, nest=nest)
    v_pix = np.vstack(v_pix) # [3xN]
    
    # Compute the great-circle angular distance between the two directions
    ad = angdist_vec(v_tap, v_pix)

    # Identify the pixel that contains the TAP direction
    containing_pixel = hp.ang2pix(NSIDE, theta0, phi0, lonlat=False, nest=nest)

    # If pix_ID is the containing pixel, distance is defined as 0
    ad[pix_ID == containing_pixel] = 0.0

    return ad


def expected_angdist(theta0, phi0, p_map, NSIDE, NEST):
    """
    Expected angular distance E_o = Σ_i p(i) A_i(o), where A_i(o) is implemented
    by angdist_pix_tap(·) above.

    Parameters
    ----------
    theta0, phi0 : float
        TAP coordinates (radians) in healpy convention: theta0=colatitude, phi0=longitude.
    p_map : array-like
        Probability mass function over pixels: p_map[i] = p(i), length NPIX.
        Should sum to 1.
    NSIDE : int 
        HEALPix NSIDE parameter.
    NEST : bool
        Pixel ordering flag (False=RING, True=NEST).

    Returns
    -------
    float
        Expected angular distance in radians.
    """
    p_map = np.asarray(p_map)

    # Basic sanity: p_map must have one entry per pixel for the given NSIDE
    NPIX = hp.nside2npix(NSIDE)
    if p_map.shape[0] != NPIX:
        raise ValueError(f"p_map length {p_map.shape[0]} does not match NPIX={NPIX} for NSIDE={NSIDE}")

    E = np.sum(p_map * angdist_pix_tap(theta0, phi0, np.arange(NPIX), NSIDE=NSIDE, nest=NEST))

    return E


def moc_expected_angdist(theta0, phi0, m_pix, m_probs):
    """
    Expected angular distance E_o = Σ_i p(i) A_i(o) for
    multiresolution map, where A_i(o) is implemented by
    angdist_pix_tap(·) above.

    Parameters
    ----------
    theta0, phi0 : float
        TAP coordinates (radians) in healpy convention: theta0=colatitude, phi0=longitude.
    m_pix : array-like of int
        unique pixel IDs from MOC map
    m_probs : 
        Probability mass function over unique pixels: p_map[i] = p(i), length |m_pix|.
        Should sum to 1.

    Returns
    -------
    float
        Expected angular distance in radians.

    """
    
    nside, pix = hp.uniq2nest(m_pix)
    E = np.sum(m_probs * angdist_pix_tap(theta0, phi0, pix, NSIDE=nside, nest=True))
        
    return E
