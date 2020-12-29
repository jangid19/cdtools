"""This module contains tools to simulate various measurement models

All the measurements here are safe to use in an automatic differentiation
model. There exist tools to simulate detectors with finite saturation
thresholds, backgrounds, and more.
"""

from __future__ import division, print_function, absolute_import

from CDTools.tools import cmath
import torch as t
import numpy as np
from torch.nn.functional import avg_pool2d

#
# This file will host tools to turn a propagated wavefield into a measured
# intensity pattern on a detector
#

__all__ = ['intensity', 'incoherent_sum', 'quadratic_background']


def intensity(wavefield, detector_slice=None, epsilon=1e-7, saturation=None, oversampling=1):
    """Returns the intensity of a wavefield
    
    The intensity is defined as the magnitude squared of the
    wavefront. If a detector slice is given, the returned array
    will only include that slice from the simulated wavefront.
    
    Parameters
    ----------
    wavefield : torch.Tensor
        A JxMxNx2 stack of complex wavefields
    detector_slice : slice
        Optional, a slice or tuple of slices defining a section of the simulation to return
    saturation : float
        Optional, a maximum saturation value to clamp the resulting intensities to
    oversampling : int
        Default 1, the width of the region pixels in the wavefield to bin into a single detector pixel

    Returns
    -------
    sim_patterns : torch.Tensor
        A real MxN array storing the wavefield's intensities
    """
    output = cmath.cabssq(wavefield)

    # Now we apply oversampling
    if oversampling != 1:
        if wavefield.dim() == 3:
            output = avg_pool2d(output.unsqueeze(0), oversampling)[0]
        else:
            output = avg_pool2d(output, oversampling)

    # Then we grab the detector slice
    if detector_slice is not None:
        if wavefield.dim() == 3:
            output = output[detector_slice]
        else:
            output = output[(np.s_[:],) + detector_slice]

    # And now saturation    
    if saturation is None:
        return output + epsilon
    else:
        return t.clamp(output + epsilon,0,saturation)


def incoherent_sum(wavefields, detector_slice=None, epsilon=1e-7, saturation=None, oversampling=1):
    """Returns the incoherent sum of the intensities of the wavefields
    
    The intensity is defined as the sum of the magnitudes squared of
    the wavefields. If a detector slice is given, the returned array
    will only include that slice from the simulated wavefronts.

    The first index is the set of incoherently adding patterns, and
    the second index is the index of the diffraction pattern to measure.
    The next two indices index the wavefield. The final index is the complex
    index.
    
    Parameters
    ----------
    wavefields : torch.Tensor
        An LxJxMxNx2 stack of complex wavefields
    detector_slice : slice
        Optional, a slice or tuple of slices defining a section of the simulation to return
    saturation : float
        Optional, a maximum saturation value to clamp the resulting intensities to
    oversampling : int
        Default 1, the width of the region pixels in the wavefield to bin into a single detector pixel

    Returns
    -------
    sim_patterns : torch.Tensor 
        A real JXMxN array storing the incoherently summed intensities
    """
    # This syntax just adds an axis to the slice to preserve the J direction

    output = t.sum(cmath.cabssq(wavefields),dim=0) 

    # Now we apply oversampling
    if oversampling != 1:
        if wavefields.dim() == 4:
            output = avg_pool2d(output.unsqueeze(0), oversampling)[0]
        else:
            output = avg_pool2d(output, oversampling)
            
    # Then we grab the detector slice
    if detector_slice is not None:
        if wavefields.dim() == 4:
            output = output[detector_slice]
        else:
            output = output[(np.s_[:],) + detector_slice]

    if saturation is None:
        return output + epsilon
    else:
        return t.clamp(output + epsilon,0,saturation)


def quadratic_background(wavefield, background, detector_slice=None, measurement=intensity, epsilon=1e-7, saturation=None, oversampling=1):
    """Returns the intensity of a wavefield plus a background
    
    The intensity is calculated via the given measurment function 
    Then, the square of the given background is added. This kind
    of background model is commonly used to enforce positivity of the
    background model.

    Parameters
    ----------
    wavefield : torch.Tensor
        A JxMxNx2 stack of complex wavefields
    background : torch.Tensor
        An tensor storing the square root of the detector background
    detector_slice : slice
        Optional, a slice or tuple of slices defining a section of the simulation to return
    measurement : function
        Default is measurements.intensity, the measurement function to use.
    saturation : float
        Optional, a maximum saturation value to clamp the resulting intensities to
    oversampling : int
        Default 1, the width of the region pixels in the wavefield to bin into a single detector pixel

    Returns
    -------
    sim_patterns : torch.Tensor
        A real MxN array storing the wavefield's intensities
    """
    
    if detector_slice is None:
        output = measurement(wavefield, epsilon=epsilon,
                             oversampling=oversampling) + background**2
    else:
        output = measurement(wavefield, detector_slice,
                             epsilon=epsilon, oversampling=oversampling) \
                             + background**2

    # This has to be done after the background is added, hence we replicate
    # it here
    if saturation is None:
        return output
    else:
        return t.clamp(output,0,saturation)