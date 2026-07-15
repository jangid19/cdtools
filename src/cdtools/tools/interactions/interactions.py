""" This module contains various to simulate stages in the probe-sample interaction

All the tools here are designed to work with automatic differentiation. Each
function simulates some aspect of an interaction model that can be used
for ptychographic reconstruction.
"""

import torch as t
import numpy as np
from cdtools.tools import propagators, image_processing

__all__ = ['translations_to_pixel', 'pixel_to_translations',
           'project_translations_to_sample',
           'ptycho_2D_round','ptycho_2D_linear','ptycho_2D_sinc',
           'RPI_interaction']


def translations_to_pixel(basis, translations, surface_normal=t.Tensor([0.,0.,1.])):
    """Takes real space translations and outputs them in pixel space

    This works for any 2D ptychography geometry. It takes in
    A set of translations in (x,y) space and outputs the same translations
    in internal pixel units perpendicular to the detector.

    It uses information on the wavefield basis and, if defined, the
    sample normal, to perform the conversion.

    The assumed geometry is incoming radiation with a wavevector parallel
    to the +z axis, [0,0,1]. The default sample orientation has a surface
    normal parallel to this direction

    Parameters
    ----------
    basis : torch.Tensor
        The real space basis the wavefields are defined in
    translations : torch.Tensor
        A Jx3 stack of real-space translations, or a single translation
    surface_normal : torch.Tensor
        Optional, the sample's surface normal

    Returns
    -------
    pixel_translations : torch.Tensor
        A Jx2 stack of translations, or a single translation, in (i,j) pixel-space
    """
    projection_1 = t.as_tensor(np.array([[1,0,0],
                                         [0,1,0],
                                         [0,0,0]]),
                               device=translations.device,
                               dtype=translations.dtype)
    projection_2 = projection_1.clone()
    projection_2[2] = t.as_tensor(-surface_normal / surface_normal[2],
                                  dtype=projection_2.dtype,
                                  device=projection_2.device)
    projection_2 = t.linalg.inv(projection_2)
    

    basis_vectors_inv = t.pinverse(basis).to(device=translations.device,
                                             dtype=translations.dtype)
    projection = t.mm(basis_vectors_inv,
                      t.mm(projection_2,projection_1))
    projection = projection.t()

    single_translation = False
    if len(translations.shape) == 1:
        translations = translations[None,:]
        single_translation = True

    pixel_translations = t.mm(translations, projection)

    if single_translation:
        return pixel_translations[0]
    else:
        return pixel_translations


def pixel_to_translations(basis, pixel_translations, surface_normal=t.Tensor([0,0,1])):
    """Takes pixel-space translations and outputs them in real space

    This works for any 2D ptychography geometry. It takes in
    A set of internal pixel unit translations in (i,j) space and
    outputs the same translations real (x,y) space

    It uses information on the wavefield basis and, if defined, the
    sample normal, to perform the conversion.

    The assumed geometry is incoming radiation with a wavevector parallel
    to the +z axis, [0,0,1]. The default sample orientation has a surface
    normal parallel to this direction. Because of this, the z direction
    translation is always set to zero in the conversion

    Parameters
    ----------
    basis : torch.Tensor
        The real space basis the wavefields are defined in
    translations : torch.Tensor
        A Jx2 stack of pixel-space translations, or a single translation
    surface_normal : torch.Tensor, default: torch.Tensor([0,0,1])
        The sample's surface normal

    Returns
    -------
    real_translations : torch.Tensor
        A Jx3 stack of real-space translations, or a single translation
    """
    projection_1 = t.Tensor([[1,0,0],
                             [0,1,0],
                             [0,0,0]]).to(device=basis.device,dtype=basis.dtype)
    projection_2 = t.inverse(t.Tensor([[1,0,0],
                                       [0,1,0],
                                       -surface_normal/
                                       surface_normal[2]])).to(device=basis.device,dtype=basis.dtype)
    basis_vectors_inv = t.pinverse(basis)
    projection = t.mm(basis_vectors_inv,
                      t.mm(projection_2,projection_1))
    # Literally just need the pseudoinverse of the projection we used to go
    # the other way
    projection = t.pinverse(projection).t()

    single_translation = False
    if len(pixel_translations.shape) == 1:
        pixel_translations = pixel_translations[None,:]
        single_translation = True

    translations = t.mm(pixel_translations, projection)

    if single_translation:
        return translations[0]
    else:
        return translations



def project_translations_to_sample(sample_basis, translations):
    """Takes real space translations and outputs them in pixels in a sample basis

    This projection function is designed for the Bragg2DPtycho class. More
    broadly, it works to take a set of translations in the lab frame and
    convert each one into two values. First, an (i,j) value in pixels
    describing the location of the probe's intersection with the sample
    plane, assuming the basis found in sample_basis is used. Second, an
    amount that the probe must be propagated along the z-axis to reach
    the sample plane at the given location. This includes both the effect of
    the tilted sample plane and any explicitly defined motion along the z axis
    of the probe-forming optic as included in the input translations.


    Note that because of the sign convention (that this function returns th
    relative amount the probe needs to be propagated to reach any given
    location), a positive motion along the z-axis of the probe forming optics
    will lead to a negative propagation distance.

    The assumed geometry is incoming radiation with a wavevector parallel
    to the +z axis, [0,0,1].

    Parameters
    ----------
    sample_basis : torch.Tensor
        The real space basis the wavefields are defined in
    translations : torch.Tensor
        A Jx3 stack of real-space translations, or a single translation

    Returns
    -------
    pixel_translations : torch.Tensor
        A Jx2 stack of translations in internal (i,j) pixel-space, or a single translation
    propagations : torch.Tensor
        A length-J vector of propagation distances, in meters
    """

    # Must do this all in pytorch unfortunately
    # First we calculate the surface normal for the projection onto the sample
    surface_normal =  t.cross(sample_basis[:,1],sample_basis[:,0], dim=-1)
    surface_normal /= t.norm(surface_normal)


    # Then we calculate a matrix which can do the projection

    propagation_dir = t.Tensor(np.array([0,0,1])).to(
        device=surface_normal.device,
        dtype=surface_normal.dtype)

    I = t.eye(3).to(
        device=surface_normal.device,
        dtype=surface_normal.dtype)

    # Here we're setting up a matrix-vector equation mat*answer=input
    # At some point ger will need to be replaced by outer, but for now
    # outer many places still don't have new enough versions of torch.
    mat = t.cat((I - t.ger(propagation_dir,propagation_dir),
                 surface_normal.unsqueeze(0)))

    # And we invert the matrix to do the projection
    projector = t.pinverse(mat)[:,:3].to(device=translations.device,
                                         dtype=translations.dtype)

    # Finally, we need to get the result in the right basis, so this will
    # do that conversion for us
    basis_vectors_inv = t.pinverse(sample_basis).to(device=translations.device,
                                                    dtype=translations.dtype)

    # Same for the propagation direction to extract how far it must propagate
    propagation_dir_inv = propagation_dir.unsqueeze(0).to(
        device=translations.device,
        dtype=translations.dtype)


    sample_projection = t.mm(basis_vectors_inv, projector).t()
    prop_projection = t.mm(propagation_dir_inv, projector).t()


    single_translation = False
    if len(translations.shape) == 1:
        translations = translations[None,:]
        single_translation = True

    pixel_translations = t.mm(translations, sample_projection)
    propagations = t.mm(translations, prop_projection) \
        - t.mm(translations,propagation_dir[:,None])

    if single_translation:
        return pixel_translations[0], propagations[0]
    else:
        return pixel_translations, propagations




def ptycho_2D_round(probe, obj, translations, multiple_modes=False, upsample_obj=False):
    """Returns a stack of exit waves without accounting for subpixel shifts

    This function returns a collection of exit waves, with the first
    dimension as the translation index and the final dimensions
    corresponding to the detector. The exit waves are calculated by
    shifting the probe by the rounded value of the translation

    If multiple_modes is set to False, any additional dimensions in the
    ptycho_2D_round function will be assumed to correspond to the translation
    index. If multiple_modes is set to true, the (-4th) dimension of the probe
    will always be assumed to be defining a set of (P) incoherently mixing
    modes to be broadcast all translation indices. If any additional dimensions
    closer to the start exist, they will be assumed to be translation indices


    Parameters
    ----------
    probe : torch.Tensor
        A (P)xMxL probe function to illuminate the object
    object : torch.Tensor
        The object function to be probed
    translations : torch.Tensor
        The (N)x2 array of (i,j) translations to simulate
    multuple_modes : bool
        Default False, whether to assume the probe contains multiple modes

    Returns
    -------
    exit_waves : torch.Tensor
        An (N)x(P)xMxL tensor of the calculated exit waves
    """

    single_translation = False
    if translations.dim() == 1:
        translations = translations[None,:]
        single_translation = True


    integer_translations = t.round(translations).to(dtype=t.int32)

    if upsample_obj:
        selections = t.stack([obj[tr[0]:tr[0]+probe.shape[-2]//2,
                                  tr[1]:tr[1]+probe.shape[-1]//2]
                              for tr in integer_translations])
        selections = image_processing.fourier_upsample(selections,
                                                       preserve_mean=True)

    else:
        selections = t.stack([obj[tr[0]:tr[0]+probe.shape[-2],
                              tr[1]:tr[1]+probe.shape[-1]]
                          for tr in integer_translations])


    if multiple_modes:
        # if the probe dimension is 4, then this hasn't yet been broadcast
        # over the translation dimensions
        output = probe * selections[:,None,:,:]
    else:
        output = probe * selections

    if single_translation:
        return output[0]
    else:
        return output



def ptycho_2D_linear(probe, obj, translations, shift_probe=True):
    """Returns a stack of exit waves accounting for subpixel shifts

    This function returns a collection of exit waves, with the first
    dimension as the translation index and the final dimensions
    corresponding to the detector. The exit waves are calculated by
    shifting the probe with each translation in turn, using linear
    interpolation to combine the results

    If shift_probe is True, it applies the subpixel shift to the probe,
    otherwise the subpixel shift is applied to the object

    Parameters
    ----------
    probe : torch.Tensor
        An MxL probe function for the exit waves
    object : torch.Tensor
        The object function to be probed
    translations : torch.Tensor
        The Nx2 array of translations to simulate
    shift_probe : bool
        Default True, Whether to subpixel shift the probe or object

    Returns
    -------
    exit_waves : torch.Tensor
        An NxMxL tensor of the calculated exit waves
    """
    single_translation = False
    if translations.dim() == 1:
        translations = translations[None,:]
        single_translation = True

    # Separate the translations into a part that chooses the window
    # And a part that defines the windowing function
    integer_translations = t.floor(translations)
    subpixel_translations = translations - integer_translations
    integer_translations = integer_translations.to(dtype=t.int32)

    exit_waves = []
    if shift_probe:
        for tr, sp in zip(integer_translations,
                          subpixel_translations):
            # This isn't perfectly symmetric but I think it's okay for now
            # It should get the job done
            # Basically, we shift the probe's position by a subpixel (i,j),
            # rolling the edges of the array, and use that to multiply
            # by the object
            sel00 = probe[:,:]
            sel01 = t.cat((probe[:,-1:],probe[:,:-1]),dim=1)
            sel10 = t.cat((probe[-1:,:],probe[:-1,:]),dim=0)
            sel11 = t.cat((sel01[-1:,:],sel01[:-1,:]),dim=0)

            selection = sel00 * (1-sp[0])*(1-sp[1]) + \
                sel10 * sp[0]*(1-sp[1]) + \
                sel01 * (1-sp[0])*sp[1] + \
                sel11 * sp[0]*sp[1]

            obj_slice = obj[tr[0]:tr[0]+probe.shape[0],
                            tr[1]:tr[1]+probe.shape[1]]

            exit_waves.append(selection * obj_slice)
    else:
        for tr, sp in zip(integer_translations,
                          subpixel_translations):
            #
            # Here we subpixel shift the object by (-i,-j) after
            # slicing out the correct translation of the probe
            #

            sel00 = obj[tr[0]:tr[0]+probe.shape[0],
                        tr[1]:tr[1]+probe.shape[1]]

            sel01 = obj[tr[0]:tr[0]+probe.shape[0],
                        tr[1]+1:tr[1]+1+probe.shape[1]]

            sel10 = obj[tr[0]+1:tr[0]+1+probe.shape[0],
                        tr[1]:tr[1]+probe.shape[1]]

            sel11 = obj[tr[0]+1:tr[0]+1+probe.shape[0],
                        tr[1]+1:tr[1]+1+probe.shape[1]]

            selection = sel00 * (1-sp[0])*(1-sp[1]) + \
                sel01 * (1-sp[0])*sp[1] + \
                sel10 * sp[0]*(1-sp[1]) + \
                sel11 * sp[0]*sp[1]

            exit_waves.append(probe * selection)

    if single_translation:
        return exit_waves[0]
    else:
        return t.stack(exit_waves)


def ptycho_2D_sinc(probe, obj, translations, shift_probe=True, padding=10, multiple_modes=True, probe_support=None, shift_back_ew=False):
    """Returns a stack of exit waves accounting for subpixel shifts

    This function returns a collection of exit waves, with the first
    dimension as the translation index and the final dimensions
    corresponding to the detector. The exit waves are calculated by
    shifting the probe with each translation in turn, using sinc
    interpolation (done via multiplication with a complex exponential
    in Fourier space)

    If shift_probe is True, it applies the subpixel shift to the probe,
    otherwise the subpixel shift is applied to the object [not yet implemented]

    If multiple_modes is set to False, any additional dimensions in the
    ptycho_2D_round function will be assumed to correspond to the translation
    index. If multiple_modes is set to true, the (-4th) dimension of the probe
    will always be assumed to be defining a set of (P) incoherently mixing
    modes to be broadcast all translation indices. If any additional dimensions
    closer to the start exist, they will be assumed to be translation indices

    If shift_back_ew is set to False (the default), the output exit wave will
    be shifted within it's field of view. This is not a problem for far-field
    ptychography, but it is a problem for near-field ptychography. Therefore,
    when used for near-field ptychography, shift_back_ew=True should be used.

    Parameters
    ----------
    probe : torch.Tensor
        An (P)xMxL probe function for the exit waves
    object : torch.Tensor
        The object function to be probed
    translations : torch.Tensor
        The (N)x2 array of translations to simulate
    shift_probe : bool
        Default True, Whether to subpixel shift the probe or object
    multuple_modes : bool
        Default False, whether to assume the probe contains multiple modes
    shift_back_ew : bool
        Default False, whether to subpixel-shift the exit wave back after calculating

    Returns
    -------
    exit_waves : torch.Tensor
        An (N)x(P)xMxL tensor of the calculated exit waves
    """
    
    single_translation = False
    if translations.dim() == 1:
        translations = translations[None, :]
        single_translation = True

    # Separate the translations into a part that chooses the window
    # And a part that defines the windowing function
    integer_translations = t.floor(translations)
    subpixel_translations = translations - integer_translations
    integer_translations = integer_translations.to(dtype=t.int32)

    selections = t.stack([obj[..., tr[0]:tr[0]+probe.shape[-2],
                              tr[1]:tr[1]+probe.shape[-1]]
                          for tr in integer_translations])

    exit_waves = []
    if shift_probe:
        i = t.arange(probe.shape[-2],device=probe.device,dtype=t.float32) \
            - probe.shape[-2]//2
        j = t.arange(probe.shape[-1],device=probe.device,dtype=t.float32) \
            - probe.shape[-1]//2
        I,J = t.meshgrid(i,j, indexing='ij')
        I = 2 * np.pi * I / probe.shape[-2]
        J = 2 * np.pi * J / probe.shape[-1]
        phase_masks = t.exp(1j*(-subpixel_translations[:,0,None,None]*I
                                -subpixel_translations[:,1,None,None]*J))

        
        fft_probe = t.fft.fftshift(t.fft.fft2(probe),dim=(-1,-2))
        if multiple_modes: # Multi-mode probe
            shifted_fft_probe = fft_probe * phase_masks[...,None,:,:]
        else:
            shifted_fft_probe = fft_probe * phase_masks
        shifted_probe = t.fft.ifft2(t.fft.ifftshift(shifted_fft_probe,
                                                    dim=(-1,-2)))

        # Note: resist the temptation to remultiply by the probe support here,
        # it will fail if you have a probe which is restricted in Fourier space
        
        # TODO This is a kludge, I will fix this.
        if multiple_modes and len(selections.shape) == 3: # Multi-mode probe
            output = shifted_probe * selections[...,None,:,:]
        else:
            # This will only work if the 
            output = shifted_probe * selections

        if shift_back_ew:
            fft_output = t.fft.fftshift(t.fft.fft2(output),dim=(-1,-2))
            if multiple_modes: # Multi-mode probe
                shifted_fft_output = fft_output * \
                    t.conj(phase_masks[...,None,:,:])
            else:
                shifted_fft_output = fft_output * t.conj(phase_masks)

            output = t.fft.ifft2(t.fft.ifftshift(shifted_fft_output,
                                                 dim=(-1,-2)))


    else:
        raise NotImplementedError('Object shift not yet implemented')
    
    if single_translation:
        return output[0]
    else:
        return output


def ptycho_2D_sinc_s_matrix(probe, s_matrix, translations, shift_probe=True, padding=10):
    """Returns a stack of exit waves accounting for subpixel shifts

    This function returns a collection of exit waves, with the first
    dimension as the translation index and the final dimensions
    corresponding to the detector. The exit waves are calculated by
    shifting the probe with each translation in turn, using sinc
    interpolation (done via multiplication with a complex exponential
    in Fourier space)

    If shift_probe is True, it applies the subpixel shift to the probe,
    otherwise the subpixel shift is applied to the object

    This needs to be edited to use a slightly different meaning of the s-wave
    format. Currently, each pixel in the latter two dimensions index a location
    on the input wavefield, and the first two indexes index differences from
    that pixel. It is easier to interpret the resulting matrix though if the
    latter two indices index locations in the output plane. NOTE: I believe
    this change has now been made

    Parameters
    ----------
    probe : torch.Tensor
        An MxL probe function for the exit waves
    s_matrix : torch.Tensor
        The 4D S-Matrix tensor (2B+1x2B+1xObject Shape) to be probed
    translations : torch.Tensor
        The Nx2 array of translations to simulate
    shift_probe : bool
        Default True, Whether to subpixel shift the probe or object
    padding : int
         Default 10, if shifting the object, the padding to apply to the object to avoid circular shift effects

    Returns
    -------
    exit_waves : torch.Tensor
        An NxMxL tensor of the calculated exit waves
    """
    single_translation = False
    if translations.dim() == 1:
        translations = translations[None,:]
        single_translation = True

    # Separate the translations into a part that chooses the window
    # And a part that defines the windowing function
    integer_translations = t.floor(translations)
    subpixel_translations = translations - integer_translations
    integer_translations = integer_translations.to(dtype=t.int32)

    exit_waves = []

    B = s_matrix.shape[0]//2

    if shift_probe:
        i = t.arange(probe.shape[-2]) - probe.shape[-2]//2
        j = t.arange(probe.shape[-1]) - probe.shape[-1]//2
        I,J = t.meshgrid(i,j, indexing='ij')
        I = 2 * np.pi * I.to(t.float32) / probe.shape[-2]
        J = 2 * np.pi * J.to(t.float32) / probe.shape[-1]
        I = I.to(dtype=probe.dtype,device=probe.device)
        J = J.to(dtype=probe.dtype,device=probe.device)

        for tr, sp in zip(integer_translations,
                          subpixel_translations):
            fft_probe = t.fft.fftshift(t.fft.fft2(probe), dim=(-1,-2))
            shifted_fft_probe = fft_probe * t.exp(1j*(-sp[0]*I - sp[1]*J))
            shifted_probe = t.fft.ifft2(t.fft.ifftshift(shifted_fft_probe,
                                                         dim=(-1,-2)))

            s_matrix_slice = s_matrix[:,:,tr[0]:tr[0]+probe.shape[-2]+2*B,
                                      tr[1]:tr[1]+probe.shape[-1]+2*B]


            output = t.zeros([probe.shape[-2]+2*B,probe.shape[-1]+2*B,2]).to(
                                  device=s_matrix_slice.device,
                                  dtype=s_matrix_slice.dtype)


            for i in range(s_matrix.shape[0]):
                for j in range(s_matrix.shape[1]):
                    output [i:i+probe.shape[-2],j:j+probe.shape[-1]] += \
                        shifted_probe * s_matrix_slice[i,j,i:i+probe.shape[-2],j:j+probe.shape[-1]]

            exit_waves.append(output)

    else:
        raise NotImplementedError('Object shift not yet implemented')

    if single_translation:
        return exit_waves[0]
    else:
        return t.stack(exit_waves)


def RPI_interaction(probe, obj):
    """Returns an exit wave from a high-res probe and a low-res obj

    In this interaction, the probe and object arrays are assumed to cover
    the same physical region of space, but with the probe array sampling that
    region of space more finely. Thus, to do the interaction, the object
    is first upsampled by padding it in Fourier space (equivalent to a sinc
    interpolation) before being multiplied with the probe. This function is
    called RPI_interaction because this interaction is central to the RPI
    method and is not commonly used elsewhere.

    This also works with object functions that have an extra first dimension
    for an incoherently mixing model.


    Parameters
    ----------
    probe : torch.Tensor
        An MxL probe function for simulating the exit waves
    obj : torch.Tensor
        An M'xL' or NxM'xL' object function for simulating the exit waves

    Returns
    -------
    exit_wave : torch.Tensor
        An MxL tensor of the calculated exit waves
    """

    # TODO: The upsampling only works for arrays of even dimension!

    # The far-field propagator is just a 2D FFT but with an fftshift
    fftobj = propagators.far_field(obj)
    fftobj_npix = fftobj.shape[-2] * fftobj.shape[-1]
    
    # We calculate the padding that we need to do the upsampling
    # This is carefully set up to keep the zero-frequency pixel in the correct
    # location as the overall shape changes. Don't mess with this without
    # having thought about this carefully.
    pad2l = probe.shape[-2]//2 - obj.shape[-2]//2
    pad2r = probe.shape[-2] - obj.shape[-2] - pad2l
    pad1l = probe.shape[-1]//2 - obj.shape[-1]//2
    pad1r = probe.shape[-1] - obj.shape[-1] - pad1l
        
    fftobj = t.nn.functional.pad(fftobj, (pad1l, pad1r, pad2l, pad2r))

    # This keeps the mean intensity equal, instead of spreading out the
    # intensity over the upsampled region
    # TODO: Keeping this in probably isn't the most efficient
    fftobj_npix_new = fftobj.shape[-2] * fftobj.shape[-1]
    scale_factor = np.sqrt(fftobj_npix_new / fftobj_npix)
    
    # Again, just an inverse FFT but with an fftshift
    upsampled_obj = scale_factor * propagators.inverse_far_field(fftobj)
    
    if obj.dim() >= 3:
        return probe[None,...] *  upsampled_obj
    else:
        return probe * upsampled_obj
