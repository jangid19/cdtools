import torch as t
from cdtools.models import CDIModel
from cdtools.datasets import Ptycho2DDataset
from cdtools import tools
from cdtools.tools import plotting as p
from copy import copy
from torch.utils import data as torchdata
from datetime import datetime
import numpy as np

__all__ = ['SimplePtycho']        


class SimplePtycho(CDIModel):
    """A simple ptychography model for exploring ideas and extensions



    """
    def __init__(self, wavelength, detector_geometry,
                 probe_basis, detector_slice,
                 probe_guess, obj_guess, min_translation = [0,0],
                 surface_normal=np.array([0.,0.,1.]), mask=None):

        super(SimplePtycho,self).__init__()
        self.register_buffer('wavelength', t.as_tensor(wavelength))
        self.store_detector_geometry(detector_geometry)

        self.register_buffer('min_translation', t.as_tensor(min_translation))
        self.register_buffer('probe_basis', t.as_tensor(probe_basis))
        self.detector_slice = copy(detector_slice)

        self.register_buffer('surface_normal', t.as_tensor(surface_normal))

        
        if mask is None:
            self.register_buffer('mask', None)
        else:
            self.register_buffer('mask', t.as_tensor(mask, dtype=t.bool))

        probe_guess = t.tensor(probe_guess, dtype=t.complex64)
        obj_guess = t.tensor(obj_guess, dtype=t.complex64)

        # We rescale the probe here so it learns at the same rate as the
        # object
        self.register_buffer('probe_norm', t.max(t.abs(probe_guess)))

        #self.probe_data = complexWrapper(probe_guess/self.probe_norm)
        self.probe_data = t.nn.Parameter(t.view_as_real(probe_guess / self.probe_norm))
        self.obj_data = t.nn.Parameter(t.view_as_real(obj_guess))

    # Do this for all the complex-valued parameters which are stored as real
    # valued for compatibility reasons. Note that updating model.probe.data
    # won't update the underlying storage, but something like:
    # model.probe.data[:] = ...
    # will update the underlying storage
    @property
    def probe(self):
        return t.view_as_complex(self.probe_data)

    @property
    def obj(self):
        return t.view_as_complex(self.obj_data)

    @classmethod
    def from_dataset(cls, dataset):
        wavelength = dataset.wavelength
        det_basis = dataset.detector_geometry['basis']
        det_shape = dataset[0][1].shape
        distance = dataset.detector_geometry['distance']

        # always do this on the cpu
        get_as_args = dataset.get_as_args
        dataset.get_as(device='cpu')
        (indices, translations), patterns = dataset[:]
        dataset.get_as(*get_as_args[0],**get_as_args[1])

        center = tools.image_processing.centroid(t.sum(patterns,dim=0))

        # Then, generate the probe geometry from the dataset
        ewg = tools.initializers.exit_wave_geometry
        probe_basis, probe_shape, det_slice =  ewg(det_basis,
                                                   det_shape,
                                                   wavelength,
                                                   distance,
                                                   center=center)

        if hasattr(dataset, 'sample_info') and \
           dataset.sample_info is not None and \
           'orientation' in dataset.sample_info:
            surface_normal = dataset.sample_info['orientation'][2]
        else:
            surface_normal = np.array([0.,0.,1.])

        # Next generate the object geometry from the probe geometry and
        # the translations
        pix_translations = tools.interactions.translations_to_pixel(probe_basis, translations, surface_normal=surface_normal)
        obj_size, min_translation = tools.initializers.calc_object_setup(probe_shape, pix_translations)

        # Finally, initialize the probe and  object using this information
        probe = tools.initializers.SHARP_style_probe(dataset, probe_shape, det_slice)
        
        obj = t.ones(obj_size).to(dtype=t.complex64)
        det_geo = dataset.detector_geometry


        if hasattr(dataset, 'mask') and dataset.mask is not None:
            mask = dataset.mask.to(t.bool)
        else:
            mask = None

        return cls(wavelength, det_geo, probe_basis, det_slice, probe, obj, min_translation=min_translation, mask=mask, surface_normal=surface_normal)


    def interaction(self, index, translations):
        pix_trans = tools.interactions.translations_to_pixel(self.probe_basis,
                                                             translations,
                                            surface_normal=self.surface_normal)
        pix_trans -= self.min_translation
        return tools.interactions.ptycho_2D_round(self.probe_norm * self.probe,
                                                  self.obj,
                                                  pix_trans)


    def forward_propagator(self, wavefields):
        return tools.propagators.far_field(wavefields)


    def backward_propagator(self, wavefields):
        return tools.propagators.inverse_far_field(wavefields)


    def measurement(self, wavefields):
        return tools.measurements.intensity(wavefields,
                                            detector_slice=self.detector_slice)


    def loss(self, real_data, sim_data, mask=None):
        return tools.losses.amplitude_mse(real_data, sim_data, mask=mask)


    def sim_to_dataset(self, args_list):
        # In the future, potentially add more control
        # over what metadata is saved (names, etc.)

        # First, I need to gather all the relevant data
        # that needs to be added to the dataset
        entry_info = {'program_name': 'cdtools',
                      'instrument_n': 'Simulated Data',
                      'start_time': datetime.now()}

        surface_normal = self.surface_normal.detach().cpu().numpy()
        xsurfacevec = np.cross(np.array([0.,1.,0.]), surface_normal)
        xsurfacevec /= np.linalg.norm(xsurfacevec)
        ysurfacevec = np.cross(surface_normal, xsurfacevec)
        ysurfacevec /= np.linalg.norm(ysurfacevec)
        orientation = np.array([xsurfacevec, ysurfacevec, surface_normal])

        sample_info = {'description': 'A simulated sample',
                       'orientation': orientation}

        detector_geometry = self.detector_geometry
        mask = self.mask
        wavelength = self.wavelength
        indices, translations = args_list

        # Then we simulate the results
        data = self.forward(indices, translations)

        # And finally, we make the dataset
        return Ptycho2DDataset(translations, data,
                                 entry_info = entry_info,
                                 sample_info = sample_info,
                                 wavelength=wavelength,
                                 detector_geometry=detector_geometry,
                                 mask=mask)



    plot_list = [
        ('Probe Amplitude',
         lambda self, fig: p.plot_amplitude(self.probe, fig=fig, basis=self.probe_basis)),
        ('Probe Phase',
         lambda self, fig: p.plot_phase(self.probe, fig=fig, basis=self.probe_basis)),
        ('Object Amplitude',
         lambda self, fig: p.plot_amplitude(self.obj, fig=fig, basis=self.probe_basis)),
        ('Object Phase',
         lambda self, fig: p.plot_phase(self.obj, fig=fig, basis=self.probe_basis))
    ]
    

    def ePIE(self, iterations, dataset, beta = 1.0):
        """Runs an ePIE reconstruction as described in `Maiden et al. (2017) <https://www.osapublishing.org/optica/abstract.cfm?uri=optica-4-7-736>`_.
        Optional parameters are:

        :arg ``iterations``: Controls the number of iterations run, defaults to 1.
        :arg ``beta``: Algorithmic parameter described in Maiden's implementation of rPIE. Defaults to 0.15.
        :arg ``probe``: Initial probe wavefunction.
        :arg ``object``: Initial object wavefunction.
        """
        probe_shape = self.probe.shape

        if self.mask is not None:
            mask = self.mask[...,None]
        else:
            mask=None

        def probe_update(exit_wave, exit_wave_corrected, probe, object, translation):
            new_probe = probe + tools.cmath.cmult(beta * tools.cmath.cconj(object[translation])/(self.probe_norm*t.max(tools.cmath.cabssq(object))), exit_wave_corrected-exit_wave)
            return new_probe

        def object_update(exit_wave, exit_wave_corrected, probe, object, translation):
            new_object = object.clone()
            new_object[translation] = object[translation] + tools.cmath.cmult(beta * tools.cmath.cconj(probe)/(self.probe_norm*t.max(tools.cmath.cabssq(probe))), exit_wave_corrected-exit_wave)
            return new_object

        with t.no_grad():
            data_loader = torchdata.DataLoader(dataset, shuffle=True)

            for it in range(iterations):
                loss = []
                for (i, [translations]), [patterns] in data_loader:
                    probe = self.probe.data.clone()
                    object = self.obj.data.clone()

                    exit_wave = self.interaction(i, translations).clone()
                    # Apply modulus constraint
                    exit_wave_corrected = exit_wave.clone()
                    exit_wave_corrected = self.forward_propagator(exit_wave_corrected.clone())
                    exit_wave_corrected[self.detector_slice] = tools.projectors.modulus(exit_wave_corrected.clone()[self.detector_slice], patterns, mask = mask)
                    exit_wave_corrected = self.backward_propagator(exit_wave_corrected.clone())

                    # Calculate the section of the object wavefunction to be modified
                    pix_trans = tools.interactions.translations_to_pixel(self.probe_basis,
                                                                         translations)
                    pix_trans -= self.min_translation

                    pix_trans = t.round(pix_trans).to(dtype=t.int32).detach().cpu().numpy()

                    object_slice = np.s_[pix_trans[0]:
                                      pix_trans[0]+probe_shape[0],
                                      pix_trans[1]:
                                      pix_trans[1]+probe_shape[1]]

                    # Apply probe and object updates
                    self.probe.data = probe_update(exit_wave, exit_wave_corrected, probe, object, object_slice)
                    self.obj.data = object_update(exit_wave, exit_wave_corrected, probe, object, object_slice)

                    # Calculate loss
                    loss.append(self.loss(self.measurement(self.interaction(i, translations)), patterns))

                yield t.mean(t.tensor(loss)).cpu().numpy()
