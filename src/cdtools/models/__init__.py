"""This module contains all the models for different CDI Reconstructions

All the reconstructions are coordinated through the ptychography models
defined here. The models are, at their core, just subclasses of the 
:code:`torch.nn.model` class, so they contain the same structure of
parameters, etc. Their central functionality is as a simulation that maps
some input (usually, the index number of a scan point) to an output that
corresponds to the measured data (usually, a diffraction pattern). This
model can then be used as the heart of an automatic differentiation
reconstruction which retrieves the parameters that were used in the model.

A main CDIModel class is defined in the base.py file, and models for
various CDI geometries can be defined as subclasses of this base model.
The subclasses of the main CDIModel class are required to implement a set of
functions defined in the base.py file. Example implementations of
these functions can be found in the code for the SimplePtycho class.

Finally, it is recommended to read through the tutorial section on
defining a new ptychography model before attempting to do so.

"""

# I don't believe that __all__ really needed, but it's nice to define it
# to be explicit that import * is safe
__all__ = ['CDIModel', 'SimplePtycho', 'FancyPtycho', 'PolarizedFancyPtycho', 'Bragg2DPtycho', 'Multislice2DPtycho', 'RPI', 'TimeResolvedPtychoCalibration', 'TimeResolvedRPI']

from cdtools.models.base import CDIModel
from cdtools.models.simple_ptycho import SimplePtycho
from cdtools.models.fancy_ptycho import FancyPtycho
from cdtools.models.polarized_fancy_ptycho import PolarizedFancyPtycho
from cdtools.models.bragg_2d_ptycho import Bragg2DPtycho
from cdtools.models.multislice_2d_ptycho import Multislice2DPtycho
from cdtools.models.rpi import RPI
from cdtools.models.multimode_rpi import MultimodeRPI
from cdtools.models.time_resolved_ptycho_calibration import TimeResolvedPtychoCalibration
from cdtools.models.time_resolved_rpi import TimeResolvedRPI
from cdtools.models.fastccd_ptycho import FastCCDPtycho

# Still needs to be updated for the new complex numbers
#from cdtools.models.s_matrix_ptycho import SMatrixPtycho
