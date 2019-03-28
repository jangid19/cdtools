from __future__ import division, print_function, absolute_import

from CDTools.datasets import *
from CDTools.tools import data as cdtdata
import numpy as np
import torch as t
import h5py
import pytest
import datetime


#
# We start by testing the CDataset base class
#

def test_CDataset_init():
    entry_info = {'start_time': datetime.datetime.now(),
                  'title' : 'A simple test'}
    sample_info = {'name': 'A test sample',
                   'mass' : 3.4,
                   'unit_cell' : np.array([1,1,1,87,84.5,90])}
    wavelength = 1e-9
    detector_geometry = {'distance': 0.7,
                         'basis': np.array([[0,-30e-6,0],
                                            [-20e-6,0,0]]).transpose(),
                         'corner': np.array((2550e-6,3825e-6,0.3))}
    mask = np.ones((256,256))
    dataset = CDataset(entry_info, sample_info,
                       wavelength, detector_geometry, mask)
    assert t.all(t.eq(dataset.mask,t.tensor(mask)))
    assert dataset.entry_info == entry_info
    assert dataset.sample_info == sample_info
    assert dataset.wavelength == wavelength
    assert dataset.detector_geometry == detector_geometry


def test_CDataset_from_cxi(test_ptycho_cxis):
    for cxi, expected in test_ptycho_cxis:
        dataset = CDataset.from_cxi(cxi)

        # The entry metadata loaded
        for key in expected['entry metadata']:
            assert dataset.entry_info[key] == expected['entry metadata'][key]

        # Don't test for fidelity since this is tested in the data, just test
        # that it is loaded
        if expected['sample info'] is None:
            assert dataset.sample_info is None
        else:
            assert dataset.sample_info is not None

        assert np.isclose(dataset.wavelength,expected['wavelength'])

        # Just check one of the loaded attributes
        assert np.isclose(dataset.detector_geometry['distance'],
                          expected['detector']['distance'])
        # Check that the other ones are loaded but not for fidelity
        assert 'basis' in dataset.detector_geometry
        if expected['detector']['corner'] is not None:
            assert 'corner' in dataset.detector_geometry

        if expected['mask'] is not None:
            assert t.all(t.eq(t.tensor(expected['mask']),dataset.mask))

            
def test_CDataset_to_cxi(test_ptycho_cxis, tmp_path):
    for cxi, expected in test_ptycho_cxis:
        dataset = CDataset.from_cxi(cxi)
        with cdtdata.create_cxi(tmp_path / 'test_CDataset_to_cxi.cxi') as f:
            dataset.to_cxi(f)

        # Now we have to check that all the stuff was written
        with h5py.File(tmp_path / 'test_CDataset_to_cxi.cxi', 'r') as f:
            read_dataset = CDataset.from_cxi(f)

        assert dataset.entry_info == read_dataset.entry_info

        if dataset.sample_info is None:
            assert read_dataset.sample_info is None
        else:
            assert read_dataset.sample_info is not None

        assert np.isclose(dataset.wavelength, read_dataset.wavelength)

        
         # Just check one of the loaded attributes
        assert np.isclose(dataset.detector_geometry['distance'],
                          read_dataset.detector_geometry['distance'])
        # Check that the other ones are loaded but not for fidelity
        assert 'basis' in read_dataset.detector_geometry
        if dataset.detector_geometry['corner'] is not None:
            assert 'corner' in read_dataset.detector_geometry
            
        
        if dataset.mask is not None:
            assert t.all(t.eq(dataset.mask,read_dataset.mask))
        


def test_CDataset_to(ptycho_cxi_1):
    dataset = CDataset.from_cxi(ptycho_cxi_1[0])

    dataset.to(dtype=t.float32)
    assert dataset.mask.dtype == t.uint8
    # If cuda is available, check that moving the mask to CUDA works.
    if t.cuda.is_available():
        dataset.to(device='cuda:0')
        assert dataset.mask.device == t.device('cuda:0')



#
# And we then test the derived Ptychography class
#
#


def test_Ptycho_2D_Dataset_init():
    entry_info = {'start_time': datetime.datetime.now(),
                  'title' : 'A simple test'}
    sample_info = {'name': 'A test sample',
                   'mass' : 3.4,
                   'unit_cell' : np.array([1,1,1,87,84.5,90])}
    wavelength = 1e-9
    detector_geometry = {'distance': 0.7,
                         'basis': np.array([[0,-30e-6,0],
                                            [-20e-6,0,0]]).transpose(),
                         'corner': np.array((2550e-6,3825e-6,0.3))}
    mask = np.ones((256,256))
    patterns = np.random.rand(20,256,256)
    translations = np.random.rand(20,3)
    
    dataset = Ptycho_2D_Dataset(translations, patterns,
                                entry_info=entry_info,
                                sample_info=sample_info,
                                wavelength=wavelength,
                                detector_geometry=detector_geometry,
                                mask=mask)

    assert t.all(t.eq(dataset.mask,t.tensor(mask)))
    assert dataset.entry_info == entry_info
    assert dataset.sample_info == sample_info
    assert dataset.wavelength == wavelength
    assert dataset.detector_geometry == detector_geometry
    assert t.allclose(dataset.patterns, t.tensor(patterns))
    assert t.allclose(dataset.translations, t.tensor(translations))



def test_Ptycho_2D_Dataset_from_cxi(test_ptycho_cxis):
    for cxi, expected in test_ptycho_cxis:
        dataset = Ptycho_2D_Dataset.from_cxi(cxi)

        # The entry metadata loaded
        for key in expected['entry metadata']:
            assert dataset.entry_info[key] == expected['entry metadata'][key]

        # Don't test for fidelity since this is tested in the data, just test
        # that it is loaded
        if expected['sample info'] is None:
            assert dataset.sample_info is None
        else:
            assert dataset.sample_info is not None

        assert np.isclose(dataset.wavelength,expected['wavelength'])

        # Just check one of the loaded attributes
        assert np.isclose(dataset.detector_geometry['distance'],
                          expected['detector']['distance'])
        # Check that the other ones are loaded but not for fidelity
        assert 'basis' in dataset.detector_geometry
        if expected['detector']['corner'] is not None:
            assert 'corner' in dataset.detector_geometry

        if expected['mask'] is not None:
            assert t.all(t.eq(t.tensor(expected['mask']),dataset.mask))

        assert t.allclose(t.tensor(expected['data']),dataset.patterns)
        assert t.allclose(t.tensor(expected['translations']),dataset.translations)
        


            
def test_Ptycho_2D_Dataset_to_cxi(test_ptycho_cxis, tmp_path):
    for cxi, expected in test_ptycho_cxis:
        dataset = Ptycho_2D_Dataset.from_cxi(cxi)
        with cdtdata.create_cxi(tmp_path / 'test_Ptycho_2D_Dataset_to_cxi.cxi') as f:
            dataset.to_cxi(f)

        # Now we have to check that all the stuff was written
        with h5py.File(tmp_path / 'test_Ptycho_2D_Dataset_to_cxi.cxi', 'r') as f:
            read_dataset = Ptycho_2D_Dataset.from_cxi(f)

        assert dataset.entry_info == read_dataset.entry_info

        if dataset.sample_info is None:
            assert read_dataset.sample_info is None
        else:
            assert read_dataset.sample_info is not None

        assert np.isclose(dataset.wavelength, read_dataset.wavelength)

        
         # Just check one of the loaded attributes
        assert np.isclose(dataset.detector_geometry['distance'],
                          read_dataset.detector_geometry['distance'])
        # Check that the other ones are loaded but not for fidelity
        assert 'basis' in read_dataset.detector_geometry
        if dataset.detector_geometry['corner'] is not None:
            assert 'corner' in read_dataset.detector_geometry
            
        
        if dataset.mask is not None:
            assert t.all(t.eq(dataset.mask,read_dataset.mask))

        assert t.allclose(dataset.patterns, read_dataset.patterns)
        assert t.allclose(dataset.translations, read_dataset.translations)

    
def test_Ptycho_2D_Dataset_to(ptycho_cxi_1):
    dataset = Ptycho_2D_Dataset.from_cxi(ptycho_cxi_1[0])
    
    dataset.to(dtype=t.float64)
    assert dataset.mask.dtype == t.uint8
    assert dataset.patterns.dtype == t.float64
    assert dataset.translations.dtype == t.float64
    # If cuda is available, check that moving the mask to CUDA works.
    if t.cuda.is_available():
        dataset.to(device='cuda:0')
        assert dataset.mask.device == t.device('cuda:0')
        assert dataset.patterns.device == t.device('cuda:0')
        assert dataset.translations.device == t.device('cuda:0')

    
def test_Ptycho_2D_Dataset_ops(ptycho_cxi_1):
    cxi, expected = ptycho_cxi_1
    dataset = Ptycho_2D_Dataset.from_cxi(cxi)

    assert len(dataset) == expected['data'].shape[0]
    idx, translation, pattern = dataset[3]
    assert idx == 3
    assert t.allclose(translation, t.tensor(expected['translations'][3,:]))
    assert t.allclose(pattern, t.tensor(expected['data'][3,:,:]))