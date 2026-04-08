import pytest
import numpy as np
from astropy.io import fits
import os
import matplotlib
matplotlib.use('Agg') # Fixes the headless server crash
import matplotlib.pyplot as plt

from cosipy.phase_resolved_analysis import PhaseAssigner, PhaseSelector, PlotPulseProfile

def test_phase_assigner_math():
    """Verify the core folding math: Phase = (T * F0) % 1.0"""
    par_path = "test_pulsar.par"
    with open(par_path, "w") as f:
        f.write("F0 10.0\n") 
    
    try:
        assigner = PhaseAssigner(par_path)
        assert assigner.f0 == 10.0
        
        test_times = np.array([0.0, 0.05, 0.11])
        expected_phases = np.array([0.0, 0.5, 0.1])
        
        calculated_phases = (test_times * assigner.f0) % 1.0
        np.testing.assert_allclose(calculated_phases, expected_phases, atol=1e-7)
    finally:
        if os.path.exists(par_path):
            os.remove(par_path)

def test_phase_selector_wrap_around():
    selector = PhaseSelector(intervals=[(0.0, 0.2), (0.8, 1.0)])
    test_phases = np.array([0.1, 0.5, 0.9])
    mask = selector._get_vectorized_mask(test_phases)
    assert np.array_equal(mask, [True, False, True])

def test_plotter_data_handling():
    """Verify that the plotter correctly handles NumPy structured arrays."""
    # Create mock data as a structured array (this triggered the previous error)
    data = np.zeros(10, dtype=[('PULSE_PHASE', 'f8'), ('TimeTags', 'f8')])
    plotter = PlotPulseProfile(data)
    
    assert len(plotter.phases) == 10
    assert len(plotter.times) == 10
    
    # Run plot to ensure no crash
    plotter.plot()
    plt.close('all')
