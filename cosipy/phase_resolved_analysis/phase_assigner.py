import logging
import numpy as np
from astropy.io import fits

logger = logging.getLogger(__name__)

class PhaseAssigner:
    """Calculates and assigns pulsar phases to FITS event data.

    This class parses a pulsar timing model (.par) file to retrieve the spin 
    frequency and applies a simple folding algorithm to Mission Elapsed Time (MET) 
    columns within FITS files. It is optimized for frequency-only ephemerides.

    Attributes:
        f0 (float): The pulsar spin frequency (F0) in Hertz.
    """

    def __init__(self, par_file):
        """Initializes the assigner with a spin frequency from a parameter file.

        Args:
            par_file (str): Path to the pulsar ephemeris (.par) file.

        Raises:
            ValueError: If the 'F0' parameter is not found within the provided file.
        """
        self.f0 = self._extract_f0(par_file)
        logger.info(f"Loaded Frequency: {self.f0} Hz")

    def _extract_f0(self, path):
        """Scans a .par file to locate and extract the spin frequency (F0).

        This helper handles standard pulsar timing file formats, including 
        Fortran-style double precision exponents (replacing 'D' with 'E').

        Args:
            path (str): The file system path to the .par file.

        Returns:
            float: The extracted spin frequency in Hz.

        Raises:
            ValueError: If the file does not contain a line starting with 'F0'.
        """
        with open(path, 'r') as f:
            for line in f:
                parts = line.split()
                if parts and parts[0].upper() == 'F0':
                    # Convert Fortran 'D' notation to standard scientific 'E' notation
                    return float(parts[1].replace('D', 'E'))
        raise ValueError(f"F0 (spin frequency) missing in {path}")

    def add_phase_column(self, input_fits, output_fits=None):
        """Calculates pulsar phases and injects them as a new column in a FITS file.

        The phase calculation follows a standard simple folding algorithm:
        $$\text{Phase} = (T \times F_0) \pmod{1.0}$$
        where $T$ is the time tag and $F_0$ is the spin frequency.

        Args:
            input_fits (str): Path to the source FITS file containing event data.
            output_fits (str, optional): Path where the modified FITS will be saved.
                If None, the input file will be overwritten in-place.

        Returns:
            str: The path to the saved FITS file.

        Note:
            This method specifically looks for 'TIME' or 'TimeTags' in the 
            Binary Table extension (index 1). If 'PULSE_PHASE' already exists, 
            it will be updated with the new calculations.
        """
        with fits.open(input_fits) as hdul:
            data = hdul[1].data
            header = hdul[1].header
            
            # Extract time tags—check for standard column naming conventions
            if 'TIME' in data.dtype.names:
                times = data['TIME']
            elif 'TimeTags' in data.dtype.names:
                times = data['TimeTags']
            else:
                raise KeyError("Could not find a valid time column ('TIME' or 'TimeTags').")

            # Simple folding: (T * F) mod 1
            # Using vectorized NumPy operations for speed on large datasets
            phase = (times * self.f0) % 1.0
            
            # Column Management: Update existing or append new
            if 'PULSE_PHASE' in data.dtype.names:
                logger.info("Overwriting existing PULSE_PHASE column.")
                data['PULSE_PHASE'] = phase
                new_hdu = fits.BinTableHDU(data=data, header=header)
            else:
                logger.info("Creating new PULSE_PHASE column.")
                new_col = fits.Column(name='PULSE_PHASE', format='D', array=phase)
                new_hdu = fits.BinTableHDU.from_columns(data.columns + new_col, header=header)

            # File I/O
            out = output_fits or input_fits
            # Preserve the PrimaryHDU (hdul[0]) while updating the table
            fits.HDUList([hdul[0], new_hdu]).writeto(out, overwrite=True)
            logger.info(f"PULSE_PHASE assigned to: {out}")
            return out
