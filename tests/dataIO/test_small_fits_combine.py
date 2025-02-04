#!/project/scratch01/compile/thomas.a/cosipy-env/bin/python3

from cosipy import UnBinnedData
from cosipy import test_data
import os

yaml = os.path.join(test_data.path,"inputs_crab.yaml")
tmp_path = "/tmp/"
small = UnBinnedData(yaml)
small_filename = small.data_file
small.data_file = os.path.join(test_data.path,small.data_file)
small.ori_file = os.path.join(test_data.path,small.ori_file)
small.unbinned_output = "fits"
small.small_fits_combine([tmp_path + "Crab_unbinned.fits", tmp_path+ "Crab_unbinned_copy.fits"]\
            ,output_name=tmp_path + "small_")
del small