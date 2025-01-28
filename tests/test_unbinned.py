#!/project/scratch01/compile/thomas.a/cosipy-env/bin/python
# coding: utf-8

# In[5]:


from cosipy import UnBinnedData
from cosipy import test_data
# In[7]:
yaml = os.path.join(test_data.path,"inputs_crab.yaml")
analysis = UnBinnedData(yaml)
test_filename = analysis.data_file
# In[9]:
analysis.data_file = os.path.join(test_data.path,analysis.data_file)
analysis.ori_file = os.path.join(test_data.path,analysis.ori_file)
# In[17]:
tmp_path = "/project/scratch01/compile/thomas.a/"
# Testing
# In[19]:
analysis.unbinned_output = "fits"
analysis.combine_unbinned_data([tmp_path + "Crab_unbinned.fits.gz", tmp_path+ "Crab_unbinned.fits.gz"],output_name=tmp_path + "temp_test_file_original")
# In[ ]: