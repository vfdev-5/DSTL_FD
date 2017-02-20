
# 
# Utils to preprocess images with OTB
#


# Configure the OTB path (folder with bin, lib, share)
OTB_PATH="C:\\Users\\victor.fomin\\Downloads\\OTB-5.10.0-win64"

import os
assert os.path.exists(os.path.join(OTB_PATH, 'lib', 'python', 'otbApplication.py')), "Orfeo-ToolBox is not found"
os.environ['PATH'] += ';' + os.path.join(OTB_PATH, 'bin')
os.environ['OTB_APPLICATION_PATH']=os.path.join(OTB_PATH, 'lib','otb','applications')

import sys
sys.path.append(os.path.join(OTB_PATH, 'lib', 'python'))

from image_utils import get_filename

def generate_rm_indices(image_id):
    """
    Method to generate radiometric indices (ndvi, gemi, ndwi2, ndti)
    See https://www.orfeo-toolbox.org/CookBook/Applications/app_RadiometricIndices.html
    """
    in_fname = get_filename(image_id)

    RadiometricIndices = otbApplication.Registry.CreateApplication("RadiometricIndices")
    assert RadiometricIndices is not None, "OTB application is not found"

    RadiometricIndices.SetParameterString("in", "qb_RoadExtract.tif")
