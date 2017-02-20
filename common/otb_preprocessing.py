
# 
# Utils to preprocess images with OTB
#

import os
import sys
from image_utils import get_filename

# Configure the OTB path (folder with bin, lib, share)
import yaml
assert os.path.exists('../common/otb_conf.yaml'), \
    "OTB configuration file is not found. Modify and rename otb_conf.yaml.example to otb_conf.yaml"
with open('../common/otb_conf.yaml', 'r') as f:
    cfg = yaml.load(f)
    assert "OTB_PATH" in cfg, "otb_conf.yaml does not contain OTB_PATH"
    OTB_PATH = cfg['OTB_PATH']


assert os.path.exists(os.path.join(OTB_PATH, 'lib', 'python', 'otbApplication.py')), "Orfeo-ToolBox is not found"
os.environ['PATH'] += os.pathsep + os.path.join(OTB_PATH, 'bin')
os.environ['OTB_APPLICATION_PATH'] = os.path.join(OTB_PATH, 'lib', 'otb', 'applications')
sys.path.append(os.path.join(OTB_PATH, 'lib', 'python'))


def generate_rm_indices(image_id):
    """
    Method to generate radiometric indices (ndvi, gemi, ndwi2, ndti)
    See https://www.orfeo-toolbox.org/CookBook/Applications/app_RadiometricIndices.html
    """
    in_fname = get_filename(image_id)

    RadiometricIndices = otbApplication.Registry.CreateApplication("RadiometricIndices")
    assert RadiometricIndices is not None, "OTB application is not found"

    RadiometricIndices.SetParameterString("in", "qb_RoadExtract.tif")
