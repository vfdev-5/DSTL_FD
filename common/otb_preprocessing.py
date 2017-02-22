
# 
# Utils to preprocess images with OTB
#

import os
import sys
import subprocess
import logging
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
    Method to generate radiometric indices (ndvi, gemi, ndwi2, ndti, bi, bi2)
    See https://www.orfeo-toolbox.org/CookBook/Applications/app_RadiometricIndices.html
    """
    app_name = 'otbcli_RadiometricIndices'
    if sys.platform == 'win32':
        app_name += '.bat'
    app_path = os.path.join(OTB_PATH, 'bin', app_name)
    assert os.path.exists(app_path), "OTB application 'RadiometricIndices' is not found"

    in_fname = get_filename(image_id, '17b')
    out_fname = get_filename(image_id, 'multi')
    if os.path.exists(out_fname):
        logging.warn("File '%s' is already existing" % out_fname)
        return 

    list_ch = [
        'Vegetation:NDVI', 
        'Vegetation:GEMI', 
        'Water:NDWI2', 
        'Water:NDTI', 
        'Soil:BI',
        'Soil:BI2'
    ]

    ram = ['-ram', '1024']
    channels = [
        '-channels.red', '6',
        '-channels.green', '3',
        '-channels.blue', '2',
        '-channels.nir', '7',
        '-channels.mir', '17'
    ]

    program = [app_path, '-in', in_fname, '-out', out_fname, '-list']
    program.extend(list_ch)
    program.extend(ram)
    program.extend(channels)

    p = subprocess.Popen(program, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out = p.stdout.readlines()
    err = p.stderr.readlines()
    if len(err) > 0:
        logging.error("RadiometricIndices failed with error : %s" % err)
        print err
    p.wait()

    
try:
    import otbApplication
    
    # The following line creates an instance of the RadiometricIndices application


# The following lines set all the application parameters:
RadiometricIndices.SetParameterString("in", "qb_RoadExtract.tif")

# The following line execute the application
RadiometricIndices.ExecuteAndWriteOutput()
    
    def compute_rm_indices_matrix(image_id, list_ch=(), channels_dict={}):
        """
        Method to generate radiometric indices (ndvi, gemi, ndwi2, ndti, bi, bi2)
        See https://www.orfeo-toolbox.org/CookBook/Applications/app_RadiometricIndices.html
        """
        
        RadiometricIndices = otbApplication.Registry.CreateApplication("RadiometricIndices")
        assert RadiometricIndices is not None, "OTB application 'RadiometricIndices' is not found"

        in_fname = get_filename(image_id, '17b')
        RadiometricIndices.SetParameterString("in", in_fname)
        
        if len(list_ch) == 0:
            list_ch = [
                'Vegetation:NDVI', 
            ]
        RadiometricIndices.SetParameterString("list", " ".join(list_ch))
        RadiometricIndices.SetParameterInt("ram", 1024)
        
        channels = ['red', 'green', ]
        
        channels = [
            '-channels.red', '6',
            '-channels.green', '3',
            '-channels.blue', '2',
            '-channels.nir', '7',
            '-channels.mir', '17'
        ]

        program = [app_path, '-in', in_fname, '-out', out_fname, '-list']
        program.extend(list_ch)
        program.extend(ram)
        program.extend(channels)

        p = subprocess.Popen(program, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out = p.stdout.readlines()
        err = p.stderr.readlines()
        if len(err) > 0:
            logging.error("RadiometricIndices failed with error : %s" % err)
            print err
        p.wait()

except:
    print "OTB python wrapper is not available"
    
    
    
        
