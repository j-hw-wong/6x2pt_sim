"""
Script to delete catalogue simulation by-products that have been saved on disk. Important for considerations of disk
space, especially for large numbers of galaxies and high redshfit ranges. Deletes the interpolated field maps and
the catalogue indices by-product following Poisson sampling. Repeated over a given number of realisations/iterations.
"""

import shutil
import configparser


def execute(pipeline_variables_path, iter_no):

    # pipeline_variables_path = os.environ['PIPELINE_VARIABLES_PATH']

    config = configparser.ConfigParser()
    config.read(pipeline_variables_path)

    # iter_no = int(float(os.environ['ITER_NO']))

    save_dir = str(config['simulation_setup']['SIMULATION_SAVE_DIR'])
    maps_dir = save_dir + 'flask/output/'

    try:
        shutil.rmtree(maps_dir+'iter_{}/'.format(iter_no))
    except OSError as e:
        print("Error: %s - %s." % (e.filename, e.strerror))

