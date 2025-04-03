import datetime
import configparser
import catalogue_sim
import pcl_measurement
import likelihood


def run_catalogue_sim(pipeline_variables_path, clean=True):

    config = configparser.ConfigParser()
    config.read(pipeline_variables_path)

    # Generate random sample
    catalogue_sim.init_rand_sample.execute(pipeline_variables_path)

    # Create n(z)
    catalogue_sim.create_nz.execute(pipeline_variables_path)

    # Generate fiducial Cls
    catalogue_sim.generate_cls.execute(pipeline_variables_path)

    # Convert fiducial Cls to Flask format
    catalogue_sim.conv_fields_cosmosis_flask.execute(pipeline_variables_path)

    no_realisations = int(config['simulation_setup']['REALISATIONS'])

    # Run Flask to generate correlated fields on the sky
    for i in range(no_realisations):
        catalogue_sim.run_flask.execute(pipeline_variables_path=pipeline_variables_path, iter_no=i+1)

        if clean:
            # Measure power spectra and delete maps immediately to save space
            pcl_measurement.measure_spectra.execute(pipeline_variables_path=pipeline_variables_path, realisation=i+1)
            pcl_measurement.clean_products.execute(pipeline_variables_path=pipeline_variables_path, iter_no=i+1)


def measure_pcls(pipeline_variables_path, clean=True, cov_iter=False):

    config = configparser.ConfigParser()
    config.read(pipeline_variables_path)

    no_realisations = int(config['simulation_setup']['REALISATIONS'])

    if not clean:
        # # Measure power spectra from mock catalogues
        for j in range(no_realisations):
            pcl_measurement.measure_spectra.execute(pipeline_variables_path=pipeline_variables_path, realisation=j+1)

        #     # Delete data (for saving storage space)
            pcl_measurement.clean_products.execute(pipeline_variables_path=pipeline_variables_path, iter_no=j+1)

    # Average Cls over all realisations
    pcl_measurement.av_cls.execute(pipeline_variables_path)

    # Convert Cls to bandpowers
    pcl_measurement.measure_cat_bps.execute(pipeline_variables_path)

    # Convert bandpower data vector to 1D
    pcl_measurement.conv_bps.execute(pipeline_variables_path)

    if not cov_iter:
        # Calculate numerical covariance matrix
        pcl_measurement.cov_fromsim.execute(pipeline_variables_path)

    else:
        # If we want to calculate covariance matrix for different numbers of realisations
        cov_iter_nos = [1, 2, 3]
        for cov_iter_no in cov_iter_nos:
            pcl_measurement.cov_fromsim_iters.execute(pipeline_variables_path, cov_iter_no)


def run_likelihood(pipeline_variables_path):

    # Run nautilus sampler to perform likelihood analysis
    likelihood.sampler.execute(pipeline_variables_path)


def main():

    pipeline_variables_path = \
        '/raid/scratch/wongj/mywork/3x2pt/6x2pt_sim/set_config/set_variables.ini'

    # # Create catalogues
    run_catalogue_sim(pipeline_variables_path, clean=True)

    # # Measure Pseudo-Cls
    measure_pcls(pipeline_variables_path, clean=True, cov_iter=False)

    now = datetime.datetime.now()
    print(now)
    # # Perform likelihood analysis
    run_likelihood(pipeline_variables_path)


if __name__ == '__main__':
    main()
