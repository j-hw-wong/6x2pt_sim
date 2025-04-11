import datetime
import configparser
import catalogue_sim
import pcl_measurement


def run_catalogue_sim(pipeline_variables_path, clean=True):

    config = configparser.ConfigParser()
    config.read(pipeline_variables_path)

    save_dir = str(config['simulation_setup']['SIMULATION_SAVE_DIR'])

    # Generate random sample
    print('Initialising random galaxy sample...')
    catalogue_sim.init_rand_sample.execute(pipeline_variables_path)
    print('Done')

    # Create n(z)
    print('Creating n(z) distribution...')
    catalogue_sim.create_nz.execute(pipeline_variables_path)
    print('Done')

    # Generate fiducial Cls
    print('Generating fiducial cosmology...')
    catalogue_sim.generate_cls.execute(pipeline_variables_path)
    print('Done')

    # Convert fiducial Cls to Flask format
    print('Converting fiducial Cls to Flask input format...')
    catalogue_sim.conv_fields_cosmosis_flask.execute(pipeline_variables_path)
    print('Done')

    no_realisations = int(config['simulation_setup']['REALISATIONS'])

    # Run Flask to generate correlated fields on the sky
    for i in range(no_realisations):

        # flask_out_file = save_dir + 'flask/output/iter_{}/flask_out.txt'.format(i+1)
        # sys.stdout = open(flask_out_file)

        print('Running Flask to simulate field maps - Realisation {} / {}'.format(i+1, no_realisations))
        catalogue_sim.run_flask.execute(pipeline_variables_path=pipeline_variables_path, iter_no=i+1)
        print('Done')

        # sys.stdout.close()

        if clean:
            # Measure power spectra and delete maps immediately to save space
            print('Measuring 6x2pt power spectra from simulated maps...')
            pcl_measurement.measure_spectra.execute(pipeline_variables_path=pipeline_variables_path, realisation=i+1)
            print('Done')

            print('Deleting simulated map files...')
            pcl_measurement.clean_products.execute(pipeline_variables_path=pipeline_variables_path, iter_no=i+1)
            print('Done')


def measure_pcls(pipeline_variables_path, clean=True, cov_iter=False):

    config = configparser.ConfigParser()
    config.read(pipeline_variables_path)

    no_realisations = int(config['simulation_setup']['REALISATIONS'])

    if not clean:
        # # Measure power spectra from mock catalogues
        for j in range(no_realisations):

            print('Measuring 6x2pt power spectra from simulated maps... - Realisation {} / {}'.format(j+1, no_realisations))
            pcl_measurement.measure_spectra.execute(pipeline_variables_path=pipeline_variables_path, realisation=j+1)
            print('Done')

            # Delete data (for saving storage space)
            pcl_measurement.clean_products.execute(pipeline_variables_path=pipeline_variables_path, iter_no=j+1)

    # # Average Cls over all realisations
    print('Averaging Cls over realisations...')
    pcl_measurement.av_cls.execute(pipeline_variables_path)
    print('Done')

    # # Convert Cls to bandpowers
    print('Converting Pseudo-Cls to bandpowers...')
    pcl_measurement.measure_cat_bps.execute(pipeline_variables_path)
    print('Done')

    # Convert bandpower data vector to 1D
    print('Combining to joint data vector...')
    pcl_measurement.conv_bps.execute(pipeline_variables_path)
    print('Done')

    if not cov_iter:
        # Calculate numerical covariance matrix
        print('Calculating numerical covariance matrix...')
        pcl_measurement.cov_fromsim.execute(pipeline_variables_path)
        print('Done')

    else:
        # If we want to calculate covariance matrix for different numbers of realisations
        cov_iter_nos = [1, 2, 3]
        for cov_iter_no in cov_iter_nos:
            print('Calculating numerical covariance matrix - {} realisations ...'.format(cov_iter_no))
            pcl_measurement.cov_fromsim.execute_iters(pipeline_variables_path, cov_iter_no)
            print('Done')
#
#
# def run_likelihood(pipeline_variables_path):
#
#     # Run nautilus sampler to perform likelihood analysis
#     likelihood.sampler.execute(pipeline_variables_path)


def main():

    pipeline_variables_path = \
        '/raid/scratch/wongj/mywork/3x2pt/6x2pt_sim/set_config/set_variables.ini'

    # # Create catalogues
    run_catalogue_sim(pipeline_variables_path, clean=True)
    #
    # # # Measure Pseudo-Cls
    measure_pcls(pipeline_variables_path, clean=True, cov_iter=False)

    now = datetime.datetime.now()
    print(now)
    # Perform likelihood analysis

    # run_likelihood(pipeline_variables_path)


if __name__ == '__main__':
    main()
