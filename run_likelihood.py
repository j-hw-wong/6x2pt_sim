import time
import datetime
import likelihood


def main():

    pipeline_variables_path = \
        '/raid/scratch/wongj/mywork/3x2pt/6x2pt_sim/set_config/set_variables.ini'

    now = datetime.datetime.now()
    # Perform likelihood analysis

    start_time = time.time()

    print('Performing likelihood analysis...')

    print(now)
    likelihood.sampler.execute(pipeline_variables_path)
    print('Done')
    print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == '__main__':
    main()
