

"""
main runs the experiments
"""

def main(config_file_name):
    runconfig = RunConfig(config_file_name)
    # Experiment
    experiment = Experiment(runconfig)
    experiment.run_experiment()


if __name__ == '__main__':
    if len(sys.argv) > 2:
        raise IOError
    if len(sys.argv) == 1:
        main('config_1')


    else:
        config = sys.argv[1]
        main(config)
