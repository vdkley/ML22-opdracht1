import click
import project.optimizer_test as optimizer_test
import project.learningrate_test as learningrate_test
import project.filtertest as filtertest
import project.kernelsizetest as kernelsizetest
import project.experiments as experiments
import project.experiments_combine as experiments_combine
import project.resnet_test as resnet_test
from loguru import logger
from settings_project import Settings

logger.add("logging.log")

@click.command()
@click.option("--task")
@click.option("--name")
def main(task: str, name: str) -> None:
    presets = Settings()
    
    # learningrate_test.run_trainloop(presets)

    if task == None:
        print('Select a taks: --task= options are: optimtest')
    else:
        logger.info(f"starting {task}")
    if task == "optimtest":
        optimizer_test.run_trainloop(presets)
    if task == "lrtest":
        learningrate_test.run_trainloop(presets)
    if task == "filtertest":
        filtertest.run_trainloop(presets)
    if task == "ksize":
        kernelsizetest.run_trainloop(presets)
    if task == "exp" and name == None:
        print('start experiment by giving a name --name=')
    if task == "exp" and name != None:
        experiments.run_experiment(presets,name)
    if task == "combine":
        experiments_combine.run_experiment(presets,name)
    if task == "runs":
        experiments_combine.run_experiment_runs(presets,name)
    if task == "resnet":
        resnet_test.run_trainloop(presets)                     

if __name__ == "__main__":
    main()