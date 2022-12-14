import click
from loguru import logger

import project.experiments as experiments
import project.experiments_combine as experiments_combine
import project.filtertest as filtertest
import project.kernelsizetest as kernelsizetest
import project.learningrate_test as learningrate_test
import project.optimizer_test as optimizer_test
import project.resnet_test as resnet_test
from settings_project import Settings

logger.add("logging.log")


@click.command()
@click.option("--task")
@click.option("--name")
def main(task: str, name: str) -> None:
    presets = Settings()

    # learningrate_test.run_trainloop(presets)

    if task is None:
        print("Select a taks: --task= options are: optimtest")
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
    if task == "exp" and name is None:
        print("start experiment by giving a name --name=")
    if task == "exp" and name is not None:
        experiments.run_experiment(presets, name)
    if task == "combine":
        experiments_combine.run_experiment(presets, name)
    if task == "runs":
        experiments_combine.run_experiment_runs(presets, name)
    if task == "resnet":
        resnet_test.run_trainloop(presets)


if __name__ == "__main__":
    main()
