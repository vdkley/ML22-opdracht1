import click
import optimizer_test
import learningrate_test
import filtertest
import kernelsizetest
import experiments
import experiments_combine
from loguru import logger
from settings import Settings

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
    if task == "exp2":
        experiments_combine.run_experiment(presets)
                        

if __name__ == "__main__":
    main()