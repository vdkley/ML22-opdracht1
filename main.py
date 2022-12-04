import click
import optimizer_test
import learningrate_test
import filtertest
import kernelsizetest
from loguru import logger
from settings import Settings

logger.add("logging.log")

@click.command()
@click.option("--task")
def main(task: str) -> None:
    presets = Settings()
    
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
        


if __name__ == "__main__":
    main()