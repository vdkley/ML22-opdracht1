import click
import train_model
from loguru import logger
from settings import Settings

logger.add("logging.log")

@click.command()
@click.option("--task")
def main(task: str) -> None:
    presets = Settings()
    
    if task == None:
        print('Select a taks: --task= options are: train')
    else:
        logger.info(f"starting {task}")
    if task == "train" or task == "all":
        train_model.run_trainloop(presets)


if __name__ == "__main__":
    main()