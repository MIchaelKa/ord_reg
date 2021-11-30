import hydra
from omegaconf import DictConfig, OmegaConf
import os

from run import run_training

import logging
logger = logging.getLogger(__name__)

@hydra.main(config_path="config", config_name="base")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    run_training(cfg)
    logger.info("output directory : {}".format(os.getcwd()))
    
if __name__ == "__main__":
    main()