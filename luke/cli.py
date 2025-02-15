import logging
import multiprocessing
import os
import random
import click
import numpy as np
import torch
# from wikipedia2vec.dump_db import DumpDB
# from wikipedia2vec.utils.wiki_dump_reader import WikiDumpReader
from luke.utils.medmentions_db import MedMentionsDB


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # filter out INFO messages from Tensordflow
try:
    # https://github.com/tensorflow/tensorflow/issues/27023#issuecomment-501419334
    from tensorflow.python.util import deprecation

    deprecation._PRINT_DEPRECATION_WARNINGS = False
except ImportError:
    pass

# import luke.pretraining.dataset
import luke.pretraining.medmentions_dataset
import luke.pretraining.train
import luke.utils.entity_vocab
import luke.utils.interwiki_db
import luke.utils.model_utils



@click.group()
@click.option("--verbose", is_flag=True)
@click.option("--seed", type=int, default=None)
def cli(verbose: bool, seed: int):
    fmt = "[%(asctime)s] [%(levelname)s] %(message)s (%(funcName)s@%(filename)s:%(lineno)s)"
    if verbose:
        logging.basicConfig(level=logging.DEBUG, format=fmt)
        logging.getLogger("transformers").setLevel(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO, format=fmt)
        logging.getLogger("transformers").setLevel(level=logging.WARNING)

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)




# 1 
@cli.command()
@click.argument("in_file", type=click.Path(exists=True))
@click.argument("out_file", type=click.Path())
@click.option("--pool-size", default=multiprocessing.cpu_count())
@click.option("--chunk-size", type=int, default=100)
def build_medmentions_db(in_file: str, out_file: str, **kwargs):
    MedMentionsDB.build(in_file, out_file)

# cli.add_command(luke.utils.interwiki_db.build_interwiki_db) # for multilingual, ignore

cli.add_command(luke.utils.entity_vocab.build_entity_vocab) # 2 

cli.add_command(luke.pretraining.medmentions_dataset.build_medmentions_pretraining_dataset) # 3

cli.add_command(luke.pretraining.train.start_pretraining_worker) # 4?

cli.add_command(luke.pretraining.train.pretrain) # 5

cli.add_command(luke.pretraining.train.resume_pretraining)

cli.add_command(luke.utils.model_utils.create_model_archive)


if __name__ == "__main__":
    cli()
