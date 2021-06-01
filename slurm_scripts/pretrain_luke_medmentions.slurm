#!/bin/bash

#SBATCH --job-name=train_luke_medmention
#SBATCH --output=train_luke_medmen-%j.out
#SBATCH --ntasks=6
#SBATCH --mem=60G
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --partition=interactive,general,gpu
#SBATCH --time=1-00:00:00

cd /home/vs428/Documents/luke/luke

module restore fosscuda111

# using your anaconda environment
source activate el_env111



python cli.py build-medmentions-db /home/vs428/Documents/luke/tests/test_data/1.txt /home/vs428/Documents/luke/tests/test_data/1out.json
python cli.py build-entity-vocab /home/vs428/Documents/luke/tests/test_data/1out.json /home/vs428/Documents/luke/tests/test_data/entity_vocab.json
python cli.py build-medmentions-pretraining-dataset /home/vs428/Documents/luke/tests/test_data/1out.json microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract /home/vs428/Documents/luke/tests/test_data/entity_vocab.json /home/vs428/Documents/luke/tests/dataset
python cli.py pretrain  /home/vs428/Documents/luke/tests/dataset /home/vs428/Documents/luke/tests/dataset --batch-size 128 --gradient-accumulation-steps 64  --bert-model-name microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract
