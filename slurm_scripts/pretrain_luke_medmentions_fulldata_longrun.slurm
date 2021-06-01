#!/bin/bash

#SBATCH --job-name=train_luke_medmention_long
#SBATCH --output=train_luke_medmenlong-%j.out
#SBATCH --ntasks=6
#SBATCH --mem=80G
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --partition=interactive,general,gpu
#SBATCH --time=1-00:00:00

cd /home/vs428/Documents/luke/luke

module restore fosscuda111

# using your anaconda environment
source activate el_env111


python cli.py build-medmentions-db /home/vs428/project/MedMentions/full/data/corpus_pubtator.txt /home/vs428/project/MedMentions/full/data/corpus_pubtator_db.json
echo "Built MedMentions DB"

python cli.py build-entity-vocab /home/vs428/project/MedMentions/full/data/corpus_pubtator_db.json /home/vs428/project/MedMentions/full/pretraining/entity_vocab.json
echo "Built Entity Vocab"

python cli.py build-medmentions-pretraining-dataset /home/vs428/project/MedMentions/full/data/corpus_pubtator_db.json microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract /home/vs428/project/MedMentions/full/pretraining/entity_vocab.json /home/vs428/project/MedMentions/full/pretraining
echo "Built Pretraining Dataset"
echo "pretraining num steps: 4392"

python cli.py pretrain  /home/vs428/project/MedMentions/full/pretraining /home/vs428/project/MedMentions/full/pretraining  --batch-size 1024 --gradient-accumulation-steps 512  --bert-model-name microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract --warmup-steps 900 --original-adam --num-epochs 50  --log-dir 'runs/fullmedmentions_warmup900_origadam_epochs50' 
echo "Ran Pretraining"
