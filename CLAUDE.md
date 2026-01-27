# Machine learning repo to predict gene expression

This repository is called yorzoi, and it is used to predict gene expression of yeast. It is based on the Borzoi model.

# Folder structure

Data is stored in the data folder. 
- Categorized is yeast with different names
- type_splits is split into types: Human yac, and other yeast
- samples.pkl is the folder that stores the sequences and relevant data, and it also stores a path that points to the track_values folder, which stores the gene expression of activity across its length
- train.txt is the MPRA data of yeast from the dream challenge repo, it stores sequence, and their expression. We want to use this to pretrian the Borzoi model
- The xx_names files are just the names in a text file

yorzoi folder stores the main model

