#!/bin/bash

echo "Downloading SIFTS UniProt-PDB mapping file..."
wget https://ftp.ebi.ac.uk/pub/databases/msd/sifts/flatfiles/csv/uniprot_pdb.csv.gz
gunzip uniprot_pdb.csv.gz

