#!/usr/bin/bash

MATRIX_FOLDER=/home/ubuntu/mm/

# compile mtx_to_csr util
cd ../utils
make to_csr
TO_CSR_BIN=$PWD/to_csr.hw

mkdir -p $MATRIX_FOLDER
cd $MATRIX_FOLDER

# consph graph, 83,334 x 83,334 matrix, 6,010,480 non-zeros; symmetric
if [ ! -f "consph/consph.csr" ]; then
    wget https://www.cise.ufl.edu/research/sparse/MM/Williams/consph.tar.gz
    tar xf consph.tar.gz
    rm consph.tar.gz
    "$TO_CSR_BIN" True consph/consph.mtx consph/consph.csr
else
    echo "Skip consph"
fi

# PARSEC/Ga41As41H72 graph, 268,096 x 268,096 matrix, 18,488,476 non-zeros; symmetric
if [ ! -f "Ga41As41H72/Ga41As41H72.csr" ]; then 
    wget https://www.cise.ufl.edu/research/sparse/MM/PARSEC/Ga41As41H72.tar.gz
    tar xf Ga41As41H72.tar.gz
    rm Ga41As41H72.tar.gz
    "$TO_CSR_BIN" True Ga41As41H72/Ga41As41H72.mtx Ga41As41H72/Ga41As41H72.csr
else
    echo "Skip Ga41As41H72"
fi

# HB/steam1 graph, 240 x 240 matrix, 2,248 non-zeros; symmetric
if [ ! -f "steam1/steam1.csr" ]; then
    wget https://www.cise.ufl.edu/research/sparse/MM/HB/steam1.tar.gz
    tar xf steam1.tar.gz
    rm steam1.tar.gz
    "$TO_CSR_BIN" False steam1/steam1.mtx steam1/steam1.csr
else
    echo "Skip steam1"
fi

# Schenk/nlpkkt200, 16,240,000 x 16,240,000 matrix, 440,225,632 non-zeros; symmetric
if [ ! -f "nlpkkt200/nlpkkt200.csr" ]; then 
    wget https://www.cise.ufl.edu/research/sparse/MM/Schenk/nlpkkt200.tar.gz
    tar xf nlpkkt200.tar.gz
    rm nlpkkt200.tar.gz
    "$TO_CSR_BIN" True nlpkkt200/nlpkkt200.mtx nlpkkt200/nlpkkt200.csr
else
    echo "Skip nlpkkt200"
fi

# Schenk/nlpkkt240, 27,993,600 x 27,993,600 matrix, 760,648,352 non-zeros; symmetric
if [ ! -f "nlpkkt240/nlpkkt240.csr" ]; then
#    wget https://www.cise.ufl.edu/research/sparse/MM/Schenk/nlpkkt240.tar.gz
#    tar xf nlpkkt240.tar.gz
#    rm nlpkkt240.tar.gz
#    "$TO_CSR_BIN" True nlpkkt240/nlpkkt240.mtx nlpkkt240/nlpkkt240.csr
    echo "Skip nlpkkt240 manually"
else
    echo "Skip nlpkkt240"
fi

# USA-road
if [ ! -f "USA-road-d.USA.csr" ]; then
    wget http://www.diag.uniroma1.it/challenge9/data/USA-road-d/USA-road-d.USA.gr.gz
    gunzip USA-road-d.USA.gr.gz
    "$TO_CSR_BIN" True USA-road-d.USA.gr USA-road-d.USA.csr
else
    echo "Skip USA-road-d.USA"
fi
