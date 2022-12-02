#!/bin/bash

mkdir ./Data

ndl_rep_id=1uUMgOOdJL_0vrRbrRa8tomwCUhG7e3Zw
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=$ndl_rep_id" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=$ndl_rep_id" -O ./Data/nslKDD.zip && rm -rf /tmp/cookies.txt


unzip ./Data/nslKDD.zip -d ./Data
rm ./Data/nslKDD.zip