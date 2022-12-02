#!/bin/bash

mkdir ./Pretrains

ndl_rep_id=1_bMwvYHb45PGbqcYRuGKpivLFbMgX9DX
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=$ndl_rep_id" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=$ndl_rep_id" -O ./Pretrains/models.zip && rm -rf /tmp/cookies.txt


unzip ./Pretrains/models.zip -d ./Pretrains
rm ./Pretrains/models.zip