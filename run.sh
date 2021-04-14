#!/bin/sh　-x

# ex) source run.sh [cuda]

echo -n "Please select Mode
0: Normal
input->"
read mode

echo -n "Please enter the name of experiment
input->"
read exp_name

# activate venv
source .venv/bin/activate

# git pull
git pull

git log -1

# run
# python main.py $1