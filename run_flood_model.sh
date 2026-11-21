#!/bin/sh
# Flood model: ./run_flood_model.sh STACK_FOLDER OUTPUT_FOLDER [--cpu]
[ -z "$2" ] && { echo 'usage: run_flood_model.sh STACK_FOLDER OUTPUT_FOLDER [--cpu]'; exit 1; }
exec python3 "$(dirname "$0")/predict.py" --stack "$1" --out "$2" $3
