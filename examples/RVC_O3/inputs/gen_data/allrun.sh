#!/bin/bash -i

path_to_scripts=/home/zanardi/Codes/ML/RONEK/ronek/scripts/

echo -e "\nRunning 'gen_data' script ..."
python -u $path_to_scripts/gen_data.py --inpfile gen_data.json
