# DR4sWeights_toy

Code exemplifying how to learn sWeights using Decision Trees or Neural Resamplers based on a toy example.

## Requirements

All required libraries are detailed in env_requirements.txt. To install using venvs and pip in a new environment called newenvname:

      python3 -m venv /path/to/env/location/newenvname
      source /path/to/env/location/newenvname/bin/activate.csh
      pip install -r env_requirements.txt

## To Run

In the new environment do:

      python3 train.py

or

      python3 train_neuralResampler.py

Both scripts take ~5 minutes to run.

NB: These scripts will output plots to your current directory. At the top of both scripts you can change the print_dir variable to change the location where the plots are written. You can change the endName variable so that plots are saved as name_endName.png to avoid overwritting plots.

