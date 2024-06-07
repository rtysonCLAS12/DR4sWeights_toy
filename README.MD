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

train.py uses a Hist Gradient Boosting Classifier (HistGBDT) by default which is very fast, this can be changed manually in the script to a Gradient Boosting Classifier (GBDT) which is slower. train.py takes less than a minute to run with the HistGBDT and about 3-4 minutes with the GBT. train_neuralResampler.py takes about 5 minutes to run.

NB: These scripts will output plots to your current directory. At the top of both scripts you can change the print_dir variable to change the location where the plots are written. You can change the endName variable so that plots are saved as name_endName.png to avoid overwritting plots. By default train.py will save plots as name_HistGBDTs.png and train_neuralResampler.py will save plots as name_nr.png.

