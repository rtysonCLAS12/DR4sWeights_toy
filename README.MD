# DR4sWeights_toy

Code exemplifying how to convert sWeights to definite positive probabilistic drWeights using Decision Trees or Neural Resamplers based on a toy example.

## Requirements

All required libraries are detailed in env_requirements.txt. To install using venvs and pip in a new environment called newenvname:

      python3 -m venv /path/to/env/location/newenvname
      source /path/to/env/location/newenvname/bin/activate.csh
      pip install -r env_requirements.txt

## To Run

In the new environment do:

      python3 train.py

train.py shows an example of how the density ratio estimation technique can be deployed to convert sWeights to drWeights. Several parameters of the toy dataset can be modified, such as for example the number of generated events or the signal to background ratio. The models used to produce the drWeights can be changed, for example from Gradient Boosted Decision Trees, to Histogram Gradient Boosted Decision Trees or Neural Networks. Finally, train.py can also perform boostrapping and repeat the data and generation and training N times. All of these parameters can be found at the top of train.py, to be modified as desired by the user. The comments should explain what variable changes which parameter.

**N.B.:** train.py will output plots to your current directory. At the top of the script you can change the print_dir variable to change the location where the plots are written. You can change the endName variable so that plots are saved as name_endName.png to avoid overwritting plots. By default train.py will save plots as name_GBDTs.png.

In the background, train.py calls on several classes. The plotter class produces plots, the generator class generates the toy data, the performance class does the boostrapping and repeated tests, and the trainer class trains the models. These scripts should not have to be altered, except the to tweak the neural network implementation. This can be found at the top of the trainer class.



