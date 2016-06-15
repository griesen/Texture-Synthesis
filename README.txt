# Texture-Synthesis
stimulus creation


   SETUP
-----------

This repository is designed to be cloned onto and run on Sherlock. Before running anything, execute the following commands on Sherlock (you can copy paste them, then press enter):

ml load python/2.7.5
easy_install-2.7 --user pip
export PATH=~/.local/bin:$PATH
﻿pip2.7 install --user networkx

You will only have to run this once per Sherlock user.


 EXECUTION
-----------

Order of operations to run pipeline on given set of images (either .png or .jpg):

1) Create a directory and place only the desired input images in this directory (do not add any other images if you don't want to run them)

2) In the root directory, run the following: 
python generate_stimuli.py /path/to/input/folder /path/to/output/folder

The last argument (/path/to/output/folder) is optional. The program will create a folder called "generated". This folder will be created in the directory specified by the second argument (if there is a second argument). Otherwise it will be placed in /path/to/input/folder.

The "generated" folder contains a sub-folder for each image in the input directory, which for each layer K (0 through 4)
in the network will contain:
    * (input filename)_layerK.jpg - the generated image file
    * (input filename)_layerK.npy - stores the actual raw pixel values of the generated image in a .npy file
    * (input filename)_layerK.mat - stores the actual raw pixel values of the generated image in a .mat file
    * (input filename)_layerK_history - not that important. Stores progress made by optimizer every 100 iterations

In addition to these, the folder will also contain the original, cropped and reshaped image that served as input.