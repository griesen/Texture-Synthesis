# Texture-Synthesis
stimulus creation

Order of operations to run pipeline on given set of images (either .png or .jpg):
1) Create directory where those images live (do not add any other images if you don't want to run them)
2) In the root directory, run the following: python generate_stimuli.py /path/to/input/folder /path/to/output/folder

The last argument (/path/to/output/folder) is optional. The program will create a folder called "genereated."
This folder will go wherever specified by the second argument, otherwise it will be placed in /path/to/input/folder.

The "generated" folder contains a sub-folder for each image in the input directory, which for each layer K (0 through 4)
in the network will contain:
    * (input filename)_layerK.jpg - the generated image file
    * (input filename)_layerK.npy - stores the actual raw pixel values of the generated image
    * (input filename)_layerK_history - not really that important. Stores progress made by optimizer during the process
