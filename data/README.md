# Data Preparing

You have to extract 2D slices from 3D volume of your data sets.
You have to convert them to numpy format and save as case0001_slice000.npz, case0001_slice001.npz, ... in the "data/Synapse" directory.

You also have to create a txt file named train in the "data.lists_Synapse" directory that lists all the 2D slices you want to use for the training. 
Example :   
case0001_slice000  
case0001_slice001  
case0001_slice002  
...



The directory structure of the whole project is as follows:  


data

    └──Synapse
        └──case0001_slice000.npz
        └──case0001_slice001.npz
        ...
    └──lists_Synapse
        └──train.txt


