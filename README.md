# Artificial Flying Objects Dataset
Artificial Flying Objects Dataset

- 128x128x3 
- Around 30 frames in each sequence, 
- In total 10K data for training, 2K for validation and 2K for testing

![Example Gif](/images/ArtificialFlyingObjects.gif)



# Miniconda Installation
To install miniconda firstly download the installer to the desired machine [here](https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh).

Then simply run the following commands:

- `chmod +x Miniconda3-latest-Linux-x86_64.sh` <- This will make the `.sh` executable
- `conda env create --file conda_environment.yml`
- Wait until it finish installing all necessary packages and then activate your new environment using `conda activate lab`
- Just a friendly reminder, you have a quota of 10GB, so please do not do update all to your environment/miniconda.


# Launching your jupyter

You can either use the instructions from the [Instructions](LabInstruction.PDF) to use MobaXterm or you can 
create your own tunnel via ssh using the following commands:

- `ssh -L[desiredlocalport]:localhost:[desiredandfreedistantport] -p 20022 [studentID]@[machinename].hh.se`
- `jupyter notebook --port [desiredandfreedistantport] --no-browser`
- Copy paste the url shown on the terminal to a tab on your browser
