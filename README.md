# Lab instructions
## Labs
To pass the course each lab from 1-4 needs to be sent in before their deadlines.

## Project
The last part of this course include a project.

The Deep learning course provides three different problems to be solved; **RNN**, **GAN** or **Detection**. It is allowed to solve more than one project but not required. Each project introduces the topic and tasks to solve. The dataset is provided and visualized to get an idea of the expectations of the results.


The project should be summarised in the Report notebook. 

# Installation
Follow the steps below to setup SSH, Login to Jupyter and setup an virtual environment. The labs uses Tensorboard for logging and instructions is also found below.

## SERVERS
Lund, Amsterdam, Berlin, Copenhagen, Barcelona, Istanbul, Bermingham, Liverpool, Bristol

## SSH
The school provides GPU servers for the students which can be accessed through tools like CMD or Putty.

In order to use the GPU servers the IP of your network need to be whitelisted. It is possible to access the GPU servers within the school network. However, if you want to connect outside the school network, contact helpdesk to whitelist your IP found at whatismyip.com. 

For CMD; Open CMD; Type: ssh -L 8888:localhost:8888 -p 2022 studentID@machinename.hh.se.
The parameter -L allow use to define a server port, local ip and port to access jupyter notebook. In this case, the port 8888. Replace studentID with the school username example: tmpaxr21 and machinename with a server. 

- `ssh -L[desiredlocalport]:localhost:[desiredandfreedistantport] -p 20022 [studentID]@[machinename].hh.se`
- `jupyter notebook --port [desiredandfreedistantport] --no-browser`
- Copy paste the url shown on the terminal to a tab on your browser


## Jupyter Hub
* Login
* Run a notebook
* Free memory

## Virtual Environment
* Conda
* Install requirements

Then simply run the following commands:
- `source activate`
- Install the environment with `conda env create -f deeplearn.yml`
- Do not forget to run `source activate` and `conda activate deeplearn` everytime you login
## Packages
The required packages exists within the requirements.txt. 

## Tensorboard
To user tensorboard we need to add an additional port to the SSH command. Tensorboard use port 6006 (default) so add ssh -L 6006:localhost:6006 8888:localhost:8888.

## Utils
To observe the load of a GPU, open a terminal and write **nvidia-smi**
* Nvidia-smi

