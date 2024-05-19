# Model-Training-Framework
This repository is a framework developed using Pytorch Lightning that acts as a starter code for training custom models.

## Development Environment
This project is developed with Ubuntu 20.04 LTS with NVIDIA GTX 1060 Graphics Card and Pytorch Lightning docker container.

## Setting up the Environment
- Install docker by following instructions from the [Docker](https://docs.docker.com/engine/install/ubuntu/) website.
- Pull the latest pytorch lightning docker image using the command ```docker pull pytorchlightning/pytorch_lightning:latest```
- Open the terminal and enter ```nvidia-smi``` to see your GPU and CUDA version.
- Now install Nvidia Containter ToolKit by following the steps from this [link](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html). Installation and Configuration steps are sufficient. (Installing with Apt and Configuring docker). 
- Open a new terminal on the Host Machine and run the command ```docker run --privileged --rm -it --gpus all -v path_to_the_folder_to_mount:destination_path_in_the_container pytorchlightning/pytorch_lightning bash```. 
    - Modify the **path_to_the_folder_to_mount** and **destination_path_in_the_container** according to your choice. For example: ```/media/user/SSD:/media/SSD```
- The above docker run command opens the container in the terminal. Now you can run ```nvidia-smi``` in the container to see the GPUs and CUDA version.
- Now you can download and access the data available in the drive mounted to your docker container. Any new files added/moved/modified will also reflect in your host machine folder structure.
- Clone this repository (either from host machine or docker container) using the command ```git clone https://github.com/okritvik/Model-Training-Framework```
- Have fun exploring and creating state of the art models!

## Developer Information
- Kumara Ritvik Oruganti, A masters graduate in Robotics from The University of Maryland, College Park, currently an Embedded Software Engineer (SDE-2). Visit [Website](https://www.okritvik.com) for more information!