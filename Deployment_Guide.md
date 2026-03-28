# Profiling AI Software Bootcamp

The Profiling AI Software Bootcamp covers the process and tools for profiling AI and machine learning applications to fully utilize high-performance systems. Attendees will learn to profile applications using NVIDIA Nsight™ Systems, a system-wide performance analysis tool; analyze and identify optimization opportunities; and improve application performance to scale efficiently across systems of any size and number of CPUs and GPUs. Additionally, this bootcamp will walk through the system topology to learn the dynamics of FP8 precision, multi-GPU, and multi-node connections and architecture.

## Deploying the Labs

### Prerequisites

To run this tutorial, you will need a DGX machine with a minimum of NVIDIA Hopper GPU Architecture (H100).

- You need a Linux Machine.
- Install latest [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install/linux-install). 
- Install the latest [Docker](https://docs.docker.com/engine/install/) or [Singularity](https://sylabs.io/docs/).
- Install NVIDIA [Nsight Systems](https://developer.nvidia.com/nsight-systems).


### Tested environment

We tested and ran all labs on a DGX machine equipped with a H100 GPUs (80GB).


### Deploying with conda (Lab 1, 2, & 3)

#### Creating and running on conda env

```bash

#create conda env

conda create -n env_profiler python=3.12

#activate the env

conda activate env_profiler

# Install the dependencies.

cd ~/Profiling-AI-Software-Bootcamp

pip install -r requirements.txt

#running the Jupyter Notebook

jupyter-lab --no-browser --allow-root --ip=0.0.0.0 --port=8888 --NotebookApp.token="" --notebook-dir=./workspace

```

 
### Deploying with container (Lab 4)

You can deploy this material using Conda, Docker or Apptainer containers. Please refer to the respective sections for the instructions.


#### Running Docker Container 

To run the Labs, you will need access to 2 nodes(at least 4 GPUs per node). Build a Docker container by following these steps:  

- Open a terminal window and navigate to the directory where `Dockerfile` file is located (`cd ~/Profiling-AI-Software-Bootcamp`)
- To build the docker container, run : `sudo docker build -f Dockerfile --network=host -t <imagename>:<tagnumber> .`, for instance: 


```bash

sudo docker build -f Dockerfile --network=host -t tecont:v1 .

```
- To run the built container : 

```bash

docker run --rm -it --gpus all -p 8888:8888 --ipc=host --ulimit memlock=-1 --ulimit stack=67108864
 -v ./workspace:/workspace tecont:v1 
 jupyter-lab --no-browser --allow-root --ip=0.0.0.0 --port=8888 --NotebookApp.token="" --notebook-dir=/workspace

```


flags:
- `--rm` will delete the container when finished.
- `-it` means run in interactive mode.
- `--gpus` option makes GPUs accessible inside the container.
- `-v` is used to mount host directories in the container filesystem.
- `--network=host` will share the host’s network stack to the container.
- `-p` flag explicitly maps a single port or range of ports.


Open the browser at `http://localhost:8888` and click on the `start_here.ipynb`. Go to the table of content and click on Lab 1: `Preprocessing Multi-turn Conversational Dataset`.
As soon as you are done with the rest of the labs, shut down jupyter lab by selecting `File > Shut Down` and the container by typing `exit` or pressing `ctrl d` in the terminal window.



#### Running Singularity Container


- Build the Labs Singularity container with: 

```bash

apptainer build --fakeroot --sandbox tecont.simg Singularity


```

- To run the built container: 

```bash

singularity run --nv -B workspace:/workspace tecont.simg 
jupyter-lab --no-browser --allow-root --ip=0.0.0.0 --port=8888 --NotebookApp.token="" --notebook-dir=/workspace

```
 
 The `-B` flag mounts local directories in the container filesystem and ensures changes are stored locally in the project folder. Open jupyter lab in the browser: http://localhost:8888

You may start working on the labs by clicking the `start_here.ipynb` notebook.

When you finish these notebooks, shut down jupyter lab by selecting `File > Shut Down` in the top left corner, then shut down the Singularity container by typing `exit` or pressing `ctrl + d` in the terminal window.




## Known issues

