[//]: # (Image References)

[image1]: ./Plots/Trained_Agent.gif "Trained Agent"


# Project 3: Collaboration and Competition

## Introduction

For this project, we will work with the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment.

![Trained Agent][image1]

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.

### Problem To Solve

We need to train the two tennis players to collaborate to maximize the overall score. To solve this task we implement a Multi-Agent Reinforcement Learning Algorithm. 


### Environment
The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.  Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. 

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single **score** for each episode.

The environment is considered solved, when the average (over 100 episodes) of those **scores** is at least +0.5.


## Installation

### Python Dependencies

- The software requires to install Python (3.6.1 or higher). We advocate to create a new environment with Python 3.6
     
     - Linux or Mac:
  
    ```sh
    conda create --name drlnd python=3.6
    source activate drlnd
    ```
    
    - Windows:
  
    ```sh
    conda create --name drlnd python=3.6 
    activate drlnd
    ```

- Clone the repository, and navigate to the python/ folder. Then, install several dependencies.

    ```sh
        git clone https://github.com/udacity/deep-reinforcement-learning.git
        cd deep-reinforcement-learning/python
        pip install .
    ```
- Create and activate IPython kernel for the drlnd environment.

    ```sh
    python -m ipykernel install --user --name drlnd --display-name "drlnd"
    ```
    
    In the jupyter notebook instance the kernel is activated from the dropdown menu _Kernel_



### Unity Packages
Besides the Python ML library `PyTorch` you will need to install the Unity Packages and Environments plus the relevant Python Packages following the instructions in [Unity ML-Agents](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation.md)

The ML-Agents Toolkit contains several components:

- Unity package `com.unity.ml-agents` contains the
  Unity C# SDK that will be integrated into your Unity project.  This package contains
  a sample to help you get started with ML-Agents.
  
- Unity package `com.unity.ml-agents.extensions` contains experimental C#/Unity components that are not yet ready to be part
  of the base `com.unity.ml-agents` package. `com.unity.ml-agents.extensions`
  has a direct dependency on `com.unity.ml-agents`.
  
- Three Python packages:
    - `mlagents` contains the machine learning algorithms that
      enables you to train behaviours in your Unity scene. Most users of ML-Agents
      will only need to directly install `mlagents`.
    - `mlagents_envs` contains a Python API to interact with
      a Unity scene. It is a foundational layer that facilitates data messaging
      between Unity scene and the Python machine learning algorithms.
      Consequently, `mlagents` depends on `mlagents_envs`.
    - `gym_unity` provides a Python-wrapper for your Unity scene
      that supports the OpenAI Gym interface.
      
### Unity Environment

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux_NoVis.zip) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)

2. Place the file in the DRLND GitHub repository, in the `Project_Collaboration_Competition/` folder, and unzip (or decompress) the file. 


## Instructions

You can either follow the steps in the python notebook `Tennis.ipynb` or run it locally with more diagnostic including tensorboard from `main.py` or via shell script `run_training.sh` and `run_tensorboard.sh`.

### Directory Structure

1. Main Python Notebook `Tennis.ipynb`

2. Main Python Code `main.py`

3. Multi-Agent Actor-Critic Deep Deterministic Policy from `maddpg.py` imports single agents from `ddpg.py` and eploratory noise process from `OUNoise.py`.

4. Experiences are stored adn sampled from `replaybuffer.py`

5. Other utilies functions imported from ``utilities.py`

7. The PyTorch file `./episode-latest.pt` contains the dictionary of the trained models with weights of the two actor and critic networks.  



### GPU
If Cuda library available PyTorch will automatically run on GPU otherwise on cpu.

## License
MIT License

Copyright (c) [2021] [A.M.C.S.]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


