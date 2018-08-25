# Extended Deep Q-Learning for Multilayer Perceptron for solving Deep Reinforcement Learning Nanodegree - Project 1: Navigation


This project includes the code for an extended version of the Deep Q-Learning algorithm which I wrote to solve the Project 1: Navigation of the [Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893) @ Udacity. My version is inspired by the code of the vanilla DQN algorithm provided by Udacity: https://github.com/udacity/deep-reinforcement-learning


Deep Q-Learning for Multilayer Perceptron<br>
\+ Fixed Q-Targets<br>
\+ Experience Replay<br>
\+ Reward Clipping<br>
\+ Gradient Clipping<br>
\+ Double Deep Q-Learning<br>
\+ Dueling Networks<br>

For more information on the implemented features refer to Project_Navigation_DRLND.ipynb. The notebook includes a summary of all essential concepts used in the code. It also contains an easy to use way to run the project.

### Project 1: Navigation - Details:

The goal of this project was to train an agent to navigate in a large 3D square world to collect bananas. The agent has to collect as many yellow bananas as possible while avoiding blue bananas. 

[//]: # (Image References)

[image1]: https://raw.githubusercontent.com/cpow-89/Deep-Reinforcement-Learning-Nanodegree---Project-1-Navigation/master/images/Extended_Dqn_Banana_config.gif "Trained Agents1"

#### Trained Agent

![Trained Agents1][image1]

##### Reward:
- for collecting a yellow banana the agent receives +1 reward
- for collecting a blue banana, the agent receives -1 reward

##### Search Space
- the state space has 37 dimensions 
     - it includes the agent's velocity, along with a ray-based perception of objects around the agent's forward direction
- the action space has 4 dimensions
    - move forward(0), move backward(1), turn left(2), turn right(3)

##### Task
- the task is episodic
- the agent has to collect as many yellow bananas as possible while avoiding blue bananas
- to solve the environment, the agent must get an average score of +13 over 100 consecutive episodes
    - I set the bar a little higher and let my agent train till an average score of +15 over 100 consecutive episodes
        - due to the time limit, it is tough to get higher rewards than +15 over 100 consecutive 

### Getting Started

1. Create (and activate) a new environment with Python 3.6.

> conda create --name env_name python=3.6<br>
> source activate env_name

2. Download the environment from one of the links below and place it into \Banana_Environment


- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    
- your folder should now look something like this:

\Banana_Environment<br>
&nbsp;&nbsp;&nbsp;&nbsp; \Banana_Data  <br>
&nbsp;&nbsp;&nbsp;&nbsp; \Banana.x86<br>
&nbsp;&nbsp;&nbsp;&nbsp; \Banana.x86_64<br>

3. Install Sourcecode dependencies

> conda install -c rpi matplotlib <br>
> conda install -c pytorch pytorch <br>
> conda install -c anaconda numpy <br>

- unityagents is also required
    - an easy way to get this is to install the Deep Reinforcement Learning Nanodegree with its dependencies
    
> git clone https://github.com/udacity/deep-reinforcement-learning.git<br>
> cd deep-reinforcement-learning/python<br>
> pip install .<br>

### Instructions

You can run the project via Project_Navigation_DRLND.ipynb or by running the main.py file through the console.



open the console and run: python main.py -c "your_config_file".json 
optional arguments:

-h, --help

    - show help message
    
-c , --config

    - Config file name - file must be available as .json in ./configs
    
Example: python main.py -c "Extended_Dqn_Banana_config".json 

#### Config File Description

**"general"** : <br>
> "env_name" : "Banana_from_sensor_data", # The gym environment name you want to run<br>
> "env_path" : ["Banana_Linux", "Banana.x86_64"], # path to the environment files<br>
> "checkpoint_dir": ["checkpoints"], # checkpoint file direction<br>
> "seed": 0, # random seed for numpy, gym and pytorch<br>
> "state_size" : 37, # number of states<br>
> "action_size" : 4, # number of actions<br>
> "average_score_for_solving" : 15.0 # border value for solving the task<br>

**"train"** : 
> "nb_episodes": 2000, # max number of episodes<br>
> "batch_size" : 256, # memory batch size<br>
> "epsilon_high": 1.0, # epsilon start point<br>
> "epsilon_low": 0.01, # min epsilon value<br>
> "epsilon_decay": 0.995, # epsilon decay<br>
> "run_training" : true # do you want to train? Otherwise run a test session<br>

**"agent"** :
> "learning_rate": 0.001, # model learning rate<br>
> "gamma" : 0.99, # reward weight<br>
> "tau" : 0.001, # soft update factor<br>
> "update_rate" : 4 # interval in which a learning step is done<br>

**"buffer"** :
> "size" : 100000 # experience replay buffer size<br>

**"model"** :
> "fc1_nodes" : 256, # number of fc1 output nodes<br>
> "fc2_adv" : 256, # number of fc2_adv output nodes<br>
> "fc2_val" : 256 # number of fc2_val output nodes<br>
