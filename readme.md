## Machine Learning Engineer Nanodegree
### Specializations
### Project: Capstone Proposal and Capstone Project

### Problem Definition

This capstone project implements a Deep Deterministic Policy Gradients algorithm to land a lunar lander in a simulated OpenAI Environment (Gym).

### STATE VARIABLES
The state consists of the following variables:
```
- x position
- y position
- x velocity
- y velocity
- angle
- angular velocity
- first leg ground contact indicator
- second leg ground contact indicator
```

### ACTION VARIABLES
Lunar Lander continuous action space:
```
Minimum  = -1.0
Maximum  = +1.0

Actions = (main, directional)

ON Conditions:
- Lunar Lander Directional Engine - Left Thruster = [-1, 0.5] 
- Lunar Lander Directional Engine - Right Thruster = [0.5, 1]
- Lunar Lander Main Engine - Vertical = [0, 0.5]

Engines are OFF otherwise.

```
### Prerequisites

List of libraries needed to run the project: 

First install gym

```
pip install gym
```

This project depends on OpenAI gym and Box2d.
A full installation of gym is recommended.
For this development, the following versions where used:

Name                    Version                
gym                       0.11.0                   
keras                     2.2.2             
matplotlib                2.2.3            
notebook                  5.6.0                 
numpy                     1.16.1             
pandas                    0.23.4          
python                    3.5.6            
scikit-learn              0.20.0         
scipy                     1.2.1           
tensorflow                1.12.0             

```

### Authors

* **Jean Ricardo Rusczak**

References: https://gym.openai.com/envs/LunarLanderContinuous-v2/

### License

This project is licensed under the MIT License.
