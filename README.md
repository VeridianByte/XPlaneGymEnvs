[中文](https://github.com/Picaun/XPlaneGymEnvs/blob/main/README_zh.md) | [English](https://github.com/Picaun/XPlaneGymEnvs/blob/main/README.md)
# XPlane Gym: A Reinforcement Learning Environment Compatible with X-Plane

XPlaneGymEnvs is an X-Plane flight simulator environment compliant with the OpenAI Gym interface, specifically designed for reinforcement learning research. It provides seamless integration with the X-Plane simulator, supports both discrete and continuous action spaces, and can be used to train agents to perform flight control tasks.

<div align=center>
<img src="https://github.com/VeridianByte/VeridianByte/blob/main/images/XPlaneGymEnvs.gif"/>
</div>

## Installation Requirements

- X-Plane 12 Flight Simulator（Perhaps it will also work in lower versions）
- Python 3.8+
- gymnasium
- numpy

## Installation

```
# Clone the repository
git clone https://github.com/VeridianByte/XPlaneGymEnvs.git
cd XPlaneGymEnvs

# Install the project and its dependencies
pip install -e .
```

## Available Environments

- `XPlane-v0`: Basic environment, configurable as discrete or continuous action space
- `XPlane-custom-v0`: custom environment, Import XPlaneEnv class to personalize the environment

## Quick Start

### 1. Installation

```
pip install -e .
```

### 2. Launch X-Plane

1. Start the X-Plane flight simulator
2. Ensure that the UDP communication port is set to 49000 (default value) in "Settings > Network"
3. Starting a new flight, you can set the weather conditions and flight duration by yourself
   (note that this function is not yet directly controlled with X-Plane 12 in the XPlaneGymEnvs interface)

### 3. use agent_examples
```
cd agent_examples/dqn_example
```
```
python train_dqn.py
```


## Environment Parameter Configuration

Various parameters can be configured when creating the environment:

```python
env = gym.make(
    "XPlane-v0",
    ip='127.0.0.1',                # X-Plane IP address
    port=49000,                    # X-Plane UDP port
    timeout=1.0,                   # Communication timeout
    pause_delay=0.05,              # Action execution pause delay
    starting_latitude=37.558,      # Initial latitude (default near Seoul Gimpo International Airport)
    starting_longitude=126.790,    # Initial longitude (default near Seoul Gimpo International Airport)
    starting_altitude=3000.0,      # Initial altitude
    starting_velocity=100.0,       # Initial velocity
    starting_pitch_range=10.0,     # Initial pitch angle random range
    starting_roll_range=20.0,      # Initial roll angle random range
    random_desired_state=True,     # Whether to use random target attitude
    desired_pitch_range=5.0,       # Target pitch angle random range
    desired_roll_range=10.0,       # Target roll angle random range
    continuous_actions=True
)
```

## Acknowledgements

This project is based on the following open source projects:
- [XPlaneConnectX](https://github.com/sisl/XPlaneConnectX)
- [GYM_XPLANE_ML](https://github.com/adderbyte/GYM_XPLANE_ML)
- [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) 
