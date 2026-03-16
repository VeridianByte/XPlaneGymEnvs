[中文](https://github.com/Picaun/XPlaneGymEnvs/blob/main/README_zh.md) | [English](https://github.com/Picaun/XPlaneGymEnvs/blob/main/README.md)
# XPlane Gym：兼容 X-Plane 的强化学习环境

XPlaneGymEnvs 是一个符合 OpenAI Gym 接口的 X-Plane 飞行模拟器环境，专门为强化学习研究设计。与 X-Plane 模拟器无缝集成无需第三方插件，支持离散和连续动作空间，可用于训练智能体执行飞行控制任务。

<div align=center>
<img src="https://github.com/Picaun/Picaun/blob/main/images/XPlaneGymEnvs.gif"/>
</div>

## 安装要求

* X-Plane 12 飞行模拟器（目前只在此版本测试过）
* Python 3.8+
* gymnasium
* numpy

## 安装步骤

```bash
# 克隆仓库
git clone https://github.com/Picaun/XPlaneGymEnvs.git
cd XPlaneGymEnvs

# 安装项目及依赖
pip install -e .
```

## 可用环境

* `XPlane-v0`：基础环境，可配置为离散或连续动作空间
* `XPlane-custom-v0`：自定义环境，通过``` import XPlaneEnv ```类来自定义环境

## 快速开始

### 1. 安装

```bash
pip install -e .
```

### 2. 启动 X-Plane

1. 打开 X-Plane 飞行模拟器
2. 确认在 **设置 > 网络** 中，UDP 通信端口设为 `49000`（默认值）
3. 开始新飞行，可自行设置天气和飞行时间
   （注意：目前 XPlaneGymEnvs 接口还不能直接控制 X-Plane 12 的这些功能）

### 3. 使用agent_examples

```bash
cd agent_examples/dqn_example
python train_dqn.py
```

## 环境参数配置

创建环境时可以配置多种参数：

```python
env = gym.make(
    "XPlane-v0",
    ip='127.0.0.1',                # X-Plane IP 地址
    port=49000,                    # X-Plane UDP 端口
    timeout=1.0,                   # 通信超时时间
    pause_delay=0.05,              # 动作执行延迟
    starting_latitude=37.558,      # 初始纬度（默认首尔金浦国际机场附近）
    starting_longitude=126.790,    # 初始经度（默认首尔金浦国际机场附近）
    starting_altitude=3000.0,      # 初始高度
    starting_velocity=100.0,       # 初始速度
    starting_pitch_range=10.0,     # 初始俯仰角随机范围
    starting_roll_range=20.0,      # 初始横滚角随机范围
    random_desired_state=True,     # 是否随机目标姿态
    desired_pitch_range=5.0,       # 目标俯仰角随机范围
    desired_roll_range=10.0,       # 目标横滚角随机范围
    continuous_actions=True
)
```

## 致谢

本项目基于以下开源项目：

* [XPlaneConnectX](https://github.com/sisl/XPlaneConnectX)
* [GYM\_XPLANE\_ML](https://github.com/adderbyte/GYM_XPLANE_ML)
* [Gymnasium](https://github.com/Farama-Foundation/Gymnasium)
