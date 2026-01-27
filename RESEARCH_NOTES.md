# Aerial Autonomy Stack - Research Notes

## Overview

This document captures observations about the aerial-autonomy-stack (AAS) codebase structure. The goal is to understand the system before making any changes.

---

## High-Level Architecture

The stack is organized into **4 main components**:

```
                    ┌──────────────────────────────────────────────────────────┐
                    │            AERIAL AUTONOMY STACK                         │
                    └──────────────────────────────────────────────────────────┘
                                            │
        ┌───────────────┬───────────────────┼───────────────────┬──────────────┐
        ▼               ▼                   ▼                   ▼              ▼
   ┌─────────┐    ┌──────────┐       ┌───────────┐       ┌──────────┐   ┌─────────┐
   │SIMULATION│   │ AIRCRAFT │       │  GROUND   │       │ AAS-GYM  │   │ SCRIPTS │
   │          │   │          │       │           │       │          │   │         │
   │ Gazebo   │   │ ROS2     │       │ ROS2      │       │ Gymnasium│   │ Docker  │
   │ PX4/Ardu │   │ Autonomy │       │ Monitoring│       │ Wrapper  │   │ Build   │
   │ Models   │   │ Stack    │       │           │       │          │   │         │
   └─────────┘    └──────────┘       └───────────┘       └──────────┘   └─────────┘
```

---

## 1. SIMULATION (`simulation/`)

**Purpose**: Contains everything needed to run the simulated environment.

**Key Contents**:
- **Aircraft Models** (6 models):
  - `x500` - PX4 quadcopter
  - `standard_vtol` - PX4 VTOL
  - `iris_with_ardupilot` - ArduPilot quadcopter
  - `alti_transition_quad` - ArduPilot VTOL
  - `sensor_camera` - Simulated RGB camera
  - `sensor_lidar` - Simulated LiDAR

- **Worlds** (12+ environments):
  - `impalpable_greyness.sdf` - Empty/minimal world (default)
  - `apple_orchard.sdf` - GIS-based fruit orchard
  - `shibuya_crossing.sdf` - Urban environment

- **Simulation Scripts**:
  - `gz_wind.py` - Apply wind effects
  - `gz_step.py` - Manual world stepping (for RL)

**Observation**: The simulation supports both PX4 and ArduPilot autopilots, with SITL (Software-In-The-Loop) capabilities.

---

## 2. AIRCRAFT (`aircraft/`)

**Purpose**: The autonomy stack that runs ON the drone (or simulated drone).

**ROS2 Packages**:

### 2.1 `autopilot_interface` (C++)
- High-level flight control abstraction
- Provides ROS2 **Actions**: Takeoff, Land, Orbit, Offboard
- Two implementations:
  - `px4_interface.cpp` (~2000+ lines)
  - `ardupilot_interface.cpp`
- Handles state machines for flight modes

### 2.2 `offboard_control` (C++)
- Low-level control setpoints
- Two implementations:
  - `px4_offboard.cpp` - Attitude/rates/trajectory setpoints
  - `ardupilot_guided.cpp` - Velocity/acceleration references
- Processes YOLO detections

### 2.3 `mission` (Python)
- Mission orchestrator
- Loads YAML mission files
- Action client for autopilot_interface
- Example mission: takeoff → wait → orbit → reposition → land

### 2.4 `state_sharing` (C++)
- Publishes drone state to Zenoh
- Used for swarm coordination
- Topic: `/state_sharing_drone_N`

### 2.5 `yolo_py` (Python)
- YOLOv8 object detection via ONNX
- Architecture-aware (CUDA for x86, TensorRT for Jetson)
- Publishes to `/detections`

**Observation**: The aircraft stack is designed to be autopilot-agnostic with separate implementations for PX4 and ArduPilot.

---

## 3. GROUND (`ground/`)

**Purpose**: Ground control station for monitoring and coordination.

**ROS2 Packages**:

### 3.1 `ground_system` (C++)
- Tracks swarm of drones
- Subscribes to state_sharing messages
- Publishes aggregated `/tracks` topic

**Custom Messages**:
- `DroneObs` - Single drone observation (id, position, velocity)
- `SwarmObs` - Array of DroneObs

**Observation**: The ground system is relatively simple - mainly aggregates drone states.

---

## 4. AAS-GYM (`aas-gym/`)

**Purpose**: Gymnasium (OpenAI Gym) wrapper for reinforcement learning.

**Key File**: `src/aas_gym/aas_env.py`

**Current Implementation**:
```python
class AASEnv(gym.Env):
    # Action space: continuous [-1.0, 1.0] (dummy actions currently)
    # Observation space: Gazebo sim clock [seconds, nanoseconds]

    # Key parameters:
    # - autopilot: "px4" or "ardupilot"
    # - camera: bool
    # - lidar: bool
    # - num_quads: int

    # Timing:
    # - init_duration: 80s (startup time)
    # - max_episode_length: 300s
    # - step_length: 0.05s (default)
    # - rtf: 15x real-time factor
```

**Observation**: The gym environment currently has **dummy** action/observation spaces. It manages Docker containers for simulation but doesn't have meaningful control integration yet.

---

## 5. SCRIPTS (`scripts/`)

**Build Scripts**:
- `sim_build.sh` - Build simulation Docker images (~30GB, ~25min)
- `deploy_build.sh` - Build Jetson deployment image

**Run Scripts**:
- `sim_run.sh` - Start SITL simulation
  - Key env vars: `AUTOPILOT`, `NUM_QUADS`, `NUM_VTOLS`, `WORLD`, `RTF`
- `deploy_run.sh` - Deploy to real hardware

**Gym Script**:
- `gym_run.py` - Test gymnasium environment
  - Modes: step, speedup, vectorenv-speedup, learn

---

## Key Questions / Areas to Investigate

1. **How does the gym environment interface with the simulation?**
   - Currently uses ZMQ for clock synchronization
   - Actions appear to be dummy (not connected to actual control)

2. **What is the control flow for commands?**
   - Gym → ??? → autopilot_interface → MAVROS → PX4/ArduPilot

3. **What topics/services are available for control?**
   - Need to map out ROS2 topics/services/actions

4. **What is the state representation for RL?**
   - Currently just sim clock - not useful for learning

5. **How does stepping work?**
   - `gz_step.py` suggests Gazebo can be stepped manually
   - Need to understand the stepping mechanism

---

## Communication Architecture

```
┌─────────────────┐     ZMQ      ┌─────────────────┐
│    AAS-GYM      │◄────────────►│   SIMULATION    │
│  (Python/Host)  │   (clock)    │   (Docker)      │
└─────────────────┘              │                 │
                                 │  ┌───────────┐  │
                                 │  │  Gazebo   │  │
                                 │  │  + SITL   │  │
                                 │  └─────┬─────┘  │
                                 │        │MAVLink │
                                 │  ┌─────▼─────┐  │
                                 │  │  MAVROS   │  │
                                 │  └─────┬─────┘  │
                                 │        │ROS2    │
                                 │  ┌─────▼─────┐  │
                                 │  │ AIRCRAFT  │  │
                                 │  │   STACK   │  │
                                 │  └───────────┘  │
                                 └─────────────────┘
```

---

## Next Steps for Discussion

1. What is the specific goal? (RL training, scripted testing, custom control?)
2. What observations do we need from the simulation?
3. What actions do we want to send?
4. Do we need single-drone or multi-drone control?
5. What is the reward function (if RL)?

---

*Last updated: Research phase - no code changes made*
