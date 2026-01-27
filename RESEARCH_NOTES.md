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

## CRITICAL FINDING: The Gap

After reading the actual code, here's what we discovered:

### What Currently Exists

**`aas_env.py` (Python gym environment):**
```python
# Sends action via ZMQ
action_payload = struct.pack('d', force)  # force is a single float [-1, 1]
self.socket.send(action_payload)
reply_bytes = self.socket.recv()  # Gets clock back
```

**`zeromq_bridge.cpp` (C++ ROS2 node in simulation container):**
```cpp
// Receives action from gym
double action = *static_cast<double*>(request.data());

// If action == 9999.0 → RESET mode (unpause for 80s, then pause)
// Else → STEP mode:
//   1. Publish action to /action topic
publisher_->publish(ros_msg);  // publishes Float64 to "/action"
//   2. Step Gazebo
step_gazebo();
//   3. Return clock
socket_.send(reply);
```

### The Problem: Nobody Listens to `/action`

The ZMQ bridge **publishes** the action to ROS2 topic `/action`, but:

**NOTHING IN THE AIRCRAFT STACK SUBSCRIBES TO `/action`**

The aircraft stack only has:
- ROS2 **Actions** (not topics): `/DroneN/takeoff_action`, `/DroneN/land_action`, `/DroneN/orbit_action`, `/DroneN/offboard_action`
- ROS2 **Services**: `/DroneN/set_speed`, `/DroneN/set_reposition`

These are completely different from a simple topic subscription.

### Current Data Flow (Broken)

```
┌──────────────┐                    ┌─────────────────────────────────────────┐
│   aas_env    │                    │         SIMULATION CONTAINER            │
│   (Python)   │                    │                                         │
│              │   ZMQ TCP:5555     │  ┌─────────────────┐                    │
│  step(act)  ─┼───────────────────►│  │  zeromq_bridge  │                    │
│              │                    │  │                 │                    │
│              │   [sec, nanosec]   │  │  publishes to   │    ┌────────────┐  │
│  ◄───────────┼────────────────────┼──┤  /action topic ─┼───►│  NOWHERE   │  │
│              │                    │  │                 │    │  (no sub)  │  │
└──────────────┘                    │  │  steps Gazebo   │    └────────────┘  │
                                    │  └─────────────────┘                    │
                                    │                                         │
                                    │  ┌─────────────────┐                    │
                                    │  │  AIRCRAFT STACK │ (separate container)
                                    │  │                 │                    │
                                    │  │  - autopilot_   │                    │
                                    │  │    interface    │ ← Only listens to  │
                                    │  │  - mission      │   ROS2 Actions,    │
                                    │  │  - offboard_    │   not /action topic│
                                    │  │    control      │                    │
                                    │  └─────────────────┘                    │
                                    └─────────────────────────────────────────┘
```

### What the Gym Environment Actually Does Now

1. **On reset()**:
   - Restarts Docker containers
   - Sends action=9999.0 to ZMQ bridge
   - Bridge unpauses Gazebo for 80 seconds (init period)
   - Bridge pauses Gazebo, returns clock
   - During those 80 seconds, the mission node runs and does whatever the mission YAML says

2. **On step(action)**:
   - Sends action (a single float) to ZMQ bridge
   - Bridge publishes to `/action` (nobody listening)
   - Bridge steps Gazebo by N physics steps
   - Bridge returns the new clock time
   - **The action has NO EFFECT on the drone**

---

## Available Control Interfaces in Aircraft Stack

### High-Level Control (autopilot_interface)

**ROS2 Actions** (long-running, cancellable operations):
| Action | Purpose | Parameters |
|--------|---------|------------|
| `/DroneN/takeoff_action` | Take off and hover | `takeoff_altitude`, `vtol_transition_heading`, etc. |
| `/DroneN/land_action` | Land the drone | `landing_altitude`, `vtol_transition_heading` |
| `/DroneN/orbit_action` | Fly in a circle | `east`, `north`, `altitude`, `radius` |
| `/DroneN/offboard_action` | Enter offboard/guided mode | `offboard_setpoint_type`, `max_duration_sec` |

**ROS2 Services** (instant commands):
| Service | Purpose | Parameters |
|---------|---------|------------|
| `/DroneN/set_speed` | Set flight speed | `speed` (float) |
| `/DroneN/set_reposition` | Go to position | `east`, `north`, `altitude` |

### Low-Level Control (offboard_control)

When in offboard mode, the drone listens to setpoints:

**PX4**:
- Attitude setpoints
- Rate setpoints
- Trajectory setpoints

**ArduPilot**:
- Velocity commands via MAVROS `/mavros/setpoint_velocity/cmd_vel`

---

## Goal: Drone Takeoff, Hover, User Commands

Based on your stated goal, here's what we need:

### Option A: Use Existing High-Level Interface
- Call `/DroneN/takeoff_action` to take off
- Use `/DroneN/set_reposition` to move
- Simple but coarse control

### Option B: Use Offboard/Guided Mode for Velocity Control
- Call `/DroneN/takeoff_action` to take off
- Call `/DroneN/offboard_action` to enter offboard mode
- Send velocity commands directly
- More fine-grained control

### What Needs to Be Built

1. **A new ROS2 node** (or modify zeromq_bridge) that:
   - Subscribes to `/action` topic
   - Translates gym actions to actual drone commands
   - OR: Directly call ROS2 actions/services from the bridge

2. **Modify aas_env.py** to:
   - Define meaningful action space (e.g., velocity commands)
   - Define meaningful observation space (e.g., position, velocity)
   - Handle takeoff as part of reset()

3. **Decide on action/observation format**:
   - What does action `[-1, 1]` mean? (velocity? direction?)
   - What observations do we return? (position? orientation?)

---

## Questions Before We Code

1. **Control granularity**:
   - High-level (go to waypoint) or low-level (velocity commands)?

2. **Observation needs**:
   - Just position/velocity, or also camera/lidar?

3. **Takeoff handling**:
   - Should reset() automatically takeoff and hover?
   - Or should takeoff be an explicit action?

4. **Autopilot**:
   - PX4 or ArduPilot? (Different control interfaces)

---

*Last updated: After code analysis - identified the gap*
