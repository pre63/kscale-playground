""" Terminal script to run the ZBot environment

TODO
1. Test the behavior
2. Make it work without dependency but on submodule
3. Chunk it to save and load
4. Put it on colab to
5.

"""
from datetime import datetime
import functools

from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
import cv2
import jax
from jax import numpy as jp
import matplotlib.pyplot as plt
import mediapy as media
import mujoco
import numpy as np

from mujoco_playground import registry
from mujoco_playground import wrapper
from mujoco_playground._src.gait import draw_joystick_command
from mujoco_playground.config import locomotion_params

from zbot import randomize as zbot_randomize
from zbot import joystick as zbot_joystick
from ml_collections import config_dict

# More legible printing from numpy.
np.set_printoptions(precision=3, suppress=True, linewidth=100)


def save_video_opencv(frames, fps, filename='output.mp4'):
    # Get dimensions from the first frame
    height, width, _ = frames[0].shape
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    
    # Write frames
    for frame in frames:
        # OpenCV expects BGR format
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    
    out.release()


def progress(num_steps, metrics):
    plt.figure()
    times.append(datetime.now())
    x_data.append(num_steps)
    y_data.append(metrics["eval/episode_reward"])
    y_dataerr.append(metrics["eval/episode_reward_std"])
    plt.xlim([0, ppo_params["num_timesteps"] * 1.25])
    plt.xlabel("# environment steps")
    plt.ylabel("reward per episode")
    plt.title(f"y={y_data[-1]:.3f}")
    plt.errorbar(x_data, y_data, yerr=y_dataerr, color="blue")
    plt.savefig('plot.png')  # saves to file instead of displaying
    plt.close()


env_config = zbot_joystick.default_config()


rl_config = config_dict.create(
    num_timesteps=100_000_000,
    num_evals=10,
    reward_scaling=1.0,
    episode_length=env_config.episode_length,
    normalize_observations=True,
    action_repeat=1,
    unroll_length=20,
    num_minibatches=32,
    num_updates_per_batch=4,
    discounting=0.97,
    learning_rate=3e-4,
    entropy_cost=1e-2,
    num_envs=8192,
    batch_size=256,
    max_grad_norm=1.0,
    network_factory=config_dict.create(
        policy_hidden_layer_sizes=(128, 128, 128, 128),
        value_hidden_layer_sizes=(256, 256, 256, 256, 256),
        policy_obs_key="state",
        value_obs_key="state",
    ),
)
rl_config.num_timesteps = 150_000_000
rl_config.num_evals = 15
rl_config.clipping_epsilon = 0.2
rl_config.num_resets_per_eval = 1
rl_config.entropy_cost = 0.005
rl_config.network_factory = config_dict.create(
    policy_hidden_layer_sizes=(512, 256, 128),
    value_hidden_layer_sizes=(512, 256, 128),
    policy_obs_key="state",
    value_obs_key="privileged_state",
)
ppo_params = rl_config

env_name = "ZbotJoystickFlatTerrain"
env = zbot_joystick.Joystick(task="flat_terrain")
env_cfg = zbot_joystick.default_config()
env_cfg.episode_length = 1
randomizer = zbot_randomize.domain_randomize

# pfb30
ppo_params.num_envs=1
ppo_params.batch_size=4
ppo_params.num_minibatches=1
ppo_params.num_timesteps=5
ppo_params.episode_length=1
ppo_params.num_updates_per_batch=1
assert ppo_params.batch_size * ppo_params.num_minibatches % ppo_params.num_envs == 0

print("PPO paramters")
print(ppo_params)

print("Config paramters")
print(env_cfg)

x_data, y_data, y_dataerr = [], [], []
times = [datetime.now()]


ppo_training_params = dict(ppo_params)
network_factory = ppo_networks.make_ppo_networks
if "network_factory" in ppo_params:
    del ppo_training_params["network_factory"]
    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        **ppo_params.network_factory
    )

train_fn = functools.partial(
    ppo.train, **dict(ppo_training_params),
    network_factory=network_factory,
    randomization_fn=randomizer,
    progress_fn=progress
)

make_inference_fn, params, metrics = train_fn(
    environment=env,
    eval_env=zbot_joystick.Joystick(task="flat_terrain"),
    wrap_env_fn=wrapper.wrap_for_brax_training,
)
print(f"time to jit: {times[1] - times[0]}")
print(f"time to train: {times[-1] - times[1]}")

# Save parameters using pickle
import pickle
from pathlib import Path

# Create models directory if it doesn't exist
save_dir = Path("checkpoints")
save_dir.mkdir(exist_ok=True)

if False:
    # Save params to file
    model_path = save_dir / f"{env_name}_params.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(params, f)

    print(f"Model saved to: {model_path}")

with open(f"checkpoints/{env_name}_params.pkl", "rb") as f:
    loaded_params = pickle.load(f)

params = loaded_params
print("Loaded params")


# EVAL ROLLOUT

eval_env = zbot_joystick.Joystick(task="flat_terrain")

jit_reset = jax.jit(eval_env.reset)
jit_step = jax.jit(eval_env.step)
jit_inference_fn = jax.jit(make_inference_fn(params, deterministic=True))

rng = jax.random.PRNGKey(1)

rollout = []
modify_scene_fns = []

x_vel = 1.0  #@param {type: "number"}
y_vel = 0.0  #@param {type: "number"}
yaw_vel = 0.0  #@param {type: "number"}
command = jp.array([x_vel, y_vel, yaw_vel])

phase_dt = 2 * jp.pi * eval_env.dt * 1.5
phase = jp.array([0, jp.pi])

# pfb30
env_cfg.episode_length = 3000

for j in range(2):
    print(f"episode {j}")
    state = jit_reset(rng)
    state.info["phase_dt"] = phase_dt
    state.info["phase"] = phase
    for i in range(env_cfg.episode_length):
        act_rng, rng = jax.random.split(rng)
        ctrl, _ = jit_inference_fn(state.obs, act_rng)
        state = jit_step(state, ctrl)
        if state.done:
            break
        state.info["command"] = command
        rollout.append(state)

        if env_name == "BerkeleyHumanoidJoystickFlatTerrain":
            xyz = np.array(state.data.xpos[eval_env.mj_model.body("torso").id])
        elif env_name == "ZbotJoystickFlatTerrain":
            xyz = np.array(state.data.xpos[eval_env.mj_model.body("Z-BOT2_MASTER-BODY-SKELETON").id])
        else:
            raise ValueError(f"Unknown environment: {env_name}")
        xyz += np.array([0, 0.0, 0])
        x_axis = state.data.xmat[eval_env._torso_body_id, 0]
        yaw = -np.arctan2(x_axis[1], x_axis[0])
        modify_scene_fns.append(
            functools.partial(
                draw_joystick_command,
                cmd=state.info["command"],
                xyz=xyz,
                theta=yaw,
                scl=np.linalg.norm(state.info["command"]),
            )
        )

    render_every = 1
    fps = 1.0 / eval_env.dt / render_every
    print(f"fps: {fps}")
    traj = rollout[::render_every]
    mod_fns = modify_scene_fns[::render_every]

    scene_option = mujoco.MjvOption()
    scene_option.geomgroup[2] = True
    scene_option.geomgroup[3] = False
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = False

    frames = eval_env.render(
        traj,
        camera="track",
        scene_option=scene_option,
        width=640*2,
        height=480,
        modify_scene_fns=mod_fns,
    )

    # media.show_video(frames, fps=fps, loop=False)
    # media.show_video(frames, fps=fps, loop=False) # Save the video
    save_video_opencv(frames, fps=fps, filename=f'output_{j}.mp4')
