import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import gymnasium as gym
from gymnasium import spaces
import pickle
import math
import time
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from datetime import datetime

# Constants
DT = 0.02
KV = 60.0
MAX_VOLTAGE = 12.0
GEAR_DRIVE = 6.12
WHEEL_RADIUS = 0.0508
HALF_WHEELBASE = 0.618 / 2
HALF_TRACKWIDTH = 0.618 / 2
MAX_EPISODE_STEPS = 5000
GOAL_TOLERANCE_POS = 0.2
GOAL_TOLERANCE_THETA = 0.2
GOAL_TOLERANCE_SPEED = 0.1
ACTION_BOUND_V = 4.0
ACTION_BOUND_OMEGA = np.pi

MODULE_POSITIONS = [
    (HALF_WHEELBASE, HALF_TRACKWIDTH),
    (HALF_WHEELBASE, -HALF_TRACKWIDTH),
    (-HALF_WHEELBASE, HALF_TRACKWIDTH),
    (-HALF_WHEELBASE, -HALF_TRACKWIDTH)
]

# Create output directory for logs and visualizations
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = f"training_results_{timestamp}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load trained model and necessary data
model = tf.keras.models.load_model('trained_model.keras')
with open('training_data.pkl', 'rb') as f:
    data = pickle.load(f)
    target_scaler = data['target_scaler']
    X_train = data['X_train']
    unknown_means = np.mean(X_train[:, 12:20], axis=0)

class SwerveDriveEnv(gym.Env):
    def __init__(self, render_mode=None):
        super(SwerveDriveEnv, self).__init__()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(14,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=np.array([-ACTION_BOUND_V, -ACTION_BOUND_V, -ACTION_BOUND_OMEGA]),
            high=np.array([ACTION_BOUND_V, ACTION_BOUND_V, ACTION_BOUND_OMEGA]),
            dtype=np.float32
        )
        self.render_mode = render_mode
        self.model = model
        self.target_scaler = target_scaler
        self.unknown_means = unknown_means
        self.dt = DT
        self.kv = KV
        self.max_voltage = MAX_VOLTAGE
        self.gear_drive = GEAR_DRIVE
        self.r = WHEEL_RADIUS
        self.module_positions = MODULE_POSITIONS
        self.fig = None
        self.ax = None
        self.path_line = None
        self.robot_arrow = None
        self.path = []
        self.episode_rewards = []
        self.episode_distances = []
        self.reset()

    def reset(self, seed=None, options=None):
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.vx = 0.0
        self.vy = 0.0
        self.omega = 0.0
        self.motor_vels = np.zeros(4)
        self.angles = np.full(4, np.pi)
        self.current_step = 0
        self.path = [(self.x, self.y)]
        self.current_episode_reward = 0
        obs = self._get_obs()
        info = {}
        if self.render_mode == 'human':
            self._render_frame()
        return obs, info

    def is_inside_boundary(self):
        """Check if the robot is inside the parallelogram boundary"""
        return (self.y > -0.5 and self.y < 2 and 
                self.x > -1 and self.x < 1)

    def step(self, action):
        desired_vx, desired_vy, desired_omega = action
        desired_motor_vels = np.zeros(4)
        desired_angles = np.zeros(4)
        
        # Calculate desired module states
        for i, (mod_x, mod_y) in enumerate(self.module_positions):
            temp_x = desired_vx - desired_omega * mod_y
            temp_y = desired_vy + desired_omega * mod_x
            speed = math.sqrt(temp_x**2 + temp_y**2)
            angle = math.atan2(temp_y, temp_x)
            delta = angle - self.angles[i]
            if abs(delta) > np.pi / 2:
                angle += np.pi if delta < 0 else -np.pi
                speed = -speed
            desired_angles[i] = (angle + np.pi) % (2 * np.pi) - np.pi
            desired_wheel_rad = speed / self.r
            desired_motor_vels[i] = desired_wheel_rad * self.gear_drive
        
        # Calculate voltages
        volts = desired_motor_vels / self.kv
        volts = np.clip(volts, -self.max_voltage, self.max_voltage)
        
        # Prepare input for FNN model
        input_arr = np.zeros(24)
        input_arr[0:4] = self.motor_vels
        input_arr[4:8] = volts
        input_arr[8:12] = self.angles
        input_arr[12:20] = self.unknown_means
        input_arr[20] = self.omega
        input_arr[21] = self.vx
        input_arr[22] = self.vy
        input_arr[23] = self.dt
        
        # Get prediction from FNN model
        pred_scaled = self.model.predict(input_arr.reshape(1, -1), verbose=0)
        pred = self.target_scaler.inverse_transform(pred_scaled)[0]
        
        # Check for NaN or infinity in predictions
        if np.any(np.isnan(pred)) or np.any(np.isinf(pred)):
            print(f"Warning: Invalid prediction from model: {pred}")
            pred = np.zeros(3)  # Reset to zero if invalid
        
        achieved_omega, achieved_vx, achieved_vy = pred
        
        # Store old distance for reward calculation
        old_dist = math.sqrt((self.x - 1)**2 + (self.y - 2)**2)
        
        # Update robot state
        delta_x = (achieved_vx * math.cos(self.theta) - achieved_vy * math.sin(self.theta)) * self.dt
        delta_y = (achieved_vx * math.sin(self.theta) + achieved_vy * math.cos(self.theta)) * self.dt
        delta_theta = achieved_omega * self.dt
        
        self.x += delta_x
        self.y += delta_y
        self.theta += delta_theta
        self.theta = (self.theta + np.pi) % (2 * np.pi) - np.pi
        self.vx = achieved_vx
        self.vy = achieved_vy
        self.omega = achieved_omega
        self.motor_vels = desired_motor_vels
        self.angles = desired_angles
        self.current_step += 1
        self.path.append((self.x, self.y))
        
        # Calculate new distance to goal
        new_dist = math.sqrt((self.x - 1)**2 + (self.y - 2)**2)
        
        # Calculate rewards - Modified to encourage goal-reaching
        progress_reward = (old_dist - new_dist) * 200  # Increased to encourage movement
        dist_penalty = -new_dist * 0.2  # Reduced penalty
        theta_penalty = -abs(self.theta) * 0.2  # Reduced penalty
        time_penalty = -0.5  # Reduced penalty
        energy_penalty = -np.mean(np.abs(volts)) * 0.02  # Reduced penalty
        
        # Add reward for moving toward goal
        goal_direction = math.atan2(2 - self.y, 1 - self.x)
        movement_direction = math.atan2(self.vy, self.vx)
        direction_reward = math.cos(goal_direction - movement_direction) * 0.5 if (abs(self.vx) + abs(self.vy)) > 0.1 else 0
        
        reward = progress_reward + dist_penalty + theta_penalty + time_penalty + energy_penalty + direction_reward
        self.current_episode_reward += reward
        
        # Check termination conditions
        terminated = False
        truncated = self.current_step >= MAX_EPISODE_STEPS
        
        # Check if outside boundary
        if not self.is_inside_boundary():
            terminated = True
            reward -= 200  # Reduced penalty
            print("Episode terminated: Out of bounds")
        
        # Check if reached goal position
        if new_dist < GOAL_TOLERANCE_POS:
            terminated = True
            goal_reward = 1000  # Increased reward
            
            # Reduced penalties for velocity and orientation errors
            theta_err = abs(self.theta)
            speed_norm = abs(self.vx) + abs(self.vy) + abs(self.omega)
            
            # Calculate penalties
            theta_penalty = max(0, theta_err - GOAL_TOLERANCE_THETA) * 20  # Reduced
            speed_penalty = max(0, speed_norm - GOAL_TOLERANCE_SPEED) * 2  # Reduced
            
            # Apply net reward
            reward += goal_reward - theta_penalty - speed_penalty
            print(f"Goal reached! Reward: {goal_reward}, Theta penalty: {theta_penalty}, Speed penalty: {speed_penalty}")
        
        # Store episode data for logging
        self.episode_distances.append(new_dist)
        
        obs = self._get_obs()
        info = {}
        
        if self.render_mode == 'human':
            self._render_frame()
            
        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        return np.concatenate(([self.x, self.y, self.theta, self.vx, self.vy, self.omega], self.motor_vels, self.angles))

    def render(self):
        if self.render_mode == 'human':
            self._render_frame()

    def _render_frame(self):
        if self.fig is None:
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(10, 10))
            
            # Set plot limits to accommodate the parallelogram
            self.ax.set_xlim(-2, 3)
            self.ax.set_ylim(-2, 4)
            self.ax.set_aspect('equal')
            
            # Draw parallelogram boundary
            x_vals = np.linspace(-2, 3, 100)
            self.ax.axvline(x=1,color='r',linewidth=2)
            self.ax.axvline(x=-1,color='r',linewidth=2)
            self.ax.axhline(y=-0.5, color='r', linewidth=2)
            self.ax.axhline(y=3, color='r', linewidth=2)
            
            # Draw goal
            self.ax.plot(1, 2, 'go', markersize=10, markeredgecolor='black')
            
            # Draw goal tolerance circle
            goal_circle = plt.Circle((1, 2), GOAL_TOLERANCE_POS, color='g', fill=False, linestyle='--')
            self.ax.add_artist(goal_circle)
            
            # Initialize path and robot arrow
            self.path_line, = self.ax.plot([], [], 'b-', linewidth=2)
            self.robot_arrow = self.ax.arrow(0, 0, 0.1 * math.cos(0), 0.1 * math.sin(0), 
                                           head_width=0.05, color='k', length_includes_head=True)
            
            # Add text for current step and reward
            self.step_text = self.ax.text(0.02, 0.98, '', transform=self.ax.transAxes, verticalalignment='top')
            self.reward_text = self.ax.text(0.02, 0.94, '', transform=self.ax.transAxes, verticalalignment='top')
        
        # Update path and robot position
        path_x, path_y = zip(*self.path)
        self.path_line.set_data(path_x, path_y)
        
        # Update robot arrow
        self.robot_arrow.remove()
        self.robot_arrow = self.ax.arrow(
            self.x, self.y, 
            0.3 * math.cos(self.theta), 0.3 * math.sin(self.theta), 
            head_width=0.08, color='k', length_includes_head=True
        )
        
        # Update info text
        self.step_text.set_text(f'Step: {self.current_step}')
        self.reward_text.set_text(f'Reward: {self.current_episode_reward:.2f}')
        
        # Limit rendering frequency to avoid lag
        if self.current_step % 5 == 0:
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            plt.pause(0.001)

    def close(self):
        if self.fig:
            plt.close(self.fig)
        
        # Save episode data when environment closes
        if hasattr(self, 'episode_distances') and self.episode_distances:
            episode_data = {
                'rewards': self.episode_rewards,
                'distances': self.episode_distances
            }
            with open(os.path.join(OUTPUT_DIR, 'episode_data.pkl'), 'wb') as f:
                pickle.dump(episode_data, f)

class TrainingCallback(BaseCallback):
    def __init__(self, verbose=0, check_freq=1000):
        super(TrainingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_successes = []
        self.episode_min_distances = []
        self.best_mean_reward = -np.inf
        self.episode_count = 0
        
        # Create CSV log file
        self.log_file = open(os.path.join(OUTPUT_DIR, 'training_log.csv'), 'w')
        self.log_file.write('episode,reward,length,success,min_distance\n')

    def _on_step(self):
        return True
    
    def _on_rollout_end(self):
        if 'infos' in self.locals and self.locals['infos']:
            for info in self.locals['infos']:
                if 'episode' in info:
                    mean_reward = info['episode']['r']
                    episode_length = info['episode']['l']
                    self.episode_rewards.append(mean_reward)
                    self.episode_lengths.append(episode_length)
                    
                    # Check if episode was successful (reached goal)
                    success = 1 if mean_reward > 500 else 0  # Assuming successful episodes have high reward
                    self.episode_successes.append(success)
                    
                    # Log to CSV
                    min_distance = np.min(self.training_env.envs[0].episode_distances) if hasattr(self.training_env.envs[0], 'episode_distances') else 0
                    self.episode_min_distances.append(min_distance)
                    self.log_file.write(f'{self.episode_count},{mean_reward},{episode_length},{success},{min_distance}\n')
                    self.log_file.flush()
                    
                    print(f"Episode {self.episode_count} - Reward: {mean_reward:.2f}, Length: {episode_length}, Min Distance: {min_distance:.2f}")
                    
                    # Update best model if improvement
                    if len(self.episode_rewards) > 10:
                        mean_reward_last_10 = np.mean(self.episode_rewards[-10:])
                        if mean_reward_last_10 > self.best_mean_reward:
                            self.best_mean_reward = mean_reward_last_10
                            self.model.save(os.path.join(OUTPUT_DIR, 'best_model'))
                            print(f"New best model saved with mean reward: {self.best_mean_reward:.2f}")
                    
                    self.episode_count += 1
                    
                    # Plot training progress every 10 episodes
                    if self.episode_count % 10 == 0:
                        self.plot_training_progress()
        
        return True
    
    def plot_training_progress(self):
        plt.figure(figsize=(12, 8))
        
        # Plot rewards
        plt.subplot(2, 2, 1)
        plt.plot(self.episode_rewards)
        plt.title('Episode Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        
        # Plot episode lengths
        plt.subplot(2, 2, 2)
        plt.plot(self.episode_lengths)
        plt.title('Episode Lengths')
        plt.xlabel('Episode')
        plt.ylabel('Length')
        
        # Plot success rate (moving average)
        plt.subplot(2, 2, 3)
        success_rate = np.convolve(self.episode_successes, np.ones(10)/10, mode='valid')
        plt.plot(range(10, len(self.episode_successes)+1), success_rate)
        plt.title('Success Rate (10-episode moving average)')
        plt.xlabel('Episode')
        plt.ylabel('Success Rate')
        plt.ylim(0, 1)
        
        # Plot minimum distances
        plt.subplot(2, 2, 4)
        plt.plot(self.episode_min_distances)
        plt.title('Minimum Distance to Goal')
        plt.xlabel('Episode')
        plt.ylabel('Distance')
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'training_progress.png'))
        plt.close()
    
    def _on_training_end(self):
        self.log_file.close()

if __name__ == "__main__":
    # Create environment with rendering
    env = SwerveDriveEnv(render_mode='human')
    
    # Train with PPO - Tuned parameters
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        n_steps=1024,  # Increased for more stable updates
        batch_size=64,
        n_epochs=10,
        learning_rate=0.0001,  # Reduced learning rate
        gamma=0.99,  # Increased discount factor
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log=OUTPUT_DIR
    )
    
    # Add callback for per-episode logging
    callback = TrainingCallback()
    
    # Train
    print("Starting training...")
    model.learn(total_timesteps=200000, callback=callback, tb_log_name="PPO")
    
    # Save trained policy
    model.save(os.path.join(OUTPUT_DIR, "final_model"))
    
    env.close()
    
    # Generate final training report
    print("Generating training report...")
    callback.plot_training_progress()