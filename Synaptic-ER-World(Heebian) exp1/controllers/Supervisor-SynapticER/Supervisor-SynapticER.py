from controller import Supervisor, Keyboard
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns
import pandas as pd
import sys
import warnings

warnings.filterwarnings('ignore') # ignore warning

# Synaptic Gene Evaluation
# Copyright (c) 2024 Hamze Hammami (ORCID: 0009-0004-5754-5842)

# The full controller and supervisor scripts apply 'Hebbian learning' to pre-trained genetic algorithm genes
# to study the effects of synaptic and stimuli reactions in a robotic system.

# License:
# 1. Experimental: Results here are experimental and unexpected behaviors
# are prone to occur
# 2. Academic Use Only: This code is for evaluation and academic purposes only.
# Redistribution, modification, or commercial use is prohibited.
# 3. No Modifications: Modifications or derivative works are not allowed.
# 4. Attribution: Proper credit must be given if this code is referenced in
# academic or research contexts.
# 5. Liability: This code is provided "as is," without guarantees or warranties.
# The author is not responsible for any issues arising from its use.


class SupervisorEvaluation:
    # setup parameters
    def __init__(self):
        self.time_step = 32  
        self.time_experiment = 250  
        # init supervisor and components
        self.supervisor = Supervisor()
        self.robot_node = self.supervisor.getFromDef("Controller")
        if self.robot_node is None:
            sys.stderr.write("No DEF Controller node found in the current world file\n")
            sys.exit(1)
            
        self.trans_field = self.robot_node.getField("translation")  
        self.rot_field = self.robot_node.getField("rotation")
        
        self.emitter = self.supervisor.getDevice("emitter")
        self.receiver = self.supervisor.getDevice("receiver")
        self.receiver.enable(self.time_step)
        
        self.receivedData = "" 
        self.receivedWeights = "" 
        self.receivedFitness = 0.0
        self.emitterData = ""
        self.receivedLightSensors = [0] * 8
        self.receivedProxSensors = [0] * 8
        
        self.spot_light_node = self.supervisor.getFromDef("spot_light")
        if self.spot_light_node is None:
            sys.stderr.write("Spot light node not found\n")
            sys.exit(1)
        self.spot_light_status = self.spot_light_node.getField("on")
        # data collection     
        self.plasticity_enabled = False
        self.plasticity_history = []
        self.weight_changes = []
        
    def handle_receiver(self):
        # Process received messages
        while(self.receiver.getQueueLength() > 0):
            try:
                self.receivedData = self.receiver.getString()
                
                if self.receivedData.startswith("weight_change"):
                    change = float(self.receivedData.split()[1])  
                    self.weight_changes.append(change)
                elif self.receivedData.startswith("fitness"):
                    self.receivedFitness = float(self.receivedData.split()[1])  # Changed parsing
                elif self.receivedData.startswith("sensors:"):
                    parts = self.receivedData[8:].strip().split(":")
                    if len(parts) == 2:
                        self.receivedLightSensors = [float(x) for x in parts[0].split(",")]
                        self.receivedProxSensors = [float(x) for x in parts[1].split(",")]
            except Exception as e:
                print(f"Debug - Message: '{self.receivedData}'")  
            self.receiver.nextPacket()

    
    def handle_emitter(self):
        #Send messages to robot
        if self.emitterData:
            self.emitter.send(str(self.emitterData).encode("utf-8"))
            self.emitterData = ""  
            
        plasticity_msg = f"hebbian:{self.plasticity_enabled}".encode("utf-8")
        self.emitter.send(plasticity_msg)
    
            
    def run_trial(self, trial_type, save_dir, genotype, force_genotype=False):
        # run trial
        print(f"\nRunning {trial_type} trial...")
        print(f"Mode: {'Hebbian Learning' if self.plasticity_enabled else 'Normal Operation'}")
        print(f"Trial duration: {self.time_experiment} seconds")
        
        self.emitterData = str(genotype)
        self.weight_changes = []
        
        INITIAL_TRANS = [0.007, 0, 0.35]
        INITIAL_ROT = [-0.5, 0.5, 0.5, 2.09]
        self.trans_field.setSFVec3f(INITIAL_TRANS)
        self.rot_field.setSFRotation(INITIAL_ROT)
        self.robot_node.resetPhysics()
        
        self.spot_light_node.getField('on').setSFBool(trial_type == "right")
        
        trial_data = []
        start_time = self.supervisor.getTime()
        iterations = 0
        stop = int((self.time_experiment * 1000) / self.time_step)
        
        print(f"Starting trial loop ({stop} iterations)...")
        
        while iterations < stop and self.supervisor.step(self.time_step) != -1:
            current_time = self.supervisor.getTime() - start_time
            position = self.robot_node.getPosition()
            
            self.handle_emitter()
            self.handle_receiver()
            
            if force_genotype and iterations % 10 == 0:  # Send genotype every 10 iterations
                self.emitterData = str(genotype)
            
            current_weight_change = self.weight_changes[-1] if self.weight_changes else 0.0
            
            row_data = {
                'Time': current_time,
                'Position_X': position[0],
                'Position_Y': position[1],
                'Position_Z': position[2],
                'LS_Left': np.mean(self.receivedLightSensors[0:3]),
                'LS_Center': np.mean(self.receivedLightSensors[3:5]),
                'LS_Right': np.mean(self.receivedLightSensors[5:8]),
                'Fitness': self.receivedFitness,
                'Weight_Change': current_weight_change
            }
            
            for i in range(8):
                row_data[f'PS{i}'] = self.receivedProxSensors[i]
            
            trial_data.append(row_data)
            iterations += 1
            
            if iterations % (stop // 10) == 0:
                print(f"Progress: {iterations/stop*100:.1f}%")
                if self.plasticity_enabled and self.weight_changes:
                    print(f"Current weight change: {current_weight_change:.6f}")
                    print(f"Average weight change: {np.mean(self.weight_changes):.6f}")
        
        df = pd.DataFrame(trial_data)
        
        print(f"\nTrial Summary ({trial_type}):")
        print(f"Final position: ({position[0]:.3f}, {position[2]:.3f})")
        print(f"Average fitness: {df['Fitness'].mean():.4f}")
        if self.plasticity_enabled:
            print(f"Weight changes recorded: {len(self.weight_changes)}")
            print(f"Total weight change: {np.sum(self.weight_changes):.6f}")
            print(f"Mean weight change: {np.mean(self.weight_changes):.6f}")
        
        return df

    def run_demo(self, force_genotype=False):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode = "hebbian" if self.plasticity_enabled else "normal"
        base_dir = "robot_evaluation"
        os.makedirs(base_dir, exist_ok=True)
        save_dir = os.path.join(base_dir, f"evaluation_logs_{mode}_{timestamp}")
        os.makedirs(save_dir, exist_ok=True)
        try:
            genotype = np.load("Best.npy")
            left_data = self.run_trial("left", save_dir, genotype, force_genotype)
            right_data = self.run_trial("right", save_dir, genotype, force_genotype)
            self.plot_evaluation_results(save_dir, left_data, right_data)
            print("\nEvaluation complete!")
            print(f"Results saved in: {save_dir}")
        except Exception as e:
            print(f"Error during evaluation: {str(e)}")
            
    def plot_evaluation_results(self, save_dir, left_data, right_data):
        # Generate visualization plots
        plt.style.use('seaborn')
        sns.set_theme(style="whitegrid")
        plt.rcParams['figure.figsize'] = [12, 8]
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['font.size'] = 12
        
        print(f"generating traj plot")
        self._plot_trajectories(save_dir, left_data, right_data)
        self._plot_sensor_data(save_dir, left_data, right_data)
        if self.plasticity_enabled:
            print(f"plasticty enabled genearting placticty plot")
            self._plot_plasticity_analysis(save_dir, left_data, right_data)
            
        self._save_statistics(save_dir, left_data, right_data)

    def _plot_sensor_data(self, save_dir, left_data, right_data):
        # Plot sensor readings and heatmaps
        plt.figure(figsize=(15, 7))
        
        plt.subplot(1, 2, 1)
        plt.plot(left_data['Time'], left_data['LS_Left'], label='Left', alpha=0.7)
        plt.plot(left_data['Time'], left_data['LS_Center'], label='Center', alpha=0.7)
        plt.plot(left_data['Time'], left_data['LS_Right'], label='Right', alpha=0.7)
        plt.title('Left Turn - Light Sensors')
        plt.xlabel('Time (s)')
        plt.ylabel('Sensor Value')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(right_data['Time'], right_data['LS_Left'], label='Left', alpha=0.7)
        plt.plot(right_data['Time'], right_data['LS_Center'], label='Center', alpha=0.7)
        plt.plot(right_data['Time'], right_data['LS_Right'], label='Right', alpha=0.7)
        plt.title('Right Turn - Light Sensors')
        plt.xlabel('Time (s)')
        plt.ylabel('Sensor Value')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/light_sensors.png')
        plt.close()
        
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 1, 1)
        light_cols = ['LS_Left', 'LS_Center', 'LS_Right']
        sns.heatmap(left_data[light_cols].T, cmap='viridis', xticklabels=100, yticklabels=light_cols, cbar_kws={'label': 'Sensor Value'})
        plt.title('Left Turn - Light Sensor Heatmap')
        plt.xlabel('Time Steps')
        plt.ylabel('Light Sensors')
        
        plt.subplot(2, 1, 2)
        sns.heatmap(right_data[light_cols].T, cmap='viridis', xticklabels=100, yticklabels=light_cols, cbar_kws={'label': 'Sensor Value'})
        plt.title('Right Turn - Light Sensor Heatmap')
        plt.xlabel('Time Steps')
        plt.ylabel('Light Sensors')
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/light_sensor_heatmaps.png')
        plt.close()
        
        self._plot_proximity_sensors(save_dir, left_data, right_data)
        
    def _plot_proximity_sensors(self, save_dir, left_data, right_data):
        # Plot proximity sensor readings and heatmaps
        # Time series plot for proximity sensors
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 1, 1)
        for i in range(8):
            plt.plot(left_data['Time'], left_data[f'PS{i}'], 
                    label=f'PS{i}', alpha=0.7)
        plt.title('Left Turn - Proximity Sensors')
        plt.xlabel('Time (s)')
        plt.ylabel('Sensor Value')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        
        plt.subplot(2, 1, 2)
        for i in range(8):
            plt.plot(right_data['Time'], right_data[f'PS{i}'], 
                    label=f'PS{i}', alpha=0.7)
        plt.title('Right Turn - Proximity Sensors')
        plt.xlabel('Time (s)')
        plt.ylabel('Sensor Value')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/proximity_sensors.png', bbox_inches='tight')
        plt.close()
        
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 1, 1)
        prox_cols = [f'PS{i}' for i in range(8)]
        sns.heatmap(left_data[prox_cols].T, cmap='viridis', 
                    xticklabels=100, yticklabels=prox_cols,
                    cbar_kws={'label': 'Sensor Value'})
        plt.title('Left Turn - Proximity Sensor Heatmap')
        plt.xlabel('Time Steps')
        plt.ylabel('Proximity Sensors')
        
        plt.subplot(2, 1, 2)
        sns.heatmap(right_data[prox_cols].T, cmap='viridis',
                    xticklabels=100, yticklabels=prox_cols,
                    cbar_kws={'label': 'Sensor Value'})
        plt.title('Right Turn - Proximity Sensor Heatmap')
        plt.xlabel('Time Steps')
        plt.ylabel('Proximity Sensors')
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/proximity_sensor_heatmaps.png')
        plt.close()
        
    def _plot_trajectories(self, save_dir, left_data, right_data):
        # Plot robot trajectories
        plt.figure(figsize=(15, 7))
        
        plt.subplot(1, 2, 1)
        scatter = plt.scatter(left_data['Position_X'], 
                            left_data['Position_Z'],
                            c=left_data['Time'], 
                            cmap='viridis',
                            s=50, 
                            alpha=0.6)
        plt.colorbar(scatter, label='Time (s)')
        plt.scatter(left_data['Position_X'].iloc[0], 
                   left_data['Position_Z'].iloc[0],
                   color='green', s=200, label='Start', marker='*')
        plt.scatter(left_data['Position_X'].iloc[-1], 
                   left_data['Position_Z'].iloc[-1],
                   color='red', s=200, label='End', marker='*')
        plt.scatter(-0.36, -0.155, color='yellow', s=200, label='Target', marker='*')
        plt.title('Left Turn Trajectory')
        plt.xlabel('X Position (m)')
        plt.ylabel('Z Position (m)')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        
        plt.subplot(1, 2, 2)
        scatter = plt.scatter(right_data['Position_X'], 
                            right_data['Position_Z'],
                            c=right_data['Time'], 
                            cmap='viridis',
                            s=50, 
                            alpha=0.6)
        plt.colorbar(scatter, label='Time (s)')
        plt.scatter(right_data['Position_X'].iloc[0], 
                   right_data['Position_Z'].iloc[0],
                   color='green', s=200, label='Start', marker='*')
        plt.scatter(right_data['Position_X'].iloc[-1], 
                   right_data['Position_Z'].iloc[-1],
                   color='red', s=200, label='End', marker='*')
        plt.scatter(0.36, -0.155, color='yellow', s=200, label='Target', marker='*')
        plt.title('Right Turn Trajectory')
        plt.xlabel('X Position (m)')
        plt.ylabel('Z Position (m)')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/trajectories.png')
        plt.close()

    def _plot_plasticity_analysis(self, save_dir, left_data, right_data):
        # Plot plasticity analysis
        if 'Weight_Change' not in left_data.columns:
            print("No weight change data available for plasticity analysis")
            return
        
        # Plot 1: Overall weight changes over time
        plt.figure(figsize=(12, 10))
        
        plt.subplot(2, 1, 1)
        plt.plot(left_data['Time'], left_data['Weight_Change'], 
                 label='Left Turn', alpha=0.7, color='blue')
        plt.title('Overall Synaptic Weight Changes - Left Turn')
        plt.xlabel('Time (s)')
        plt.ylabel('Weight Change Magnitude')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 1, 2)
        plt.plot(right_data['Time'], right_data['Weight_Change'], 
                 label='Right Turn', alpha=0.7, color='red')
        plt.title('Overall Synaptic Weight Changes - Right Turn')
        plt.xlabel('Time (s)')
        plt.ylabel('Weight Change Magnitude')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/overall_weight_changes.png')
        plt.close()
        
        # Plot 2: Fitness and Light Sensors vs Weight Changes
        plt.figure(figsize=(12, 16))
        
        # Left turn - Fitness
        plt.subplot(3, 2, 1)
        plt.scatter(left_data['Weight_Change'], left_data['Fitness'], 
                    label='Left Turn', alpha=0.5, color='blue')
        plt.title('Fitness vs Weight Changes - Left Turn')
        plt.xlabel('Weight Change Magnitude')
        plt.ylabel('Fitness')
        plt.grid(True)
        plt.legend()
        
        # Right turn - Fitness
        plt.subplot(3, 2, 2)
        plt.scatter(right_data['Weight_Change'], right_data['Fitness'], 
                    label='Right Turn', alpha=0.5, color='red')
        plt.title('Fitness vs Weight Changes - Right Turn')
        plt.xlabel('Weight Change Magnitude')
        plt.ylabel('Fitness')
        plt.grid(True)
        plt.legend()
        
        # Left turn - Light Response
        plt.subplot(3, 2, 3)
        light_response_left = left_data[['LS_Left', 'LS_Center', 'LS_Right']].max(axis=1)
        plt.scatter(left_data['Weight_Change'], light_response_left, 
                    label='Left Turn', alpha=0.5, color='blue')
        plt.title('Light Response vs Weight Changes - Left Turn')
        plt.xlabel('Weight Change Magnitude')
        plt.ylabel('Light Sensor Response')
        plt.grid(True)
        plt.legend()
        
        # Right turn - Light Response
        plt.subplot(3, 2, 4)
        light_response_right = right_data[['LS_Left', 'LS_Center', 'LS_Right']].max(axis=1)
        plt.scatter(right_data['Weight_Change'], light_response_right, 
                    label='Right Turn', alpha=0.5, color='red')
        plt.title('Light Response vs Weight Changes - Right Turn')
        plt.xlabel('Weight Change Magnitude')
        plt.ylabel('Light Sensor Response')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/weight_correlation_light.png')
        plt.close()
        
        # Plot 3: Fitness and Distance Sensors vs Weight Changes
        plt.figure(figsize=(12, 16))
        
        # Left turn - Fitness
        plt.subplot(3, 2, 1)
        plt.scatter(left_data['Weight_Change'], left_data['Fitness'], 
                    label='Left Turn', alpha=0.5, color='blue')
        plt.title('Fitness vs Weight Changes - Left Turn')
        plt.xlabel('Weight Change Magnitude')
        plt.ylabel('Fitness')
        plt.grid(True)
        plt.legend()
        
        # Right turn - Fitness
        plt.subplot(3, 2, 2)
        plt.scatter(right_data['Weight_Change'], right_data['Fitness'], 
                    label='Right Turn', alpha=0.5, color='red')
        plt.title('Fitness vs Weight Changes - Right Turn')
        plt.xlabel('Weight Change Magnitude')
        plt.ylabel('Fitness')
        plt.grid(True)
        plt.legend()
        
        # Left turn - Distance Response
        plt.subplot(3, 2, 3)
        distance_response_left = left_data[['PS0', 'PS1', 'PS2', 'PS3', 'PS4', 'PS5', 'PS6', 'PS7']].max(axis=1)
        plt.scatter(left_data['Weight_Change'], distance_response_left, 
                    label='Left Turn', alpha=0.5, color='blue')
        plt.title('Distance Response vs Weight Changes - Left Turn')
        plt.xlabel('Weight Change Magnitude')
        plt.ylabel('Distance Sensor Response')
        plt.grid(True)
        plt.legend()
        
        # Right turn - Distance Response
        plt.subplot(3, 2, 4)
        distance_response_right = right_data[['PS0', 'PS1', 'PS2', 'PS3', 'PS4', 'PS5', 'PS6', 'PS7']].max(axis=1)
        plt.scatter(right_data['Weight_Change'], distance_response_right, 
                    label='Right Turn', alpha=0.5, color='red')
        plt.title('Distance Response vs Weight Changes - Right Turn')
        plt.xlabel('Weight Change Magnitude')
        plt.ylabel('Distance Sensor Response')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/weight_correlation_distance.png')
        plt.close()
        
        # Plot 4: Weight change distribution
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.hist(left_data['Weight_Change'], bins=30, color='blue', alpha=0.7)
        plt.title('Distribution of Weight Changes - Left Turn')
        plt.xlabel('Weight Change Magnitude')
        plt.ylabel('Frequency')
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.hist(right_data['Weight_Change'], bins=30, color='red', alpha=0.7)
        plt.title('Distribution of Weight Changes - Right Turn')
        plt.xlabel('Weight Change Magnitude')
        plt.ylabel('Frequency')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/weight_change_distribution.png')
        plt.close()
    
    def _save_statistics(self, save_dir, left_data, right_data):
        # performance statistics
        stats = {
            'Left Turn': {
                'Total Time (s)': left_data['Time'].max(),
                'Distance': self._calculate_path_length(left_data),
                'Average Speed': self._calculate_path_length(left_data) / left_data['Time'].max(),
                'Average Fitness': left_data['Fitness'].mean(),
                'Final Position Error': self._calculate_target_error(left_data, 'left')
            },
            'Right Turn': {
                'Total Time (s)': right_data['Time'].max(),
                'Distance': self._calculate_path_length(right_data),
                'Average Speed': self._calculate_path_length(right_data) / right_data['Time'].max(),
                'Average Fitness': right_data['Fitness'].mean(),
                'Final Position Error': self._calculate_target_error(right_data, 'right')
            }
        }
        
        if self.plasticity_enabled:
            stats['Left Turn']['Average Weight Change'] = left_data['Weight_Change'].mean()
            stats['Right Turn']['Average Weight Change'] = right_data['Weight_Change'].mean()
        
        with open(f'{save_dir}/statistics.txt', 'w') as f:
            f.write(f"Evaluation Mode: {'Hebbian' if self.plasticity_enabled else 'Normal'}\n\n")
            for turn, data in stats.items():
                f.write(f'\n{turn}:\n')
                f.write('='* len(turn) + '\n')
                for key, value in data.items():
                    f.write(f'{key}: {value:.4f}\n')

    def _calculate_path_length(self, data):
        # total distance traveled
        dx = np.diff(data['Position_X'])
        dy = np.diff(data['Position_Y'])
        dz = np.diff(data['Position_Z'])
        return np.sum(np.sqrt(dx**2 + dy**2 + dz**2))
    
    def _calculate_target_error(self, data, turn_type):
        # distance to target at end of trial
        target = np.array([-0.36, -0.155]) if turn_type == 'left' else np.array([0.36, -0.155])
        final_pos = np.array([data['Position_X'].iloc[-1], data['Position_Z'].iloc[-1]])
        return np.linalg.norm(final_pos - target)

import time

# ...

if __name__ == "__main__":
    supervisor = SupervisorEvaluation()
    
    keyboard = Keyboard()
    keyboard.enable(50)
    
    print("============================================")
    print("Press R to evaluate with geneotype weights")
    print("Press H to evaluate with Synaptic weights")
    print("============================================")
    
    force_genotype = False
    last_key_press_time = 0
    key_press_delay = 1  
    
    while supervisor.supervisor.step(supervisor.time_step) != -1:
        key = keyboard.getKey()
        current_time = time.time()
        
        if current_time - last_key_press_time >= key_press_delay:
            if key == ord('R') or key == ord('r'):
                supervisor.plasticity_enabled = False
                print("\nStarting evaluation with original weights...")
                supervisor.run_demo(force_genotype)
                last_key_press_time = current_time
            elif key == ord('H') or key == ord('h'):
                supervisor.plasticity_enabled = True
                print("\nStarting evaluation with Hebbian learning...")
                supervisor.run_demo(force_genotype)
                last_key_press_time = current_time
            elif key == ord('F') or key == ord('f'):
                force_genotype = not force_genotype
                print(f"\nForced genotype: {'Enabled' if force_genotype else 'Disabled'}")
                last_key_press_time = current_time
            elif key == -1:
                pass 
            else:
                print("\nInvalid key. Please press R, H, or F.")
        else:
            pass  