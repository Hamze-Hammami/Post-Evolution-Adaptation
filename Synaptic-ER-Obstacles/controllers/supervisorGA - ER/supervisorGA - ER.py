from controller import Supervisor, Keyboard
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns
import pandas as pd
import sys
import warnings
from collections import defaultdict
from hebbian_visualizer import HebbianVisualizer

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
        self.time_experiment = 230  
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
        
        # Network state data for visualization
        self.velocity_left = 0.0
        self.velocity_right = 0.0
        self.network_activations = []
        self.network_weights = []
        self.weight_matrices = []
        self.received_weights = {}
        
        # Current trial info
        self.trial_type = ""
        self.current_save_dir = ""
        self.visualization_active = False
        
    def init_visualizer(self):
        """Initialize the Hebbian network visualizer with preloaded weights"""
        try:
            # Network architecture from the controller
            network_arch = [16, 7, 5, 8, 4, 2]  # [input, hidden layers..., output]
            
            print(f"\nInitializing network visualizer with architecture: {network_arch}")
            print(f"Current learning mode: {'Hebbian' if self.plasticity_enabled else 'Normal (non-learning)'}")
            
            # STEP 1: Preload weights from Best.npy BEFORE creating visualizer
            preloaded_weights = None
            try:
                weights_path = "Best.npy"
                if os.path.exists(weights_path):
                    print(f"Preloading weights from {weights_path}")
                    genotype = np.load(weights_path)
                    print(f"Loaded Best.npy with {len(genotype)} values")
                    
                    # Parse the weights into matrices for visualization
                    preloaded_weights = []
                    layer_sizes = network_arch
                    start = 0
                    
                    # First layer includes bias
                    size = (layer_sizes[0] + 1) * layer_sizes[1]
                    if start + size <= len(genotype):
                        layer_weights = genotype[start:start+size].reshape(
                            layer_sizes[0] + 1, layer_sizes[1])
                        # Take only input rows (not bias) for visualization
                        preloaded_weights.append(layer_weights[:-1, :])
                        start += size
                        
                        # Remaining layers
                        for i in range(1, len(layer_sizes)-1):
                            next_size = (layer_sizes[i] + 1) * layer_sizes[i+1]
                            if start + next_size <= len(genotype):
                                layer_weights = genotype[start:start+next_size].reshape(
                                    layer_sizes[i] + 1, layer_sizes[i+1])
                                # Take only input rows (not bias) for visualization
                                preloaded_weights.append(layer_weights[:-1, :])
                                start += next_size
                            else:
                                print(f"Warning: Genotype too short for layer {i}")
                                break
                        
                        print(f"Successfully preloaded {len(preloaded_weights)} weight matrices from genotype")
                        
                        # Also send weights to the robot controller
                        self.emitterData = str(genotype)
                        self.handle_emitter()
                        print("Sent genetic weights to controller")
                    else:
                        print(f"Warning: Genotype from Best.npy is too short ({len(genotype)} values)")
                else:
                    print(f"Warning: {weights_path} not found for initial weight loading")
            except Exception as e:
                print(f"Error preloading weights: {str(e)}")
                import traceback
                traceback.print_exc()
                
            # STEP 2: Create the visualizer, passing the preloaded weights directly
            self.visualizer = HebbianVisualizer(network_arch, preloaded_weights)
            self.visualization_active = False
            
            # STEP 3: Store preloaded weights for future reference
            self.preloaded_weights = preloaded_weights
            
            print("Network visualizer initialized and ready to run with preloaded weights")
            
        except Exception as e:
            print(f"Error initializing visualizer: {str(e)}")
            import traceback
            traceback.print_exc()
    def stop_visualization(self):
        """Stop the network visualization"""
        if hasattr(self, 'visualizer') and self.visualization_active:
            try:
                self.visualizer.stop_animation()
                self.visualization_active = False
                print("Network visualization stopped")
            except Exception as e:
                print(f"Error stopping visualization: {str(e)}")
    
    def start_visualization(self):
        """Start the network visualization with preloaded weights"""
        if hasattr(self, 'visualizer'):
            try:
                # First reset the visualization while keeping weights
                if hasattr(self, 'preloaded_weights') and self.preloaded_weights is not None:
                    print("Resetting visualization with preloaded weights")
                    # Manually reset the visualizer state
                    for i, w in enumerate(self.preloaded_weights):
                        if i < len(self.visualizer.widget.weights):
                            self.visualizer.widget.weights[i] = w.copy()
                            self.visualizer.widget.prev_weights[i] = w.copy()
                    
                    # Reset change detection state
                    for i in range(len(self.visualizer.widget.weight_changes)):
                        self.visualizer.widget.weight_changes[i].fill(0)
                    
                    self.visualizer.widget.changes_detected = False
                    self.visualizer.widget.has_received_real_weights = True
                    # Mark that we're in grace period - don't detect changes yet
                    self.visualizer.in_grace_period = True
                    self.visualizer.first_update = False
                else:
                    # No preloaded weights, just reset flags
                    self.visualizer.reset_visualization(keep_weights=False)
                    self.visualizer.in_grace_period = True
                
                # Set appropriate highlight decay rate based on mode
                if hasattr(self.visualizer, 'widget'):
                    if self.plasticity_enabled:
                        # Slower decay for Hebbian mode to better see subtle changes
                        self.visualizer.widget.highlight_decay_rate = 0.95  # Very slow decay
                        print("Setting slow highlight decay for Hebbian mode")
                    else:
                        # Faster decay for normal mode
                        self.visualizer.widget.highlight_decay_rate = 0.8
                        print("Setting standard highlight decay for normal mode")
                
                self.visualizer.start_animation()
                self.visualization_active = True
                print(f"Network visualization started in {'Hebbian' if self.plasticity_enabled else 'Normal'} mode")
                
                # Force initial weight status update WITHOUT change detection
                self.update_visualization(detect_changes=False)
                
            except Exception as e:
                print(f"Error starting visualization: {str(e)}")
                import traceback
                traceback.print_exc()
        else:
            print("Visualizer not initialized. Call init_visualizer() first.")
    
    
    # Modify update_visualization to accept a detect_changes parameter
    def update_visualization(self, detect_changes=True):
        """Update the visualization with current robot state with accurate learning detection"""
        if hasattr(self, 'visualizer') and self.visualization_active:
            try:
                # Prepare sensor data
                sensors = self.receivedLightSensors + self.receivedProxSensors
                
                # Handle actual network activations if available
                activations = []
                if len(self.network_activations) > 0:
                    for i, act in enumerate(self.network_activations):
                        # Check if activation includes bias node - if so, remove it
                        if i > 0 and len(act) > 0:  # Hidden or output layers may have bias
                            if len(act) == self.visualizer.layer_sizes[i] + 1:
                                act = act[:-1]  # Remove the bias term
                        activations.append(np.array(act))
                
                # If we don't have complete activation data, fill in with dummy data
                if not activations or len(activations) != len(self.visualizer.layer_sizes):
                    # Initialize dummy activations if we don't have them yet
                    if not hasattr(self, 'dummy_activations'):
                        self.dummy_activations = [
                            np.array(sensors),  # Input layer (will be updated)
                            np.random.uniform(-1, 1, self.visualizer.layer_sizes[1]),  # Hidden 1
                            np.random.uniform(-1, 1, self.visualizer.layer_sizes[2]),  # Hidden 2
                            np.random.uniform(-1, 1, self.visualizer.layer_sizes[3]),  # Hidden 3
                            np.random.uniform(-1, 1, self.visualizer.layer_sizes[4]),  # Hidden 4
                            np.array([0, 0])  # Output layer (will be updated)
                        ]
                    
                    # Copy dummy activations but update input and output
                    activations = list(self.dummy_activations)
                    activations[0] = np.array(sensors)  # Always use real sensor data
                    activations[-1] = np.array([self.velocity_left, self.velocity_right])  # Always use real motor data
                
                # Make sure each activation layer matches the expected size
                for i in range(len(activations)):
                    if len(activations[i]) != self.visualizer.layer_sizes[i]:
                        # If size doesn't match, replace with dummy data
                        if i == 0:  # Input layer
                            activations[i] = np.array(sensors)
                        elif i == len(activations) - 1:  # Output layer
                            activations[i] = np.array([self.velocity_left, self.velocity_right])
                        else:  # Hidden layers
                            if hasattr(self, 'dummy_activations') and i < len(self.dummy_activations):
                                activations[i] = self.dummy_activations[i]
                            else:
                                activations[i] = np.random.uniform(-1, 1, self.visualizer.layer_sizes[i])
                
                # Use actual weight matrices if available
                weights = []
                has_valid_weights = False
                if len(self.weight_matrices) > 0:
                    has_valid_weights = True
                    for i, matrix in enumerate(self.weight_matrices):
                        if matrix is not None and matrix.shape == (self.visualizer.layer_sizes[i], self.visualizer.layer_sizes[i+1]):
                            weights.append(matrix)
                        else:
                            has_valid_weights = False
                            break
                
                # If we don't have complete valid weight data, use preloaded weights
                if not has_valid_weights:
                    if hasattr(self, 'preloaded_weights') and self.preloaded_weights is not None:
                        weights = self.preloaded_weights
                        if not hasattr(self, 'using_preloaded_notice'):
                            print("Using preloaded weights from Best.npy")
                            self.using_preloaded_notice = True
                    elif not hasattr(self, 'dummy_weights'):
                        # Fall back to random weights only if we don't have preloaded
                        print("WARNING: No valid weights available, using random dummy weights")
                        self.dummy_weights = []
                        for i in range(len(self.visualizer.layer_sizes) - 1):
                            self.dummy_weights.append(np.random.uniform(-0.5, 0.5, 
                                                    (self.visualizer.layer_sizes[i], 
                                                    self.visualizer.layer_sizes[i+1])))
                        weights = self.dummy_weights
                    else:
                        weights = self.dummy_weights
                else:
                    if not hasattr(self, 'valid_weights_notice_shown'):
                        print(f"Using actual weight matrices from network ({len(weights)} matrices)")
                        self.valid_weights_notice_shown = True
                
                # Get motor outputs
                motor_outputs = np.array([self.velocity_left, self.velocity_right])
                
                # Get weight change magnitude from actual measurements
                weight_change = 0.0
                if self.weight_changes:
                    # For Hebbian mode, use the actual measured change
                    if self.plasticity_enabled:
                        weight_change = self.weight_changes[-1]
                        # Extra debug for Hebbian mode
                        if weight_change > 0.00001 and not hasattr(self, 'last_change_log_time'):
                            self.last_change_log_time = 0
                        
                        # Log every 2 seconds if changes are occurring
                        current_time = time.time()
                        if weight_change > 0.00001 and (not hasattr(self, 'last_change_log_time') or 
                                                    current_time - self.last_change_log_time > 2):
                            print(f"Hebbian weight change: {weight_change:.8f}")
                            self.last_change_log_time = current_time
                
                # If we are in grace period, end it after the first few updates
                if hasattr(self.visualizer, 'in_grace_period') and self.visualizer.in_grace_period:
                    if not hasattr(self.visualizer, 'grace_period_counter'):
                        self.visualizer.grace_period_counter = 0
                    
                    self.visualizer.grace_period_counter += 1
                    if self.visualizer.grace_period_counter >= 10:  # End grace period after 10 updates
                        print("Grace period ended - weight change detection now active")
                        self.visualizer.in_grace_period = False
                    else:
                        # Still in grace period - don't detect changes
                        detect_changes = False
                
                # Update the visualization with the data
                self.visualizer.update_data(
                    sensors=sensors,
                    activations=activations,
                    weights=weights,
                    motor_outputs=motor_outputs,
                    weight_change=weight_change,
                    detect_changes=detect_changes
                )
            except Exception as e:
                print(f"Error updating visualization: {str(e)}")
                import traceback
                traceback.print_exc()

        
    def handle_receiver(self):
        # Process received messages
        while(self.receiver.getQueueLength() > 0):
            try:
                self.receivedData = self.receiver.getString()
                
                if self.receivedData.startswith("weight_change"):
                    change = float(self.receivedData.split()[1])  
                    self.weight_changes.append(change)
                elif self.receivedData.startswith("fitness"):
                    self.receivedFitness = float(self.receivedData.split()[1])
                elif self.receivedData.startswith("sensors:"):
                    parts = self.receivedData[8:].strip().split(":")
                    if len(parts) == 2:
                        self.receivedLightSensors = [float(x) for x in parts[0].split(",")]
                        self.receivedProxSensors = [float(x) for x in parts[1].split(",")]
                elif self.receivedData.startswith("velocity:"):
                    parts = self.receivedData[9:].strip().split(":")
                    if len(parts) == 2:
                        self.velocity_left = float(parts[0])
                        self.velocity_right = float(parts[1])
                elif self.receivedData.startswith("layer"):
                    # Process layer activations
                    layer_idx = int(self.receivedData[5:self.receivedData.find(":")])
                    values_str = self.receivedData[self.receivedData.find(":")+1:]
                    values = [float(x) for x in values_str.split(",")]
                    
                    # Make sure we have enough space in our activations list
                    while len(self.network_activations) <= layer_idx:
                        self.network_activations.append([])
                    
                    self.network_activations[layer_idx] = values
                elif self.receivedData.startswith("weight"):
                    # Process weight matrix data
                    # Format: "weight{layer_idx}_{batch_start}:{values}"
                    header = self.receivedData[:self.receivedData.find(":")]
                    parts = header.split("_")
                    layer_idx = int(parts[0][6:])  # Extract layer index
                    batch_start = int(parts[1])    # Extract batch start index
                    
                    values_str = self.receivedData[self.receivedData.find(":")+1:]
                    values = [float(x) for x in values_str.split(",")]
                    
                    # Store in received_weights dictionary
                    key = f"weight{layer_idx}"
                    if key not in self.received_weights:
                        self.received_weights[key] = {}
                    
                    self.received_weights[key][batch_start] = values
                    
                    # If this is a complete set of weights, reconstruct the matrix
                    self._reconstruct_weight_matrices()
                    
            except Exception as e:
                print(f"Debug - Message processing error: '{str(e)}' for message '{self.receivedData}'")  
                
            self.receiver.nextPacket()

    def _reconstruct_weight_matrices(self):
        """Reconstruct complete weight matrices from received batches"""
        try:
            matrices_constructed = 0
            # Set up network architecture (example - update to match your network)
            layer_sizes = [16, 7, 5, 8, 4, 2]  # [input, hidden..., output]
            
            # For each layer that we've received weights for
            for layer_idx in range(len(layer_sizes) - 1):
                key = f"weight{layer_idx}"
                
                if key not in self.received_weights:
                    continue
                    
                # Calculate expected matrix size
                rows = layer_sizes[layer_idx]
                cols = layer_sizes[layer_idx + 1]
                expected_size = rows * cols
                
                # Count total received values
                received_values = []
                for batch_start in sorted(self.received_weights[key].keys()):
                    received_values.extend(self.received_weights[key][batch_start])
                
                # If we have all the values, reshape into matrix
                if len(received_values) >= expected_size:
                    # Trim to expected size in case of extra values
                    received_values = received_values[:expected_size]
                    matrix = np.array(received_values).reshape(rows, cols)
                    
                    # Make sure we have enough entries in weight_matrices
                    while len(self.weight_matrices) <= layer_idx:
                        self.weight_matrices.append(None)
                        
                    self.weight_matrices[layer_idx] = matrix
                    matrices_constructed += 1
                    
                    # Clear the received batches to save memory
                    self.received_weights[key] = {}
                    
            if matrices_constructed > 0:
                print(f"Successfully reconstructed {matrices_constructed} weight matrices")
                
            return matrices_constructed > 0
        except Exception as e:
            print(f"Error reconstructing weight matrices: {str(e)}")
            return False
    
    def handle_emitter(self):
        #Send messages to robot
        if self.emitterData:
            self.emitter.send(str(self.emitterData).encode("utf-8"))
            self.emitterData = ""  
            
        plasticity_msg = f"hebbian:{self.plasticity_enabled}".encode("utf-8")
        self.emitter.send(plasticity_msg)
    
            
    def run_trial(self, trial_type, save_dir, genotype, force_genotype=False, visualize=False):
        # run trial
        self.trial_type = trial_type
        self.current_save_dir = save_dir
        
        print(f"\nRunning {trial_type} trial...")
        print(f"Mode: {'Hebbian Learning' if self.plasticity_enabled else 'Normal Operation'}")
        print(f"Trial duration: {self.time_experiment} seconds")
        print(f"Visualization: {'Enabled' if visualize else 'Disabled'}")
        
        self.emitterData = str(genotype)
        self.weight_changes = []
        
        # First, ensure the controller has the correct genetic weights - BEFORE ANYTHING ELSE
        self.handle_emitter()  # Send genotype to controller
        print("Sent genetic weights to controller")
        
        # Wait for the controller to process the weights - LONGER INITIAL WAIT
        print("Waiting for controller to process initial weights...")
        for i in range(20):  # Increased from 5 to 20 timesteps
            if self.supervisor.step(self.time_step) == -1:
                return None
            self.handle_receiver()  # Process any messages from controller
            
            # Add a progress indicator
            if i % 5 == 0:
                print(f"  Controller initialization progress: {i*5}%")
        
        print("Controller initialized with genetic weights")
        
        # Initialize and start visualization AFTER controller has processed weights
        if visualize:
            try:
                # Initialize visualizer if needed
                if not hasattr(self, 'visualizer') or self.visualizer is None:
                    self.init_visualizer()
                    print("Visualizer initialized")
                    
                # Reset any previous state
                if hasattr(self, 'visualizer'):
                    self.visualizer.reset_visualization(keep_weights=False)
                    print("Visualization state reset")
                
                # Explicitly set a longer grace period
                if hasattr(self, 'visualizer'):
                    self.visualizer.in_grace_period = True
                    # Set a higher counter threshold for ending grace period
                    self.visualizer.grace_period_counter = 0
                    self.visualizer.grace_period_threshold = 40  # Increased threshold (was default 20)
                    print(f"Extended grace period activated: {self.visualizer.grace_period_threshold} iterations")
                
                # Start visualization in grace period mode
                self.start_visualization()
                
                # Give the visualization time to initialize and stabilize - LONGER WAIT
                print("Allowing visualization to fully initialize...")
                time.sleep(2.0)  # Increased from 1.0 to 2.0 seconds
                
                # Force weights to be sent to the controller again to ensure synchronization
                self.emitterData = str(genotype)
                self.handle_emitter()
                print("Reinforced genetic weights to controller after visualization start")
                
                # Additional wait to ensure controller and visualization are in sync - LONGER SYNC PERIOD
                print("Synchronizing controller and visualization states...")
                for i in range(10):  # Increased from 3 to 10 timesteps
                    if self.supervisor.step(self.time_step) == -1:
                        return None
                    self.handle_receiver()  # Process any messages from controller
                    self.update_visualization(detect_changes=False)  # Update visualization without detecting changes
                    
                    # Add a small delay between updates for better visualization stability
                    time.sleep(0.1)
                
                print("Grace period active - visualization ready and stabilized")
                
            except Exception as e:
                print(f"Visualization failed to start: {e}")
                import traceback
                traceback.print_exc()
        
        # Reset robot position AFTER visualization is initialized
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
        
        # Set lower time steps to slow down the simulation if visualization is active
        effective_time_step = self.time_step
        if visualize and hasattr(self, 'visualization_active') and self.visualization_active:
            print("Simulation ready - slowing down for better visualization...")
            # Don't change the time_step, but add small pauses between iterations
        
        # CLEAR SEPARATOR to indicate transition from initialization to main loop
        print("\n" + "="*50)
        print("INITIALIZATION COMPLETE - STARTING TRIAL")
        print("="*50 + "\n")
        
        print(f"Starting trial loop ({stop} iterations)...")
        
        while iterations < stop and self.supervisor.step(effective_time_step) != -1:
            current_time = self.supervisor.getTime() - start_time
            position = self.robot_node.getPosition()
            
            # ----- bookkeeping -----
            GOAL_LEFT = np.array([-0.36, -0.155])
            GOAL_RIGHT = np.array([0.36, -0.155])
            goal_pt = GOAL_RIGHT if trial_type == "right" else GOAL_LEFT
            dist_to_goal = np.linalg.norm(goal_pt - np.array([position[0], position[2]]))
            collision = (np.max(self.receivedProxSensors) > 0.8)  # tweak threshold if needed
            success = (dist_to_goal < 0.05)  # reached target cone?
            # --------------------------------
            
            # Handle communication with the robot
            self.handle_emitter()
            self.handle_receiver()
            
            # Update visualization if active - normal updates with change detection
            if hasattr(self, 'visualization_active') and self.visualization_active:
                # End grace period after the specified threshold of iterations
                if (hasattr(self.visualizer, 'in_grace_period') and 
                    self.visualizer.in_grace_period and 
                    iterations > getattr(self.visualizer, 'grace_period_threshold', 20)):
                    print(f"Ending grace period at iteration {iterations} - weight change detection now active")
                    self.visualizer.in_grace_period = False
                
                # Update visualization - detect changes only if grace period has ended
                detect_changes = not getattr(self.visualizer, 'in_grace_period', False)
                self.update_visualization(detect_changes=detect_changes)
                
                # Add a small pause every few iterations to let the visualization update
                if iterations % 5 == 0:  # More frequent updates (was 10)
                    time.sleep(0.02)  # Small delay to allow visualization to update (was 0.01)
            
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
                'Weight_Change': current_weight_change,
                'Dist_To_Goal': dist_to_goal,
                'Collision': int(collision),
                'Success': int(success)
            }
            
            for i in range(8):
                row_data[f'PS{i}'] = self.receivedProxSensors[i]
            
            trial_data.append(row_data)
            iterations += 1
            
            if iterations % (stop // 10) == 0:
                print(f"Progress: {iterations/stop*100:.1f}%")
                if self.plasticity_enabled and self.weight_changes:
                    print(f"Current weight change: {current_weight_change:.8f}")
                    print(f"Average weight change: {np.mean(self.weight_changes):.8f}")
            
            # Print a message when grace period ends
            if (visualize and 
                hasattr(self, 'visualizer') and 
                hasattr(self.visualizer, 'in_grace_period') and 
                iterations == getattr(self.visualizer, 'grace_period_threshold', 20)):
                print("\n" + "-"*50)
                print("GRACE PERIOD ENDED - WEIGHT CHANGE DETECTION NOW ACTIVE")
                print("-"*50 + "\n")
        
        # Keep visualization running for a moment after trial ends
        if hasattr(self, 'visualization_active') and self.visualization_active:
            print("Trial completed. Visualization will remain open for a few seconds...")
            time.sleep(5)  # Give user time to see final state
            self.stop_visualization()
        
        df = pd.DataFrame(trial_data)
        
        print(f"\nTrial Summary ({trial_type}):")
        print(f"Final position: ({position[0]:.3f}, {position[2]:.3f})")
        print(f"Average fitness: {df['Fitness'].mean():.4f}")
        if self.plasticity_enabled:
            print(f"Weight changes recorded: {len(self.weight_changes)}")
            print(f"Total weight change: {np.sum(self.weight_changes):.8f}")
            print(f"Mean weight change: {np.mean(self.weight_changes):.8f}")
        
        return df 
    
    def run_demo(self, force_genotype=False, visualize=False):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode = "hebbian" if self.plasticity_enabled else "normal"
        base_dir = "robot_evaluation"
        os.makedirs(base_dir, exist_ok=True)
        save_dir = os.path.join(base_dir, f"evaluation_logs_{mode}_{timestamp}")
        os.makedirs(save_dir, exist_ok=True)
        try:
            # Load the weights from the NPY file
            genotype = np.load("Best.npy")
            print(f"Loaded Best.npy genotype with {len(genotype)} values")
            
            # Print mode information
            print(f"\n{'='*50}")
            print(f"RUNNING EVALUATION DEMO")
            print(f"Mode: {'HEBBIAN LEARNING ENABLED' if self.plasticity_enabled else 'NORMAL MODE (NO LEARNING)'}")
            print(f"Force Genotype: {'Enabled' if force_genotype else 'Disabled'}")
            print(f"Visualization: {'Enabled' if visualize else 'Disabled'}")
            print(f"{'='*50}\n")
            
            # If visualization is enabled, ensure any previous instance is properly closed
            if visualize and hasattr(self, 'visualizer') and self.visualization_active:
                print("Stopping previous visualization...")
                self.stop_visualization()
                time.sleep(1)  # Give time for cleanup
            
            print("Running left turn trial...")
            left_data = self.run_trial("left", save_dir, genotype, force_genotype, visualize)
            
            # If visualization is enabled, ensure it's properly closed between trials
            if visualize and hasattr(self, 'visualizer') and self.visualization_active:
                print("Stopping visualization before next trial...")
                self.stop_visualization()
                time.sleep(1)  # Give time for cleanup
            
            print("Running right turn trial...")
            right_data = self.run_trial("right", save_dir, genotype, force_genotype, visualize)
            
            print("Generating visualizations and statistics...")
            self.plot_evaluation_results(save_dir, left_data, right_data)
            print("\nEvaluation complete!")
            print(f"Results saved in: {save_dir}")
        except Exception as e:
            print(f"Error during evaluation: {str(e)}")
            import traceback
            traceback.print_exc()
            
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
    
    def _aggregate_trial_stats(self, df):
        """Return dict of scalar metrics for a single DataFrame trial."""
        return {
            'success_rate': df['Success'].iloc[-1],  # 0/1 for this trial
            'time_to_goal': df.loc[df['Success'] == 1, 'Time'].min() if df['Success'].any() else np.nan,
            'avg_speed': self._calculate_path_length(df) / df['Time'].max(),
            'path_length': self._calculate_path_length(df),
            'mean_collision': df['Collision'].mean(),
            'mean_weight_delta': df['Weight_Change'].mean() if 'Weight_Change' in df else np.nan,
        }
    
    def _save_statistics(self, save_dir, left_df, right_df):
        # performance statistics
        left_metrics = self._aggregate_trial_stats(left_df)
        right_metrics = self._aggregate_trial_stats(right_df)
        
        stats = {
            'Left Turn': {
                'Success (0/1)': left_metrics['success_rate'],
                'Time-to-Goal (s)': left_metrics['time_to_goal'],
                'Path Length (m)': left_metrics['path_length'],
                'Average Speed (m/s)': left_metrics['avg_speed'],
                'Collisions / step': left_metrics['mean_collision'],
                'Average Fitness': left_df['Fitness'].mean(),
                'Final Position Error': self._calculate_target_error(left_df, 'left'),
                'Weight Change Mean': left_metrics['mean_weight_delta'] if self.plasticity_enabled else 'N/A',
            },
            'Right Turn': {
                'Success (0/1)': right_metrics['success_rate'],
                'Time-to-Goal (s)': right_metrics['time_to_goal'], 
                'Path Length (m)': right_metrics['path_length'],
                'Average Speed (m/s)': right_metrics['avg_speed'],
                'Collisions / step': right_metrics['mean_collision'],
                'Average Fitness': right_df['Fitness'].mean(),
                'Final Position Error': self._calculate_target_error(right_df, 'right'),
                'Weight Change Mean': right_metrics['mean_weight_delta'] if self.plasticity_enabled else 'N/A',
            }
        }
        
        # Write statistics to both text and CSV formats
        with open(f'{save_dir}/statistics.txt', 'w', encoding='utf-8') as f:
            f.write(f"Evaluation Mode: {'Hebbian' if self.plasticity_enabled else 'Normal'}\n\n")
            for turn, data in stats.items():
                f.write(f'\n{turn}:\n')
                f.write('='* len(turn) + '\n')
                for key, value in data.items():
                    f.write(f'{key}: {value:.4f}\n' if isinstance(value, (int, float)) else f'{key}: {value}\n')
                    
        # Create a combined metrics CSV with all the important metrics
        combined_data = {
            'Metric': list(stats['Left Turn'].keys()),
            'Left Turn': [stats['Left Turn'][k] for k in stats['Left Turn'].keys()],
            'Right Turn': [stats['Right Turn'][k] for k in stats['Right Turn'].keys()]
        }
        # Use encoding='ascii' to handle special characters
        pd.DataFrame(combined_data).to_csv(f'{save_dir}/statistics.csv', index=False, float_format='%.4f', encoding='ascii', errors='replace')

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

# Main program
if __name__ == "__main__":
    supervisor = SupervisorEvaluation()
    
    keyboard = Keyboard()
    keyboard.enable(50)
    
    print("============================================")
    print("Press R to evaluate with geneotype weights")
    print("Press H to evaluate with Synaptic weights")
    print("Press V to toggle visualization")
    print("============================================")
    
    force_genotype = False
    visualize = False
    last_key_press_time = 0
    key_press_delay = 1  
    
    while supervisor.supervisor.step(supervisor.time_step) != -1:
        key = keyboard.getKey()
        current_time = time.time()
        
        if current_time - last_key_press_time >= key_press_delay:
            if key == ord('R') or key == ord('r'):
                supervisor.plasticity_enabled = False
                print("\nStarting evaluation with original weights...")
                supervisor.run_demo(force_genotype, visualize)
                last_key_press_time = current_time
            elif key == ord('H') or key == ord('h'):
                supervisor.plasticity_enabled = True
                print("\nStarting evaluation with Hebbian learning...")
                supervisor.run_demo(force_genotype, visualize)
                last_key_press_time = current_time
            elif key == ord('F') or key == ord('f'):
                force_genotype = not force_genotype
                print(f"\nForced genotype: {'Enabled' if force_genotype else 'Disabled'}")
                last_key_press_time = current_time
            elif key == ord('V') or key == ord('v'):
                visualize = not visualize
                print(f"\nVisualization: {'Enabled' if visualize else 'Disabled'}")
                last_key_press_time = current_time
            elif key == -1:
                pass 
            else:
                print("\nInvalid key. Please press R, H, F, or V.")
        else:
            pass