from controller import Robot, Receiver, Emitter
import numpy as np
import mlp as ntw

# Synaptic Hebbian Evaluation
# Copyright (c) 2024 Hamze Hammami (ORCID: 0009-0004-5754-5842)
#
# This code applies 'Hebbian learning' to pre-trained genetic algorithm genes 
# to study the effects of synaptic and stimuli reactions in a robotic system.
#
# License:
# 1. Academic Use Only: This code is for evaluation and academic purposes only. 
#    Redistribution, modification, or commercial use is prohibited.
# 2. No Modifications: Modifications or derivative works are not allowed until 
#    the related research paper is officially published.
# 3. Attribution: Proper credit must be given if this code is referenced in 
#    academic or research contexts.
# 4. Liability: This code is provided "as is," without guarantees or warranties. 
#    The author is not responsible for any issues arising from its use.

class Controller:
    def __init__(self, robot):
        self.fitness_values = []
        self.current_fitness = 0
        self.fitness = 0
        
        self.robot = robot
        self.time_step = 32 
        self.max_speed = 1   
 
        self.number_input_layer = 16 
        self.number_hidden_layer = [7,5,8,4]
        self.number_output_layer = 2
        
        self.number_neuros_per_layer = [self.number_input_layer]
        self.number_neuros_per_layer.extend(self.number_hidden_layer)
        self.number_neuros_per_layer.append(self.number_output_layer)
        
        self.network = ntw.MLP(self.number_neuros_per_layer, hebbian_rate=0.000015)
        self.inputs = []

        self.left_motor = self.robot.getDevice('left wheel motor')
        self.right_motor = self.robot.getDevice('right wheel motor')
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)
        self.velocity_left = 0
        self.velocity_right = 0
    
        self.proximity_sensors = []
        self.light_sensors = []
        for i in range(8):
            sensor_name = 'ps' + str(i)
            self.proximity_sensors.append(self.robot.getDevice(sensor_name))
            self.proximity_sensors[i].enable(self.time_step)
            sensor_name = 'ls' + str(i)
            self.light_sensors.append(self.robot.getDevice(sensor_name))
            self.light_sensors[i].enable(self.time_step)
        
        self.emitter = self.robot.getDevice("emitter") 
        self.receiver = self.robot.getDevice("receiver") 
        self.receiver.enable(self.time_step)
        
        self.initial_light_state = None
        self.has_reached_junction = False
        self.stage = "FORWARD"
        
        # For tracking weight changes
        self.last_sent_weights = None
    
    def detect_junction(self):
        """Detect T-junction based on sensor readings"""
        front_proximity = max(self.inputs[9:11])  
        side_proximity = max(self.inputs[8], self.inputs[11])  
        return (front_proximity < 0.2 and side_proximity > 0.6)

    def read_initial_light(self):
        """Detect initial light condition"""
        return np.mean(self.inputs[0:8]) < 0.3

    def determine_turn_direction(self):
        """Determine turn direction based on light detection"""
        return "RIGHT" if self.initial_light_state else "LEFT"

    def execute_turn(self, direction):
        """Execute turning movement"""
        if direction == "RIGHT":
            self.velocity_left = self.max_speed
            self.velocity_right = -self.max_speed
        else:  
            self.velocity_left = -self.max_speed
            self.velocity_right = self.max_speed


    def calculate_fitness(self):
        light_readings = self.inputs[0:8]
        proximity_readings = self.inputs[8:16]
        
        ###########
        ### DEFINE the fitness function to increase the speed of the robot and 
        ### to encourage the robot to move forward
        forwardFitness=(self.velocity_left+self.velocity_right)/1.5 
        if (forwardFitness>1):
            forwardFitness=1
        if (forwardFitness<0):
            forwardFitness=0
                      
        ###########
        ### DEFINE the fitness function equation to avoid collision
        if(np.max(self.inputs[8:16])):
            avoidCollisionFitness =1-((np.max(self.inputs[8:16]))**3) 


        junctionFitness = 0
        if self.has_reached_junction:
            if self.initial_light_state and self.velocity_right > self.velocity_left:
                junctionFitness = 1  # Reward right turn when light detected (low sensor values)
            elif not self.initial_light_state and self.velocity_left > self.velocity_right:
                junctionFitness = 1  # Reward left turn when no light detected (high sensor values)
        
        
        ###########
        ### DEFINE the fitness function equation to avoid spining behaviour
        if(self.velocity_right>=self.velocity_left):#Condition to check if the robot is left spinning or right spinning
            spinningFitness=np.abs(self.velocity_left-self.velocity_right)
        if(self.velocity_right<self.velocity_left):
            spinningFitness=np.abs(self.velocity_right-self.velocity_left)
        spinningFitness=1-(spinningFitness/2)
        if(spinningFitness<0):
            spinningFitness=0
        
        ###########
        ### DEFINE the fitness function equation of this iteration which should be a combination of the previous functions         
        combinedFitness = (forwardFitness + (2*avoidCollisionFitness) + spinningFitness +junctionFitness)/5
        
        self.fitness_values.append(combinedFitness)
        self.fitness = np.mean(self.fitness_values) 
        
        return self.fitness
    
    def send_network_state(self):
        """Send neural network state to supervisor for visualization"""
        if not hasattr(self, 'network'):
            return
        
        # Send velocity information
        velocity_msg = f"velocity:{self.velocity_left:.6f}:{self.velocity_right:.6f}"
        self.emitter.send(velocity_msg.encode("utf-8"))
        
        # Pack network state information
        # This is more complex and may need to be split into multiple messages
        
        # 1. First, send layer activations
        for i, layer in enumerate(self.network.layers):
            # Convert layer activations to string (avoid sending bias unit for hidden layers)
            if i == 0:  # Input layer (no bias)
                activation_str = ",".join([f"{x:.6f}" for x in layer])
            else:  # Hidden/output layers (exclude bias unit)
                activation_str = ",".join([f"{x:.6f}" for x in layer])
                
            layer_msg = f"layer{i}:{activation_str}"
            self.emitter.send(layer_msg.encode("utf-8"))
        
        # 2. Then, send weight matrices (can be large, so we'll compress)
        # We'll only send this periodically to avoid overloading communication
        if self.robot.getTime() % 1.0 < 0.1:  # Send weights every ~1 second
            # Initialize last_sent_weights if needed
            if not hasattr(self, 'last_sent_weights') or self.last_sent_weights is None:
                self.last_sent_weights = [None] * len(self.network.weights)
            
            for i, w in enumerate(self.network.weights):
                # Flatten and compress the weight matrix
                flat_weights = w.flatten()
                
                # Only send weights that have changed significantly
                should_send = True
                if self.last_sent_weights[i] is not None:  # Only check if we have previous weights
                    weight_diff = np.abs(flat_weights - self.last_sent_weights[i])
                    if np.max(weight_diff) < 0.01:
                        should_send = False  # Skip if weights haven't changed much
                
                if should_send:
                    # Store weights for comparison next time
                    self.last_sent_weights[i] = flat_weights.copy()
                    
                    # Send in batches of 50 values to avoid buffer overflow
                    batch_size = 50
                    for batch_start in range(0, len(flat_weights), batch_size):
                        batch_end = min(batch_start + batch_size, len(flat_weights))
                        batch = flat_weights[batch_start:batch_end]
                        weight_str = ",".join([f"{x:.6f}" for x in batch])
                        weight_msg = f"weight{i}_{batch_start}:{weight_str}"
                        self.emitter.send(weight_msg.encode("utf-8"))
    
    def handle_receiver(self):
        """Process received network weights and hebbian control"""
        if self.receiver.getQueueLength() > 0:
            while self.receiver.getQueueLength() > 0:
                message = self.receiver.getString()
                
                if message.startswith("hebbian:"):
                    enabled = message[8:] == "True"
                    self.network.set_plasticity(enabled)
                    
                elif message.startswith("["):
                    data = message[1:-1].split()
                    weights = np.array(data).astype(float)
                    
                    start = 0
                    network_weights = []
                    
                    size = (self.number_neuros_per_layer[0] + 1) * self.number_neuros_per_layer[1]
                    layer_weights = weights[start:start+size].reshape(
                        self.number_neuros_per_layer[0] + 1, self.number_neuros_per_layer[1])
                    network_weights.append(layer_weights)
                    start += size
                    
                    for i in range(1, len(self.number_neuros_per_layer)-1):
                        size = self.number_neuros_per_layer[i] * self.number_neuros_per_layer[i+1]
                        layer_weights = weights[start:start+size].reshape(
                            self.number_neuros_per_layer[i], self.number_neuros_per_layer[i+1])
                        network_weights.append(layer_weights)
                        start += size
                    
                    self.network.set_genetic_weights(network_weights)
                
                self.receiver.nextPacket()

    def handle_emitter(self):
        """Send sensor and weight change data to supervisor"""
        if len(self.inputs) >= 16:
            light_sensors = ",".join([f"{x:.6f}" for x in self.inputs[0:8]])
            prox_sensors = ",".join([f"{x:.6f}" for x in self.inputs[8:16]])
            sensor_msg = f"sensors:{light_sensors}:{prox_sensors}"
            self.emitter.send(sensor_msg.encode("utf-8"))
            
            fitness_msg = f"fitness {self.fitness:.6f}"  
            self.emitter.send(fitness_msg.encode("utf-8"))
            
            if self.network.plasticity_enabled:
                total_change = sum(np.sum(np.abs(w - iw)) for w, iw in 
                                 zip(self.network.weights, self.network.initial_weights))
                change_msg = f"weight_change {total_change:.6f}"  
                self.emitter.send(change_msg.encode("utf-8"))
            
            # Send network state for visualization
            self.send_network_state()

    def sense_compute_and_actuate(self):
        """Main control loop with plasticity"""
        self.inputs = []
        
        for i in range(8):
            temp = np.clip(self.light_sensors[i].getValue(), 0, 4096)
            self.inputs.append(temp/4096)
        
        for i in range(8):
            temp = np.clip(self.proximity_sensors[i].getValue(), 0, 2000)
            self.inputs.append(temp/2000)
    
        output = self.network.propagate_forward(self.inputs)
        self.velocity_left = output[0]
        self.velocity_right = output[1]
        self.calculate_fitness()
        if self.network.plasticity_enabled:
            weight_change = self.network.update_synaptic_weights(self.fitness)
        
        self.left_motor.setVelocity(self.velocity_left * self.max_speed)
        self.right_motor.setVelocity(self.velocity_right * self.max_speed)
        
        self.handle_emitter()

    def run_robot(self):        
        while self.robot.step(self.time_step) != -1:
            self.handle_emitter()
            self.handle_receiver()
            self.sense_compute_and_actuate()

if __name__ == "__main__":
    controller = Controller(Robot())
    controller.run_robot()