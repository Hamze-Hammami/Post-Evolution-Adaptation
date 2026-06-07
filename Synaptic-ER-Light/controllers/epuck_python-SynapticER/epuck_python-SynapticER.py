from controller import Robot, Receiver, Emitter
import numpy as np
import hebmlp as ntw

# Synaptic Gene Evaluation
# Copyright (c) 2024 Hamze Hammami (ORCID: 0009-0004-5754-5842)

# The full controller and supervisor scripts apply'Hebbian learning' to pre-trained genetic algorithm genes
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

class Controller:
    # fitness tracking
    def __init__(self, robot):
        self.fitness_values = []
        self.current_fitness = 0
        self.fitness = 0
        # robot parameters 
        self.robot = robot
        self.time_step = 32 
        self.max_speed = 1   
        # MLP architecture 
        self.number_input_layer = 16 
        self.number_hidden_layer = [7,5,8,4]
        self.number_output_layer = 2
        
        self.number_neuros_per_layer = [self.number_input_layer]
        self.number_neuros_per_layer.extend(self.number_hidden_layer)
        self.number_neuros_per_layer.append(self.number_output_layer)
        # init network with hebbian  
        self.network = ntw.MLP(self.number_neuros_per_layer, hebbian_rate=0.002) #0.002
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
    
    def detect_junction(self):
        # Detect T-junction
        front_proximity = max(self.inputs[9:11])  
        side_proximity = max(self.inputs[8], self.inputs[11])  
        return (front_proximity < 0.2 and side_proximity > 0.6)

    def read_initial_light(self):
        #Detect light condition
        return np.mean(self.inputs[0:8]) < 0.3

    def determine_turn_direction(self):
        # Determine turn direction based on light detection
        return "RIGHT" if self.initial_light_state else "LEFT"

    def execute_turn(self, direction):
        # Execute turning movement
        if direction == "RIGHT":
            self.velocity_left = self.max_speed
            self.velocity_right = -self.max_speed
        else:  
            self.velocity_left = -self.max_speed
            self.velocity_right = self.max_speed
            
    def calculate_fitness(self):
        # Calculate fitness to influnce plasticity
        light_readings = self.inputs[0:8]
        proximity_readings = self.inputs[8:16]
        
        forwardFitness = (self.velocity_left + self.velocity_right)/1.5 
        forwardFitness = np.clip(forwardFitness, 0, 1)
        
        if(np.max(self.inputs[3:11])):
            avoidCollisionFitness = 1-((np.max(self.inputs[3:11]))**3) 
    
        right_light = 1-np.mean(light_readings[0:4])
        left_light = 1-np.mean(light_readings[4:8])
        lightFollowingFitness = 1 - abs(right_light - left_light)
        
        junctionFitness = 0
        if self.has_reached_junction:
            if self.initial_light_state and self.velocity_right > self.velocity_left:
                junctionFitness = 1  # Reward right turn
            elif not self.initial_light_state and self.velocity_left > self.velocity_right:
                junctionFitness = 1  # Reward left turn
        
        if(self.velocity_right >= self.velocity_left):
            spinningFitness = np.abs(self.velocity_left-self.velocity_right)
        else:
            spinningFitness = np.abs(self.velocity_right-self.velocity_left)
        spinningFitness = 1-(spinningFitness/2)
        spinningFitness = max(spinningFitness, 0)
        
        self.fitness = (forwardFitness + (2*avoidCollisionFitness) + 
                       spinningFitness + lightFollowingFitness + junctionFitness)/6
        
        self.fitness_values.append(self.fitness)
        self.current_fitness = np.mean(self.fitness_values)
        
        return self.fitness
    
    def handle_receiver(self):
        # receiver
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
        # emitter
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

    def sense_compute_and_actuate(self):
        # Main control loop ( plasticity added ) 
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