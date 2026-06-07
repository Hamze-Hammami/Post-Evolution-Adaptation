from controller import Supervisor
from controller import Keyboard
from controller import Display
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import numpy as np
import ga, os, sys, struct

class SupervisorGA:
    def __init__(self):
        # Simulation Parameters
        # Please, do not change these parameters
        self.time_step = 32 # ms
        self.time_experiment = 60 # s
        
        # Initiate Supervisor Module
        self.supervisor = Supervisor()
        # Check if the robot node exists in the current world file
        self.robot_node = self.supervisor.getFromDef("Controller")
        if self.robot_node is None:
            sys.stderr.write("No DEF Controller node found in the current world file\n")
            sys.exit(1)
        # Get the robots translation and rotation current parameters    
        self.trans_field = self.robot_node.getField("translation")  
        self.rot_field = self.robot_node.getField("rotation")
        
        # Check Receiver and Emitter are enabled
        self.emitter = self.supervisor.getDevice("emitter")
        self.receiver = self.supervisor.getDevice("receiver")
        self.receiver.enable(self.time_step)
        
        # Initialize the receiver and emitter data to null
        self.receivedData = "" 
        self.receivedWeights = "" 
        self.receivedFitness = "" 
        self.emitterData = ""
        
        ###########
        ### DEFINE here the 3 GA Parameters:
        self.num_generations = 30
        self.num_population = 50
        self.num_elite = 6
        
        # size of the genotype variable
        self.num_weights = 0
        
        # Creating the initial population
        self.population = []
        
        # All Genotypes
        self.genotypes = []
        
        # Lists to store fitness history for plotting
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.generations_history = []
        
        # Create matplotlib figure
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.ax.set_xlabel('Generation')
        self.ax.set_ylabel('Fitness')
        self.ax.set_title('Evolution Progress')
        self.ax.grid(True)
        self.ax.set_ylim(0, 1)
        
        # Display: screen to plot the fitness values
        self.display = self.supervisor.getDevice("display")
        self.width = self.display.getWidth()
        self.height = self.display.getHeight()
        self.prev_best_fitness = 0.0
        self.prev_average_fitness = 0.0
        self.display.drawText("Fitness (Best - Red)", 0,0)
        self.display.drawText("Fitness (Average - Green)", 0,10)
        
        # Black mark
        self.mark_node = self.supervisor.getFromDef("Mark")
        if self.mark_node is None:
            sys.stderr.write("No DEF Mark node found in the current world file\n")
            sys.exit(1)
        self.mark_trans_field = self.mark_node.getField("translation")      

        # spot light node 
        self.spot_light_node = self.supervisor.getFromDef("spot_light")
        if self.spot_light_node is None:
            sys.stderr.write("the spot_light DEF node not found :)")
            sys.exit(1)
        self.spot_light_status = self.spot_light_node.getField("on")

    def update_plot(self, generation, best_fitness, average_fitness):
       
        self.generations_history.append(generation)
        self.best_fitness_history.append(best_fitness)
        self.avg_fitness_history.append(average_fitness)
        
        # Clear the previous plot
        self.ax.clear()
        
        # Plot the new data
        self.ax.plot(self.generations_history, self.best_fitness_history, 'r-', label='Best Fitness')
        self.ax.plot(self.generations_history, self.avg_fitness_history, 'g-', label='Average Fitness')
        
        # Customize the plot
        self.ax.set_xlabel('Generation')
        self.ax.set_ylabel('Fitness')
        self.ax.set_title('Evolution Progress')
        self.ax.grid(True)
        self.ax.legend()
        self.ax.set_ylim(0, 1)
        
        # Add minor gridlines
        self.ax.grid(True, which='minor', linestyle=':', alpha=0.5)
        self.ax.minorticks_on()
        
        # Save plot to file
        plt.savefig('fitness_evolution.png')
        plt.close()

    def save_plot(self):
        #Save the final plot
        plt.figure(figsize=(10, 8))
        plt.plot(self.generations_history, self.best_fitness_history, 'r-', label='Best Fitness', linewidth=2)
        plt.plot(self.generations_history, self.avg_fitness_history, 'g-', label='Average Fitness', linewidth=2)
        plt.xlabel('Generation', fontsize=12)
        plt.ylabel('Fitness', fontsize=12)
        plt.title('Evolution Progress', fontsize=14)
        plt.grid(True)
        plt.legend(fontsize=10)
        plt.ylim(0, 1)
        
        # Add minor gridlines
        plt.grid(True, which='minor', linestyle=':', alpha=0.5)
        plt.minorticks_on()
        
        # Save high-resolution plot
        plt.savefig('final_fitness_evolution.png', dpi=300, bbox_inches='tight')
        plt.close()

    def createRandomPopulation(self):
        # Wait until the supervisor receives the size of the genotypes (number of weights)
        if(self.num_weights > 0):
            #  Size of the population and genotype
            pop_size = (self.num_population,self.num_weights)
            # Create the initial population with random weights
            self.population = np.random.uniform(low=-1.0, high=1.0, size=pop_size)

    def handle_receiver(self):
        while(self.receiver.getQueueLength() > 0):
            self.receivedData = self.receiver.getString()
            typeMessage = self.receivedData[0:7]
            # Check Message 
            if(typeMessage == "weights"):
                self.receivedWeights = self.receivedData[9:len(self.receivedData)] 
                self.num_weights = int(self.receivedWeights)
            elif(typeMessage == "fitness"):  
                self.receivedFitness = float(self.receivedData[9:len(self.receivedData)])
            self.receiver.nextPacket()
        
    def handle_emitter(self):
        if(self.num_weights > 0):
            # Send genotype of an individual
            string_message = str(self.emitterData)
            string_message = string_message.encode("utf-8")
            self.emitter.send(string_message)     
        
    def run_seconds(self,seconds):
        stop = int((seconds*1000)/self.time_step)
        iterations = 0
        while self.supervisor.step(self.time_step) != -1:
            self.handle_emitter()
            self.handle_receiver()
            if(stop == iterations):
                break    
            iterations = iterations + 1

    def evaluate_genotype(self,genotype,generation):
        numberofInteractionLoops = 3
        currentInteraction = 0
        fitnessPerTrial = []
        while currentInteraction < numberofInteractionLoops:
            #######################################
            # TRIAL: TURN RIGHT
            #######################################
            self.emitterData = str(genotype)
            
            INITIAL_TRANS = [0.007, 0, 0.35]
            self.trans_field.setSFVec3f(INITIAL_TRANS)
            INITIAL_ROT = [-0.5, 0.5, 0.5, 2.09]
            self.rot_field.setSFRotation(INITIAL_ROT)
            self.robot_node.resetPhysics()
            
            self.spot_light_node.getField('on').setSFBool(True)
            
            self.run_seconds(self.time_experiment)
            
            fitness = self.receivedFitness
            
            [tx,ty,tz] = self.robot_node.getPosition()
            dist = np.sqrt((0.36-tx)**2 + (-0.155-tz)**2)
            dist = (dist*1.7)**3
            if(dist > 1):
                dist = 1
            reward = 1-dist
            fitness = (fitness+reward)/2
            fitnessPerTrial.append(fitness)
            print("Fitness: {}".format(fitness))     
            
            #######################################
            # TRIAL: TURN LEFT
            #######################################
            self.emitterData = str(genotype)
            
            INITIAL_TRANS = [0.007, 0, 0.35]
            self.trans_field.setSFVec3f(INITIAL_TRANS)
            INITIAL_ROT = [-0.5, 0.5, 0.5, 2.09]
            self.rot_field.setSFRotation(INITIAL_ROT)
            self.robot_node.resetPhysics()
            
            self.spot_light_node.getField('on').setSFBool(False)
            
            self.run_seconds(self.time_experiment)
            
            fitness = self.receivedFitness
            
            [tx,ty,tz] = self.robot_node.getPosition()
            dist = np.sqrt((-0.36-tx)**2+(-0.155-tz)**2)
            dist = (dist*1.7)**3
            if(dist > 1):
                dist = 1
            reward = 1-dist
            
            fitness = (fitness+reward)/2
            print("Fitness: {}".format(fitness))
            fitnessPerTrial.append(fitness)
            
            currentInteraction += 1
            
        print(fitnessPerTrial)    
        
        fitness = np.mean(fitnessPerTrial)
        current = (generation,genotype,fitness)
        self.genotypes.append(current)  
        
        return fitness

    def run_optimization(self):
        # Wait until the number of weights is updated
        while(self.num_weights == 0):
            self.handle_receiver()
            self.createRandomPopulation()
        
        print(">>>Starting Evolution using GA optimization ...\n")
        
        # For each Generation
        for generation in range(self.num_generations):
            print("Generation: {}".format(generation))
            current_population = []   
            # Select each Genotype or Individual
            for population in range(self.num_population):
                genotype = self.population[population]
                # Evaluate
                fitness = self.evaluate_genotype(genotype,generation)
                # Save its fitness value
                current_population.append((genotype,float(fitness)))
                
            # After checking the fitness value of all individuals
            best = ga.getBestGenotype(current_population)
            average = ga.getAverageGenotype(current_population)
            np.save("Best.npy", best[0])
            
            # Update both plots
            self.plot_fitness(generation, best[1], average)
            self.update_plot(generation, best[1], average)
            
            # Generate the new population using genetic operators
            if (generation < self.num_generations - 1):
                self.population = ga.population_reproduce(current_population,self.num_elite)
        
        # Save the final plot
        self.save_plot()
        print("GA optimization terminated.\n")   

    def draw_scaled_line(self, generation, y1, y2): 
        XSCALE = int(self.width/self.num_generations)
        YSCALE = 100
        self.display.drawLine((generation-1)*XSCALE, self.height-int(y1*YSCALE), generation*XSCALE, self.height-int(y2*YSCALE))
    
    def plot_fitness(self, generation, best_fitness, average_fitness):
        if (generation > 0):
            self.display.setColor(0xff0000)  # red
            self.draw_scaled_line(generation, self.prev_best_fitness, best_fitness)
    
            self.display.setColor(0x00ff00)  # green
            self.draw_scaled_line(generation, self.prev_average_fitness, average_fitness)
    
        self.prev_best_fitness = best_fitness
        self.prev_average_fitness = average_fitness

    def run_demo(self):
        # Read File
        genotype = np.load("Best.npy")
        
        # Turn Left
        self.emitterData = str(genotype) 
        
        INITIAL_TRANS = [0.007, 0, 0.35]
        self.trans_field.setSFVec3f(INITIAL_TRANS)
        INITIAL_ROT = [-0.5, 0.5, 0.5, 2.09]
        self.rot_field.setSFRotation(INITIAL_ROT)
        self.robot_node.resetPhysics()
        
        self.spot_light_node.getField('on').setSFBool(False)
        
        self.run_seconds(self.time_experiment) 
        
        fitness = self.receivedFitness
        print("Fitness without reward or penalty: {}".format(fitness))
        
        # Turn Right
        self.emitterData = str(genotype) 
        
        INITIAL_TRANS = [0.007, 0, 0.35]
        self.trans_field.setSFVec3f(INITIAL_TRANS)
        INITIAL_ROT = [-0.5, 0.5, 0.5, 2.09]
        self.rot_field.setSFRotation(INITIAL_ROT)
        self.robot_node.resetPhysics()
        
        self.spot_light_node.getField('on').setSFBool(True)
        
        self.run_seconds(self.time_experiment)  
        
        fitness = self.receivedFitness
        print("Fitness without reward or penalty: {}".format(fitness))    

if __name__ == "__main__":
    gaModel = SupervisorGA()
    keyboard = Keyboard()
    keyboard.enable(50)
    
    print("***************************************************************************************************")
    print("To start the simulation please click anywhere in the SIMULATION WINDOW(3D Window) and press either:")
    print("(S|s)to Search for New Best Individual OR (R|r) to Run Best Individual")
    print("***************************************************************************************************")
    
    while gaModel.supervisor.step(gaModel.time_step) != -1:
        resp = keyboard.getKey()
        if(resp == 83 or resp == 65619):
            gaModel.run_optimization()
            print("(S|s)to Search for New Best Individual OR (R|r) to Run Best Individual")
        elif(resp == 82 or resp == 65619):
            gaModel.run_demo()
            print("(S|s)to Search for New Best Individual OR (R|r) to Run Best Individual")