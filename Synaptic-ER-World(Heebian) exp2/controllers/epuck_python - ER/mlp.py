import numpy as np

# Synaptic Hebbian Evaluation
# Copyright (c) 2024 Hamze Hammami,  (ORCID: 0009-0004-5754-5842)


# License:
# 1. Academic Use Only: This code is for evaluation and academic purposes only. 
#    Redistribution, modification, or commercial use is prohibited.
# 2. No Modifications: Modifications or derivative works are not allowed until 
#    the related research paper is officially published.
# 3. Attribution: Proper credit must be given if this code is referenced in 
#    academic or research contexts.
# 4. Liability: This code is provided "as is," without guarantees or warranties. 
#    The author is not responsible for any issues arising from its use.

# For permissions or inquiries, contact: hh2095@hw.ac.uk

def sigmoid(x):
    """tangent activation function"""
    return np.tanh(x)

class MLP:
    def __init__(self, layer_sizes, hebbian_rate=0.1):
        """Initialize network with hebbian learning"""
        self.layer_sizes = layer_sizes
        self.hebbian_rate = hebbian_rate
        self.plasticity_enabled = False
        
        self.layers = [
            np.ones(size + (1 if i == 0 else 0))
            for i, size in enumerate(layer_sizes)
        ]
        
        self.weights = [
            np.random.uniform(-0.5, 0.5, (self.layers[i].size, layer_sizes[i+1]))
            for i in range(len(layer_sizes)-1)
        ]
        
        self.initial_weights = [w.copy() for w in self.weights]
        
        self.hebbian_traces = [np.zeros_like(w) for w in self.weights]
        self.weight_changes = []
        self.total_change = 0.0
        
        self.trace_decay = 0.95
        self.trace_update = 0.05
        self.max_weight = 2.0
        
    def propagate_forward(self, inputs):
        """Forward pass through the network"""
        self.layers[0][:-1] = inputs
        
        for i in range(len(self.weights)):
            self.weights[i] = np.clip(self.weights[i], -self.max_weight, self.max_weight)
            self.layers[i+1] = sigmoid(np.dot(self.layers[i], self.weights[i]))
            
        return self.layers[-1]
    
    def update_synaptic_weights(self, fitness):
        """Apply hebbian based on current activity"""
        if not self.plasticity_enabled:
            return 0.0
        
        total_change = 0.0
        effective_rate = self.hebbian_rate * max(0.2, fitness) 
        
        for i in range(len(self.weights)):
            pre = self.layers[i].reshape(-1, 1)
            post = self.layers[i+1].reshape(1, -1)
            
            hebbian_update = np.outer(pre, post)
            
            self.hebbian_traces[i] = (self.trace_decay * self.hebbian_traces[i] + 
                                    self.trace_update * hebbian_update)
            
            weight_change = effective_rate * self.hebbian_traces[i]
            
            self.weights[i] += weight_change
            
            self.weights[i] = np.clip(self.weights[i], -self.max_weight, self.max_weight)
            
            change_magnitude = np.mean(np.abs(weight_change))
            total_change += change_magnitude
        
        self.weight_changes.append(total_change)
        return total_change
    
    def set_plasticity(self, enabled):
        """Enable or disable plasticity / hebbian learning"""
        if self.plasticity_enabled != enabled:
            self.plasticity_enabled = enabled
            if not enabled:
                self.weights = [w.copy() for w in self.initial_weights]
                self.hebbian_traces = [np.zeros_like(w) for w in self.weights]
                self.weight_changes = []
    
    def set_genetic_weights(self, weights):
        """Set and store initial weights"""
        scaled_weights = []
        for w in weights:
            w_scaled = np.clip(w, -self.max_weight, self.max_weight)
            scaled_weights.append(w_scaled)
        
        self.initial_weights = [w.copy() for w in scaled_weights]
        self.weights = [w.copy() for w in scaled_weights]
        self.hebbian_traces = [np.zeros_like(w) for w in self.weights]
        self.weight_changes = []
    
    def get_weight_changes(self):
        """track weight changes"""
        if not self.weight_changes:
            return 0.0
        return np.mean(self.weight_changes[-20:]) if len(self.weight_changes) > 20 else np.mean(self.weight_changes)