"""
Fast Hebbian Network Visualizer that accurately detects weight changes
"""

import sys
import numpy as np
import time
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget
from PyQt5.QtGui import QPainter, QColor, QPen, QBrush, QFont
from PyQt5.QtCore import Qt, QTimer, QRectF, QPointF

class NetworkVisualizerWidget(QWidget):
    """Widget for visualizing neural network with accurate weight change detection"""
    def __init__(self, layer_sizes, initial_weights=None, parent=None):
        super().__init__(parent)
        self.layer_sizes = layer_sizes
        
        # Neural network state data - initialize with small non-zero values rather than zeros
        self.sensors = np.zeros(layer_sizes[0])
        self.activations = [np.zeros(size) for size in layer_sizes]
        
        # Initialize weights with initial_weights if provided, otherwise small random values
        self.weights = []
        for i in range(len(layer_sizes) - 1):
            if initial_weights is not None and i < len(initial_weights):
                self.weights.append(initial_weights[i].copy())
                print(f"Initialized layer {i} with provided weights: shape {initial_weights[i].shape}")
            else:
                self.weights.append(np.random.uniform(-0.01, 0.01, (layer_sizes[i], layer_sizes[i + 1])))
                print(f"Initialized layer {i} with random small values")
        
        # Track previous weights for comparison - initialize to SAME as current weights
        self.prev_weights = []
        for i in range(len(layer_sizes) - 1):
            self.prev_weights.append(self.weights[i].copy())
        
        # Set flag to indicate we're starting with real weights
        self.has_received_real_weights = initial_weights is not None
        if self.has_received_real_weights:
            print("Visualization starting with preloaded weights - baseline established")
            
        # Thresholds for weight changes
        self.significant_change_threshold = 0.05  # Threshold for significant changes
        self.small_change_threshold = 0.00001     # Threshold for small changes
            
        # Track weight changes (for highlighting, with decay)
        self.weight_changes = []
        self.weight_change_signs = []  # Track the sign of changes: 1=positive, -1=negative, 0=unchanged
        self.weight_change_times = []  # Track when each change was detected (for time-limited highlighting)
        self.max_highlight_duration = 2.0  # Maximum time in seconds to show a change
        
        for i in range(len(layer_sizes) - 1):
            self.weight_changes.append(np.zeros((layer_sizes[i], layer_sizes[i + 1])))
            self.weight_change_signs.append(np.zeros((layer_sizes[i], layer_sizes[i + 1])))
            self.weight_change_times.append(np.zeros((layer_sizes[i], layer_sizes[i + 1])))
            
        self.motor_outputs = np.zeros(layer_sizes[-1])
        self.weight_change = 0.0
        self.linear_velocity = 0.0
        self.angular_velocity = 0.0
        
        # Change detection 
        self.changes_detected = False
        self.highlight_decay_rate = 0.9  # Slower decay (was 0.8)
        
        # Maximum line thickness for weight connections
        self.max_line_thickness = 8.0  # Maximum line width to prevent oversized lines
        
        # Pulse effect settings for small changes
        self.pulse_counter = 0
        self.pulse_period = 30    # Full pulse cycle in frames
        self.pulse_min_alpha = 0.1  # Minimum alpha during pulse
        self.pulse_max_alpha = 0.9  # Maximum alpha during pulse
        
        # Colors
        self.bg_color = QColor(0, 0, 0)  # Black background
        self.input_color = QColor(0, 255, 255)  # Cyan for input
        self.hidden_color = QColor(255, 255, 255)  # White for hidden
        self.output_color = QColor(0, 255, 0)  # Green for output
        self.pos_weight_color = QColor(0, 255, 255)  # Cyan for positive weights
        self.neg_weight_color = QColor(255, 0, 0)  # Red for negative weights
        self.positive_change_color = QColor(0, 255, 0)  # Green for strengthening connections
        self.negative_change_color = QColor(255, 100, 0)  # Orange-red for weakening connections
        self.text_color = QColor(255, 255, 255)  # White text
        
        # Calculate neuron positions
        self.neuron_positions = []
        self.neuron_radius = 5
        self.update_layout()
        
        # Update timer
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.update)
        self.update_timer.start(50)  # 50ms = 20fps
        
        # Set window properties
        self.setMinimumSize(800, 600)
        self.setWindowTitle("Neural Network Visualization")
        
        # Fade-out timer for highlighting
        self.highlight_timer = QTimer(self)
        self.highlight_timer.timeout.connect(self.update_highlights)
        self.highlight_timer.start(50)  # 50ms - faster updates for smoother fade and time tracking

    def update_layout(self):
        """Calculate neuron positions based on current widget size"""
        width, height = self.width(), self.height()
        
        # Margin
        margin_x, margin_y = width * 0.1, height * 0.1
        usable_width = width - 2 * margin_x
        usable_height = height - 2 * margin_y
        
        # Calculate layer x-positions (evenly spaced)
        layer_x = []
        num_layers = len(self.layer_sizes)
        for i in range(num_layers):
            x = margin_x + (i / (num_layers - 1)) * usable_width if num_layers > 1 else margin_x + usable_width / 2
            layer_x.append(x)
            
        # Calculate neuron positions for each layer
        self.neuron_positions = []
        for layer_idx, size in enumerate(self.layer_sizes):
            layer_positions = []
            for i in range(size):
                # Evenly space neurons vertically, centered in usable height
                y = margin_y + ((i + 0.5) / size) * usable_height
                layer_positions.append(QPointF(layer_x[layer_idx], y))
            self.neuron_positions.append(layer_positions)
            
        # Adjust neuron radius based on widget size
        self.neuron_radius = min(width, height) * 0.013
    
    def update_highlights(self):
        """Gradually fade out the weight change highlights and enforce time limits"""
        current_time = time.time()
        
        # Update pulse counter for small changes
        self.pulse_counter = (self.pulse_counter + 1) % self.pulse_period
        pulse_alpha_factor = 0.5 + 0.5 * np.sin(2 * np.pi * self.pulse_counter / self.pulse_period)
        pulse_alpha = self.pulse_min_alpha + (self.pulse_max_alpha - self.pulse_min_alpha) * pulse_alpha_factor
        
        for l in range(len(self.weight_changes)):
            # Find indices where changes have existed for more than max_highlight_duration
            time_exceeded = (current_time - self.weight_change_times[l]) > self.max_highlight_duration
            time_exceeded = time_exceeded & (self.weight_change_times[l] > 0)  # Only where changes actually exist
            
            # Force clear changes that have exceeded time limit
            if np.any(time_exceeded):
                self.weight_changes[l][time_exceeded] = 0
                self.weight_change_signs[l][time_exceeded] = 0
                self.weight_change_times[l][time_exceeded] = 0
            
            # Find indices where magnitude is near zero after decay
            old_magnitude = self.weight_changes[l].copy()
            
            # Reduce highlight values by decay rate
            self.weight_changes[l] *= self.highlight_decay_rate
            
            # Reset signs and times where magnitude becomes negligible
            near_zero = (self.weight_changes[l] < 0.01) & (old_magnitude >= 0.01)
            if np.any(near_zero):
                self.weight_change_signs[l][near_zero] = 0
                self.weight_change_times[l][near_zero] = 0
                
        self.update()  # Trigger repaint
    
    def resizeEvent(self, event):
        """Update layout when widget is resized"""
        self.update_layout()
        super().resizeEvent(event)
    
    def set_initial_weights(self, weights):
        """Set initial weights and mark as real weights"""
        if not self.has_received_real_weights:
            for i, w in enumerate(weights):
                if i < len(self.weights):
                    self.weights[i] = w.copy()
                    self.prev_weights[i] = w.copy()  # Set previous weights to same
            
            self.has_received_real_weights = True
            print("Initial weights set from NPY file")
            
            # Clear weight changes
            for i in range(len(self.weight_changes)):
                self.weight_changes[i].fill(0)
                self.weight_change_signs[i].fill(0)
                self.weight_change_times[i].fill(0)
    
    def detect_weight_changes(self, new_weights):
        """Detect which weights have changed significantly, tracking sign"""
        # If we haven't received real weights yet, set them and don't detect changes
        if not self.has_received_real_weights:
            for i, w in enumerate(new_weights):
                if i < len(self.weights) and i < len(self.prev_weights):
                    self.weights[i] = w.copy()
                    self.prev_weights[i] = w.copy()
            
            self.has_received_real_weights = True
            print("First real weights received and set as baseline")
            return False
        
        any_changes = False
        current_time = time.time()
        
        for l in range(len(new_weights)):
            if l < len(self.prev_weights):
                # Calculate change from previous weights
                if self.weights[l].shape == new_weights[l].shape and self.prev_weights[l].shape == new_weights[l].shape:
                    # Calculate raw difference (preserves sign)
                    raw_diff = new_weights[l] - self.prev_weights[l]
                    # Calculate absolute difference
                    diff = np.abs(raw_diff)
                    
                    # Check for significant changes - very low threshold to catch small changes
                    # Reduced from 0.0001 to 0.00001 to catch very small hebbian changes
                    significant_changes = diff > self.small_change_threshold
                    
                    if np.any(significant_changes):
                        any_changes = True
                        max_diff = np.max(diff)
                        
                        # Only log significant changes to reduce console spam
                        if max_diff > self.significant_change_threshold:
                            print(f"Weight changes detected in layer {l}: max diff = {max_diff:.8f}")
                            
                            # Print some stats about changes
                            num_changes = np.sum(significant_changes)
                            total_weights = diff.size
                            percent = 100 * num_changes / total_weights
                            print(f"Changes in {num_changes}/{total_weights} weights ({percent:.2f}%)")
                            
                            # Log positive vs negative changes
                            pos_changes = np.sum((raw_diff > 0) & significant_changes)
                            neg_changes = np.sum((raw_diff < 0) & significant_changes)
                            print(f"Positive (strengthening): {pos_changes}, Negative (weakening): {neg_changes}")
                        
                        # Create highlight matrix with magnitude, distinguishing significant vs small changes
                        highlight = np.zeros_like(diff)
                        
                        # Mark significant and small changes differently
                        significant_magnitude = significant_changes & (diff > self.significant_change_threshold)
                        small_magnitude = significant_changes & (diff <= self.significant_change_threshold)
                        
                        # Amplify significant changes more than small changes
                        highlight[significant_magnitude] = diff[significant_magnitude] * 20.0  # Amplify significant changes more
                        highlight[small_magnitude] = diff[small_magnitude] * 10.0  # Less amplification for small changes
                        
                        # Create sign matrix (1 for positive, -1 for negative)
                        sign_matrix = np.zeros_like(raw_diff)
                        sign_matrix[significant_changes & (raw_diff > 0)] = 1  # Positive changes
                        sign_matrix[significant_changes & (raw_diff < 0)] = -1  # Negative changes
                        
                        # Add to cumulative change tracking (for highlighting)
                        new_significant = significant_changes & (self.weight_changes[l] < highlight)
                        
                        # Update highlights where new significant changes are detected
                        self.weight_changes[l][new_significant] = highlight[new_significant]
                        
                        # Update signs and timestamps for new significant changes
                        mask = new_significant & ((sign_matrix != 0) | (self.weight_change_signs[l] == 0))
                        if np.any(mask):
                            self.weight_change_signs[l][mask] = sign_matrix[mask]
                            self.weight_change_times[l][mask] = current_time  # Record time of change
                    
                # Update previous weights for next comparison
                self.prev_weights[l] = new_weights[l].copy()
        
        # Update change detection state        
        self.changes_detected = any_changes
        return any_changes
    
    def paintEvent(self, event):
        """Draw the neural network visualization"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Check if there are any active weight changes
        has_active_highlights = False
        for l in range(len(self.weight_changes)):
            if np.max(self.weight_changes[l]) > 0.01:
                has_active_highlights = True
                break
        
        # Calculate pulse alpha for this frame
        pulse_alpha_factor = 0.5 + 0.5 * np.sin(2 * np.pi * self.pulse_counter / self.pulse_period)
        pulse_alpha = self.pulse_min_alpha + (self.pulse_max_alpha - self.pulse_min_alpha) * pulse_alpha_factor
        
        # Draw background
        painter.fillRect(self.rect(), self.bg_color)
        
        # Draw connections
        for l in range(len(self.layer_sizes) - 1):
            for i in range(self.layer_sizes[l]):
                for j in range(self.layer_sizes[l + 1]):
                    # Get weight and change values
                    weight = self.weights[l][i, j]
                    weight_change = self.weight_changes[l][i, j] if l < len(self.weight_changes) else 0
                    change_sign = self.weight_change_signs[l][i, j] if l < len(self.weight_change_signs) else 0
                    
                    # Skip very weak connections (unless changing)
                    if abs(weight) < 0.01 and weight_change < 0.01:
                        continue
                    
                    # Determine connection properties
                    # Base width on weight strength
                    width = abs(weight) * 4.0
                    if width < 0.5:
                        width = 0.5
                    
                    # Different handling for significant vs small changes
                    if weight_change > self.significant_change_threshold:  # Significant change
                        # Boost width for significant changes
                        width += weight_change * 4.0
                        
                        # Determine color based on change sign
                        if change_sign > 0:
                            color = self.positive_change_color
                            opacity = min(0.3 + weight_change * 4.0, 0.95)
                        elif change_sign < 0:
                            color = self.negative_change_color
                            opacity = min(0.3 + weight_change * 4.0, 0.95)
                        else:
                            color = QColor(200, 200, 0)  # Yellow for undefined change
                            opacity = min(0.3 + weight_change * 3.0, 0.9)
                            
                    elif weight_change > self.small_change_threshold:  # Small change - FLASHING
                        # Small highlight for subtle changes
                        width += weight_change * 2.0
                        
                        # Use the original weight color (same as the connection's base color)
                        if weight >= 0:
                            color = self.pos_weight_color  # Cyan for positive weights
                            opacity = pulse_alpha  # Pulsing opacity
                        else:
                            color = self.neg_weight_color  # Red for negative weights
                            opacity = pulse_alpha  # Pulsing opacity
                            
                    else:  # Normal, non-changing connection
                        if weight >= 0:
                            color = self.pos_weight_color
                            opacity = min(0.1 + abs(weight) * 0.9, 0.8)
                        else:
                            color = self.neg_weight_color
                            opacity = min(0.1 + abs(weight) * 0.9, 0.8)
                    
                    # Apply maximum width threshold to prevent over-thick lines
                    width = min(width, self.max_line_thickness)
                    
                    # Set the color with opacity
                    color.setAlphaF(opacity)
                    
                    # Draw the connection line
                    pen = QPen(color, width)
                    painter.setPen(pen)
                    start = self.neuron_positions[l][i]
                    end = self.neuron_positions[l+1][j]
                    painter.drawLine(start, end)
        
        # Draw neurons
        for layer_idx, layer_pos in enumerate(self.neuron_positions):
            for i, pos in enumerate(layer_pos):
                # Determine neuron color and alpha
                if layer_idx == 0:  # Input layer
                    if i < 8:  # Light sensors
                        color = self.input_color
                        alpha = 0.3 + self.sensors[i] * 0.7
                    else:  # Proximity sensors
                        value = self.sensors[i]
                        if value > 0.3:
                            color = QColor(255, 0, 0)  # Red for proximity detection
                            alpha = 0.3 + value * 0.7
                        else:
                            color = self.input_color
                            alpha = 0.3
                elif layer_idx == len(self.layer_sizes) - 1:  # Output layer
                    color = self.output_color
                    value = self.motor_outputs[i]
                    alpha = 0.3 + ((value + 1) / 2) * 0.7
                else:  # Hidden layer
                    color = self.hidden_color
                    value = self.activations[layer_idx][i]
                    alpha = 0.3 + ((value + 1) / 2) * 0.7
                
                # Apply alpha to color
                color_with_alpha = QColor(color)
                color_with_alpha.setAlphaF(alpha)
                
                # Draw the neuron
                painter.setBrush(QBrush(color_with_alpha))
                painter.setPen(Qt.NoPen)
                painter.drawEllipse(pos, self.neuron_radius, self.neuron_radius)
        
        # Draw labels
        painter.setPen(QPen(self.text_color))
        font = QFont("Arial", 10)
        painter.setFont(font)
        
        # Layer labels
        layer_labels = ["Sensors", "Hidden Layers", "Motors"]
        for i, layer_idx in enumerate([0, 1, len(self.layer_sizes)-1]):  # Input, first hidden, output
            if i == 1 and len(self.layer_sizes) > 2:  # Hidden layers label
                # Center over all hidden layers
                start_x = self.neuron_positions[1][0].x()
                end_x = self.neuron_positions[-2][0].x()
                x = (start_x + end_x) / 2
            else:
                x = self.neuron_positions[layer_idx][0].x()
            y = self.rect().top() + 30
            painter.drawText(QRectF(x-50, y, 100, 20), Qt.AlignCenter, layer_labels[i])
        
        # Sensor/motor labels
        if self.layer_sizes[0] == 16 and self.layer_sizes[-1] == 2:
            input_labels = ["LS0", "LS1", "LS2", "LS3", "LS4", "LS5", "LS6", "LS7", 
                           "PS0", "PS1", "PS2", "PS3", "PS4", "PS5", "PS6", "PS7"]
            output_labels = ["Left", "Right"]
            
            # Draw input labels
            font.setPointSize(8)
            painter.setFont(font)
            for i, pos in enumerate(self.neuron_positions[0]):
                painter.drawText(QRectF(pos.x() - 40, pos.y() - 6, 35, 12), Qt.AlignRight, input_labels[i])
            
            # Draw output labels
            for i, pos in enumerate(self.neuron_positions[-1]):
                painter.drawText(QRectF(pos.x() + 10, pos.y() - 6, 40, 12), Qt.AlignLeft, output_labels[i])
        
        # Draw a legend for connection colors
        font.setPointSize(9)
        painter.setFont(font)
        legend_y = self.height() - 80
        
        # Draw colored lines for legend
        pen_width = 3
        line_length = 30
        
        # Normal connections
        painter.setPen(QPen(self.pos_weight_color, pen_width))
        painter.drawLine(10, legend_y, 10 + line_length, legend_y)
        painter.setPen(QPen(self.text_color))
        painter.drawText(10 + line_length + 5, legend_y + 5, "Positive weight")
        
        painter.setPen(QPen(self.neg_weight_color, pen_width))
        painter.drawLine(180, legend_y, 180 + line_length, legend_y)
        painter.setPen(QPen(self.text_color))
        painter.drawText(180 + line_length + 5, legend_y + 5, "Negative weight")
        
        # Changed connections
        painter.setPen(QPen(self.positive_change_color, pen_width))
        painter.drawLine(350, legend_y, 350 + line_length, legend_y)
        painter.setPen(QPen(self.text_color))
        painter.drawText(350 + line_length + 5, legend_y + 5, "Strengthening")
        
        painter.setPen(QPen(self.negative_change_color, pen_width))
        painter.drawLine(500, legend_y, 500 + line_length, legend_y)
        painter.setPen(QPen(self.text_color))
        painter.drawText(500 + line_length + 5, legend_y + 5, "Weakening")
        
        # Add "Flashing" annotation to legend
        painter.setPen(QPen(self.text_color))
        painter.drawText(650, legend_y + 5, "Small changes flash")
        
        # Draw weight change indicator with sign information
        font.setPointSize(10)
        painter.setFont(font)
        
        # Count positive and negative active changes
        pos_count = 0
        neg_count = 0
        small_count = 0
        for l in range(len(self.weight_changes)):
            significant = self.weight_changes[l] > self.significant_change_threshold
            small = (self.weight_changes[l] <= self.significant_change_threshold) & (self.weight_changes[l] > self.small_change_threshold)
            pos_count += np.sum(significant & (self.weight_change_signs[l] > 0))
            neg_count += np.sum(significant & (self.weight_change_signs[l] < 0))
            small_count += np.sum(small)
        
        # Make weight change text color-coded and include sign information
        if self.weight_change > 0.005 or has_active_highlights:
            if pos_count > neg_count:
                change_color = self.positive_change_color  # Green for mostly strengthening
                change_text = f"Weight Change: {self.weight_change:.8f} (Strong: +{pos_count}/-{neg_count}, Subtle: {small_count})"
            elif neg_count > pos_count:
                change_color = self.negative_change_color  # Orange-red for mostly weakening
                change_text = f"Weight Change: {self.weight_change:.8f} (Strong: +{pos_count}/-{neg_count}, Subtle: {small_count})"
            else:
                change_color = QColor(200, 200, 0)  # Yellow for balanced changes
                change_text = f"Weight Change: {self.weight_change:.8f} (Strong: +{pos_count}/-{neg_count}, Subtle: {small_count})"
        elif self.weight_change > 0.0001:
            change_color = QColor(0, 255, 0)  # Green for small changes
            change_text = f"Weight Change: {self.weight_change:.8f} (Minor Changes)"
        else:
            change_color = QColor(255, 255, 255)  # White for minimal/no changes
            change_text = f"Weight Change: {self.weight_change:.8f} (Stable Weights)"
            
        painter.setPen(QPen(change_color))
        
        # Draw weight change text
        painter.drawText(
            QRectF(0, self.height() - 50, self.width(), 20), 
            Qt.AlignCenter, 
            change_text
        )
        
        # Draw velocity info
        painter.setPen(QPen(self.text_color))
        painter.drawText(
            QRectF(0, self.height() - 30, self.width(), 20), 
            Qt.AlignCenter, 
            f"Linear Vel: {self.linear_velocity:.4f} m/s | Angular Vel: {self.angular_velocity:.4f} rad/s"
        )
 
class HebbianVisualizer:
    """Fast PyQt-based Hebbian Network Visualizer with accurate weight change detection"""
    
    def __init__(self, layer_sizes, initial_weights=None, save_dir=None, wheel_base=0.05):
        """Initialize the visualizer with network architecture and optional initial weights"""
        self.layer_sizes = layer_sizes
        self.save_dir = save_dir
        self.wheel_base = wheel_base
        
        # Create QApplication if it doesn't exist
        self.app = QApplication.instance()
        if not self.app:
            self.app = QApplication(sys.argv)
        
        # Create main window and widget with initial weights if provided
        self.window = QMainWindow()
        self.widget = NetworkVisualizerWidget(layer_sizes, initial_weights, self.window)
        self.window.setCentralWidget(self.widget)
        
        # State
        self.running = False
        self.first_update = initial_weights is None  # Only set true if we don't have initial weights
        self.weights_initialized = initial_weights is not None
        
        # Grace period settings - extended and with threshold
        self.in_grace_period = True  # Start with grace period active
        self.grace_period_counter = 0  # Counter for grace period
        self.grace_period_threshold = 40  # Number of iterations before grace period ends (increased)
        
        if initial_weights is not None:
            print(f"HebbianVisualizer initialized with {len(initial_weights)} preloaded weight matrices")
        else:
            print(f"HebbianVisualizer initialized with layers: {layer_sizes} (no preloaded weights)")
    
    def update_data(self, sensors, activations, weights, motor_outputs, weight_change=None, detect_changes=True):
        """Update visualization data with accurate change detection"""
        try:
            if not self.running or not hasattr(self, 'widget'):
                return
            
            # Special handling for first update to set initial weights without detecting changes
            if self.first_update and len(weights) > 0:
                print("Setting initial weights as baseline (no change detection)")
                self.widget.set_initial_weights(weights)
                self.first_update = False
                self.weights_initialized = True
                detect_changes = False  # Don't detect changes on first update
            elif len(weights) > 0 and detect_changes and not self.in_grace_period:
                # Normal update - detect weight changes only if not in grace period
                self.widget.detect_weight_changes(weights)
            elif len(weights) > 0 and not detect_changes:
                # Update weights without detecting changes (during grace period)
                for i, w in enumerate(weights):
                    if i < len(self.widget.weights):
                        self.widget.weights[i] = w.copy()
                        self.widget.prev_weights[i] = w.copy()  # Important: update both current and prev
            
            # Update sensor values
            if len(sensors) == self.layer_sizes[0]:
                self.widget.sensors = np.array(sensors)
            
            # Update activation values
            for i, act in enumerate(activations):
                if i < len(self.widget.activations) and len(act) == self.layer_sizes[i]:
                    self.widget.activations[i] = np.array(act)
            
            # Update weight matrices
            for i, w in enumerate(weights):
                if i < len(self.widget.weights) and w.shape == (self.layer_sizes[i], self.layer_sizes[i+1]):
                    self.widget.weights[i] = w
            
            # Update motor values and calculate velocities
            if len(motor_outputs) == self.layer_sizes[-1]:
                self.widget.motor_outputs = np.array(motor_outputs)
                v_left, v_right = motor_outputs
                self.widget.linear_velocity = (v_left + v_right) / 2
                self.widget.angular_velocity = (v_right - v_left) / self.wheel_base
            
            # Update weight change metric 
            if weight_change is not None:
                self.widget.weight_change = weight_change
                if weight_change > 0.00001 and not self.in_grace_period:  # Only log if not in grace period
                    print(f"Weight change: {weight_change:.8f}")
            
            # Update grace period counter
            if self.in_grace_period:
                self.grace_period_counter += 1
                # Don't automatically end grace period - let the supervisor do it
            
            # Process events to keep UI responsive
            self.app.processEvents()
            
        except Exception as e:
            print(f"Error updating visualization data: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def start_animation(self, interval=100):
        """Show the visualization window"""
        try:
            self.running = True
            self.window.show()
            self.window.activateWindow()
            self.widget.update_timer.setInterval(interval)
            
            # Reset grace period on animation start
            self.in_grace_period = True
            self.grace_period_counter = 0
            
            print("Visualization started in grace period. Close the window to stop.")
            print(f"Grace period: {self.grace_period_threshold} iterations before change detection activates")
        except Exception as e:
            print(f"Error starting visualization: {str(e)}")
    
    def stop_animation(self):
        """Hide the visualization window"""
        try:
            self.running = False
            self.window.hide()
            print("Visualization stopped.")
        except Exception as e:
            print(f"Error stopping visualization: {str(e)}")
    
    def reset_visualization(self, keep_weights=False):
        """Reset the visualization state while optionally keeping weights"""
        try:
            if hasattr(self, 'widget'):
                print("Resetting visualization state")
                
                # Reset weight changes
                for i in range(len(self.widget.weight_changes)):
                    self.widget.weight_changes[i].fill(0)
                    if hasattr(self.widget, 'weight_change_signs'):
                        self.widget.weight_change_signs[i].fill(0)
                    if hasattr(self.widget, 'weight_change_times'):
                        self.widget.weight_change_times[i].fill(0)
                
                # Reset change detection flags
                self.widget.changes_detected = False
                self.first_update = not keep_weights
                
                # Reset grace period
                self.in_grace_period = True
                self.grace_period_counter = 0
                
                if not keep_weights:
                    self.widget.has_received_real_weights = False
                    print("Weights will be reinitialized on next update")
                
                # Force redraw
                self.widget.update()
        except Exception as e:
            print(f"Error resetting visualization: {str(e)}")