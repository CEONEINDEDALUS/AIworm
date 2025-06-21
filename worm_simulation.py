import pygame
import numpy as np
import random
import math

# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 800, 600
BG_COLOR = (10, 10, 10)
WORM_COLOR = (0, 255, 0)
FOOD_COLOR = (255, 0, 0)
OBSTACLE_COLOR = (100, 100, 100)
SENSOR_COLOR = (0, 0, 255, 100)  # Semi-transparent blue

# Set up the display
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Worm AI Simulation")
clock = pygame.time.Clock()

class Environment:
    def __init__(self, width, height, num_food=10, num_obstacles=15):
        self.width = width
        self.height = height
        self.food = []
        self.obstacles = []
        
        # Generate food
        for _ in range(num_food):
            self.add_food()
            
        # Generate obstacles
        for _ in range(num_obstacles):
            size = random.randint(10, 40)
            x = random.randint(size, width - size)
            y = random.randint(size, height - size)
            self.obstacles.append((x, y, size))
    
    def add_food(self):
        x = random.randint(10, self.width - 10)
        y = random.randint(10, self.height - 10)
        self.food.append((x, y))
    
    def check_food_collision(self, x, y, radius=10):
        for i, (food_x, food_y) in enumerate(self.food):
            if math.sqrt((x - food_x)**2 + (y - food_y)**2) < radius:
                self.food.pop(i)
                self.add_food()
                return True
        return False
    
    def check_obstacle_collision(self, x, y, radius=5):
        # Check wall collisions
        if x < radius or x > self.width - radius or y < radius or y > self.height - radius:
            return True
        
        # Check obstacle collisions
        for obs_x, obs_y, obs_size in self.obstacles:
            if math.sqrt((x - obs_x)**2 + (y - obs_y)**2) < (radius + obs_size):
                return True
        return False
    
    def draw(self, screen):
        # Draw food
        for food_x, food_y in self.food:
            pygame.draw.circle(screen, FOOD_COLOR, (food_x, food_y), 5)
        
        # Draw obstacles
        for obs_x, obs_y, obs_size in self.obstacles:
            pygame.draw.circle(screen, OBSTACLE_COLOR, (obs_x, obs_y), obs_size)

class Worm:
    def __init__(self, x, y, segment_length=15, num_segments=10):
        self.head_x = x
        self.head_y = y
        self.segment_length = segment_length
        self.num_segments = num_segments
        self.segments = [(x, y)]
        self.angle = 0  # Direction in radians
        self.angular_velocity = 0
        self.speed = 2
        self.sensor_range = 100
        self.sensor_count = 8
        self.alive = True
        self.score = 0
        
        # Initialize segments behind the head
        for i in range(1, num_segments):
            self.segments.append((x - i * segment_length, y))
    
    def sense_environment(self, environment):
        # Create sensor readings
        sensor_readings = []
        
        for i in range(self.sensor_count):
            sensor_angle = self.angle + (2 * math.pi * i / self.sensor_count)
            sensor_x = self.head_x + math.cos(sensor_angle) * self.sensor_range
            sensor_y = self.head_y + math.sin(sensor_angle) * self.sensor_range
            
            # Check for collisions along the sensor line
            collision_distance = self.sensor_range
            
            # Check for wall collisions
            wall_distances = [
                self.head_x / math.cos(sensor_angle) if math.cos(sensor_angle) > 0 else float('inf'),  # Left wall
                (environment.width - self.head_x) / math.cos(sensor_angle) if math.cos(sensor_angle) < 0 else float('inf'),  # Right wall
                self.head_y / math.sin(sensor_angle) if math.sin(sensor_angle) > 0 else float('inf'),  # Top wall
                (environment.height - self.head_y) / math.sin(sensor_angle) if math.sin(sensor_angle) < 0 else float('inf')  # Bottom wall
            ]
            
            for dist in wall_distances:
                if 0 < dist < collision_distance:
                    collision_distance = dist
            
            # Check for obstacle collisions
            for obs_x, obs_y, obs_size in environment.obstacles:
                # Calculate distance to obstacle center
                dx = obs_x - self.head_x
                dy = obs_y - self.head_y
                
                # Project onto sensor line
                proj = dx * math.cos(sensor_angle) + dy * math.sin(sensor_angle)
                
                if 0 < proj < collision_distance:
                    # Calculate perpendicular distance to line
                    perp = abs(dx * math.sin(sensor_angle) - dy * math.cos(sensor_angle))
                    
                    if perp < obs_size:
                        # Calculate actual collision distance
                        collision_distance = proj - math.sqrt(obs_size**2 - perp**2)
            
            # Normalize the reading
            sensor_readings.append(1.0 - min(1.0, collision_distance / self.sensor_range))
            
            # Draw sensor line (for visualization)
            end_x = self.head_x + math.cos(sensor_angle) * collision_distance
            end_y = self.head_y + math.sin(sensor_angle) * collision_distance
            sensor_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            pygame.draw.line(sensor_surface, SENSOR_COLOR, (self.head_x, self.head_y), (end_x, end_y), 2)
            screen.blit(sensor_surface, (0, 0))
        
        # Add food sensor readings
        food_readings = [0] * self.sensor_count
        for food_x, food_y in environment.food:
            dx = food_x - self.head_x
            dy = food_y - self.head_y
            distance = math.sqrt(dx**2 + dy**2)
            
            if distance < self.sensor_range:
                # Calculate angle to food
                food_angle = math.atan2(dy, dx)
                # Find closest sensor
                angle_diff = [(food_angle - (self.angle + 2 * math.pi * i / self.sensor_count)) % (2 * math.pi) for i in range(self.sensor_count)]
                closest_sensor = angle_diff.index(min(angle_diff))
                # Set food reading based on proximity
                food_readings[closest_sensor] = max(food_readings[closest_sensor], 1.0 - distance / self.sensor_range)
        
        # Combine all sensor readings
        return sensor_readings + food_readings
    
    def update(self, action, environment):
        if not self.alive:
            return
        
        # Update angular velocity based on neural network output
        self.angular_velocity = (action[0] - 0.5) * 0.2  # Convert [0,1] to [-0.1,0.1]
        
        # Update angle
        self.angle += self.angular_velocity
        
        # Calculate new head position
        new_x = self.head_x + math.cos(self.angle) * self.speed
        new_y = self.head_y + math.sin(self.angle) * self.speed
        
        # Check for collisions
        if environment.check_obstacle_collision(new_x, new_y):
            self.alive = False
            return
        
        # Check for food
        if environment.check_food_collision(new_x, new_y):
            self.score += 1
        
        # Update head position
        self.head_x = new_x
        self.head_y = new_y
        
        # Update segments (follow the leader)
        self.segments[0] = (self.head_x, self.head_y)
        for i in range(1, len(self.segments)):
            prev_x, prev_y = self.segments[i-1]
            curr_x, curr_y = self.segments[i]
            
            # Calculate direction and distance
            dx = prev_x - curr_x
            dy = prev_y - curr_y
            distance = math.sqrt(dx**2 + dy**2)
            
            # Normalize and scale to segment length
            if distance > 0:
                dx = dx / distance * self.segment_length
                dy = dy / distance * self.segment_length
            
            # Update segment position
            self.segments[i] = (prev_x - dx, prev_y - dy)
    
    def draw(self, screen):
        # Draw segments
        for i, (x, y) in enumerate(self.segments):
            # Gradient color from head to tail
            color_intensity = 255 - int(180 * i / len(self.segments))
            segment_color = (0, color_intensity, 0)
            radius = 8 - 3 * i / len(self.segments)  # Decreasing size from head to tail
            pygame.draw.circle(screen, segment_color, (int(x), int(y)), int(radius))
        
        # Draw connections between segments
        for i in range(len(self.segments) - 1):
            pygame.draw.line(screen, (0, 150, 0), self.segments[i], self.segments[i+1], 3)

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize with random weights
        self.weights1 = np.random.randn(input_size, hidden_size) * 0.1
        self.weights2 = np.random.randn(hidden_size, output_size) * 0.1
        self.bias1 = np.zeros((1, hidden_size))
        self.bias2 = np.zeros((1, output_size))
    
    def forward(self, x):
        # Convert input to numpy array
        x = np.array(x).reshape(1, -1)
        
        # Forward pass
        self.layer1 = self.sigmoid(np.dot(x, self.weights1) + self.bias1)
        self.output = self.sigmoid(np.dot(self.layer1, self.weights2) + self.bias2)
        
        return self.output[0]
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def mutate(self, mutation_rate=0.1, mutation_scale=0.2):
        # Randomly mutate weights and biases
        mask1 = np.random.random(self.weights1.shape) < mutation_rate
        mask2 = np.random.random(self.weights2.shape) < mutation_rate
        mask3 = np.random.random(self.bias1.shape) < mutation_rate
        mask4 = np.random.random(self.bias2.shape) < mutation_rate
        
        self.weights1 += mask1 * np.random.randn(*self.weights1.shape) * mutation_scale
        self.weights2 += mask2 * np.random.randn(*self.weights2.shape) * mutation_scale
        self.bias1 += mask3 * np.random.randn(*self.bias1.shape) * mutation_scale
        self.bias2 += mask4 * np.random.randn(*self.bias2.shape) * mutation_scale

class GeneticAlgorithm:
    def __init__(self, population_size=20, input_size=16, hidden_size=12, output_size=1):
        self.population_size = population_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.generation = 0
        self.best_fitness = 0
        self.best_network = None
        
        # Initialize population
        self.population = [NeuralNetwork(input_size, hidden_size, output_size) for _ in range(population_size)]
    
    def select_parents(self, fitnesses):
        # Tournament selection
        parents = []
        for _ in range(self.population_size):
            tournament_indices = random.sample(range(self.population_size), 3)
            tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
            winner_idx = tournament_indices[tournament_fitnesses.index(max(tournament_fitnesses))]
            parents.append(self.population[winner_idx])
        return parents
    
    def crossover(self, parent1, parent2):
        # Create a new neural network
        child = NeuralNetwork(self.input_size, self.hidden_size, self.output_size)
        
        # Perform crossover for weights and biases
        # For weights1
        mask = np.random.random(parent1.weights1.shape) < 0.5
        child.weights1 = parent1.weights1 * mask + parent2.weights1 * (1 - mask)
        
        # For weights2
        mask = np.random.random(parent1.weights2.shape) < 0.5
        child.weights2 = parent1.weights2 * mask + parent2.weights2 * (1 - mask)
        
        # For bias1
        mask = np.random.random(parent1.bias1.shape) < 0.5
        child.bias1 = parent1.bias1 * mask + parent2.bias1 * (1 - mask)
        
        # For bias2
        mask = np.random.random(parent1.bias2.shape) < 0.5
        child.bias2 = parent1.bias2 * mask + parent2.bias2 * (1 - mask)
        
        return child
    
    def evolve(self, fitnesses):
        # Keep track of best network
        best_idx = fitnesses.index(max(fitnesses))
        if fitnesses[best_idx] > self.best_fitness:
            self.best_fitness = fitnesses[best_idx]
            self.best_network = self.population[best_idx]
        
        # Select parents
        parents = self.select_parents(fitnesses)
        
        # Create new population
        new_population = []
        
        # Elitism: keep the best individual
        new_population.append(self.population[best_idx])
        
        # Create offspring
        for i in range(1, self.population_size):
            parent1 = random.choice(parents)
            parent2 = random.choice(parents)
            child = self.crossover(parent1, parent2)
            child.mutate()
            new_population.append(child)
        
        self.population = new_population
        self.generation += 1
        
        return self.best_network

# Main simulation function
def run_simulation(train_mode=True, display_mode=True, max_generations=100):
    # Initialize environment
    env = Environment(WIDTH, HEIGHT)
    
    # Initialize genetic algorithm
    input_size = 16  # 8 obstacle sensors + 8 food sensors
    ga = GeneticAlgorithm(population_size=20, input_size=input_size, hidden_size=12, output_size=1)
    
    # Training loop
    generation = 0
    best_network = None
    
    while train_mode and generation < max_generations:
        fitnesses = []
        
        # Evaluate each neural network
        for network in ga.population:
            # Reset environment and worm
            env = Environment(WIDTH, HEIGHT)
            worm = Worm(WIDTH // 2, HEIGHT // 2)
            
            # Run simulation for this network
            steps = 0
            max_steps = 1000
            
            while worm.alive and steps < max_steps:
                # Get sensor readings
                sensor_data = worm.sense_environment(env)
                
                # Get action from neural network
                action = network.forward(sensor_data)
                
                # Update worm
                worm.update(action, env)
                
                steps += 1
                
                # Display if needed
                if display_mode and network == ga.best_network:
                    screen.fill(BG_COLOR)
                    env.draw(screen)
                    worm.draw(screen)
                    pygame.display.flip()
                    clock.tick(60)
                    
                    # Handle events
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            pygame.quit()
                            return
            
            # Calculate fitness
            fitness = worm.score * 100 + steps  # Reward for both food and survival time
            fitnesses.append(fitness)
        
        # Evolve population
        best_network = ga.evolve(fitnesses)
        generation += 1
        
        print(f"Generation {generation}, Best Fitness: {ga.best_fitness}")
    
    # Run display mode with best network
    if display_mode:
        # Use best network or last one if training wasn't done
        network = best_network if best_network else ga.population[0]
        
        # Reset environment and worm
        env = Environment(WIDTH, HEIGHT)
        worm = Worm(WIDTH // 2, HEIGHT // 2)
        
        running = True
        while running:
            # Get sensor readings
            sensor_data = worm.sense_environment(env)
            
            # Get action from neural network
            action = network.forward(sensor_data)
            
            # Update worm
            worm.update(action, env)
            
            # Display
            screen.fill(BG_COLOR)
            env.draw(screen)
            worm.draw(screen)
            
            # Display stats
            font = pygame.font.SysFont(None, 24)
            score_text = font.render(f"Score: {worm.score}", True, (255, 255, 255))
            screen.blit(score_text, (10, 10))
            
            pygame.display.flip()
            clock.tick(60)
            
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:  # Reset
                        env = Environment(WIDTH, HEIGHT)
                        worm = Worm(WIDTH // 2, HEIGHT // 2)
    
    pygame.quit()

# Run the simulation
if __name__ == "__main__":
    run_simulation(train_mode=True, display_mode=True, max_generations=50)