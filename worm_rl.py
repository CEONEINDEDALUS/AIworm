import pygame
import numpy as np
import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import copy

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
pygame.display.set_caption("Worm RL Simulation")
clock = pygame.time.Clock()

# Define experience replay memory
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
    
    def push(self, *args):
        self.memory.append(Experience(*args))
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

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
        self.speed = 2
        self.sensor_range = 100
        self.sensor_count = 8
        self.alive = True
        self.score = 0
        self.steps_since_last_food = 0
        self.max_steps_without_food = 500
        
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
            return 0  # No reward if dead
        
        # Convert action to angular change
        # Action is a value between -1 and 1, representing turning left or right
        angular_change = action * 0.2  # Scale the action to control turn rate
        
        # Update angle
        self.angle += angular_change
        
        # Calculate new head position
        new_x = self.head_x + math.cos(self.angle) * self.speed
        new_y = self.head_y + math.sin(self.angle) * self.speed
        
        # Initialize reward
        reward = -0.01  # Small negative reward for each step to encourage efficiency
        
        # Check for collisions
        if environment.check_obstacle_collision(new_x, new_y):
            self.alive = False
            return -10  # Large negative reward for dying
        
        # Check for food
        if environment.check_food_collision(new_x, new_y):
            self.score += 1
            reward = 10  # Large positive reward for eating food
            self.steps_since_last_food = 0
        else:
            self.steps_since_last_food += 1
            
            # Check if worm is starving
            if self.steps_since_last_food >= self.max_steps_without_food:
                self.alive = False
                return -5  # Negative reward for starving
        
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
        
        return reward
    
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

class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_size)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, state_size, action_size, device="cpu"):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        self.memory = ReplayMemory(10000)
        self.gamma = 0.99  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        
        # Q-Networks
        self.policy_net = DQN(state_size, action_size).to(device)
        self.target_net = DQN(state_size, action_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        
        # Discrete action space: -1 (turn left), 0 (go straight), 1 (turn right)
        self.actions = torch.tensor([-1.0, 0.0, 1.0], device=device)
    
    def act(self, state):
        if random.random() <= self.epsilon:
            return random.choice([-1.0, 0.0, 1.0])
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            action_idx = q_values.argmax().item()
            return self.actions[action_idx].item()
    
    def remember(self, state, action, reward, next_state, done):
        # Convert action to index
        action_idx = torch.where(self.actions == torch.tensor(action, device=self.device))[0].item()
        self.memory.push(state, action_idx, reward, next_state, done)
    
    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        
        experiences = self.memory.sample(batch_size)
        batch = Experience(*zip(*experiences))
        
        # Convert to tensors
        state_batch = torch.FloatTensor(batch.state).to(self.device)
        action_batch = torch.LongTensor(batch.action).unsqueeze(1).to(self.device)
        reward_batch = torch.FloatTensor(batch.reward).unsqueeze(1).to(self.device)
        next_state_batch = torch.FloatTensor(batch.next_state).to(self.device)
        done_batch = torch.FloatTensor(batch.done).unsqueeze(1).to(self.device)
        
        # Compute Q values
        current_q_values = self.policy_net(state_batch).gather(1, action_batch)
        
        # Compute next Q values
        with torch.no_grad():
            next_q_values = self.target_net(next_state_batch).max(1, keepdim=True)[0]
        
        # Compute target Q values
        target_q_values = reward_batch + (1 - done_batch) * self.gamma * next_q_values
        
        # Compute loss
        loss = self.criterion(current_q_values, target_q_values)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def save(self, filename):
        torch.save(self.policy_net.state_dict(), filename)
    
    def load(self, filename):
        self.policy_net.load_state_dict(torch.load(filename))
        self.target_net.load_state_dict(self.policy_net.state_dict())

# Main simulation function
def run_simulation(train_mode=True, display_mode=True, max_episodes=1000):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize environment
    env = Environment(WIDTH, HEIGHT)
    
    # Initialize agent
    state_size = 16  # 8 obstacle sensors + 8 food sensors
    action_size = 3  # Left, straight, right
    agent = DQNAgent(state_size, action_size, device)
    
    # Training loop
    episode = 0
    batch_size = 64
    update_target_every = 10
    save_model_every = 100
    
    while train_mode and episode < max_episodes:
        # Reset environment and worm
        env = Environment(WIDTH, HEIGHT)
        worm = Worm(WIDTH // 2, HEIGHT // 2)
        
        # Get initial state
        state = worm.sense_environment(env)
        
        # Run episode
        total_reward = 0
        steps = 0
        max_steps = 2000
        
        while worm.alive and steps < max_steps:
            # Get action from agent
            action = agent.act(state)
            
            # Update worm
            reward = worm.update(action, env)
            
            # Get new state
            next_state = worm.sense_environment(env)
            
            # Remember experience
            done = not worm.alive or steps >= max_steps - 1
            agent.remember(state, action, reward, next_state, done)
            
            # Update state
            state = next_state
            
            # Accumulate reward
            total_reward += reward
            
            steps += 1
            
            # Train agent
            agent.replay(batch_size)
            
            # Display if needed
            if display_mode and episode % 10 == 0:  # Only display every 10 episodes during training
                screen.fill(BG_COLOR)
                env.draw(screen)
                worm.draw(screen)
                
                # Display stats
                font = pygame.font.SysFont(None, 24)
                episode_text = font.render(f"Episode: {episode}", True, (255, 255, 255))
                score_text = font.render(f"Score: {worm.score}", True, (255, 255, 255))
                reward_text = font.render(f"Reward: {total_reward:.2f}", True, (255, 255, 255))
                epsilon_text = font.render(f"Epsilon: {agent.epsilon:.2f}", True, (255, 255, 255))
                
                screen.blit(episode_text, (10, 10))
                screen.blit(score_text, (10, 40))
                screen.blit(reward_text, (10, 70))
                screen.blit(epsilon_text, (10, 100))
                
                pygame.display.flip()
                clock.tick(60)
                
                # Handle events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return
        
        # Update target network periodically
        if episode % update_target_every == 0:
            agent.update_target_network()
        
        # Save model periodically
        if episode % save_model_every == 0 and episode > 0:
            agent.save(f"worm_dqn_model_ep{episode}.pth")
        
        episode += 1
        print(f"Episode {episode}, Score: {worm.score}, Steps: {steps}, Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}")
    
    # Save final model
    if train_mode:
        agent.save("worm_dqn_model_final.pth")
    
    # Run display mode
    if display_mode:
        # Load best model if available
        try:
            agent.load("worm_dqn_model_final.pth")
            print("Loaded trained model")
        except:
            print("No trained model found, using current model")
        
        # Set epsilon to minimum for exploitation
        agent.epsilon = agent.epsilon_min
        
        # Reset environment and worm
        env = Environment(WIDTH, HEIGHT)
        worm = Worm(WIDTH // 2, HEIGHT // 2)
        
        # Get initial state
        state = worm.sense_environment(env)
        
        running = True
        total_reward = 0
        
        while running:
            # Get action from agent
            action = agent.act(state)
            
            # Update worm
            reward = worm.update(action, env)
            total_reward += reward
            
            # Get new state
            next_state = worm.sense_environment(env)
            state = next_state
            
            # Display
            screen.fill(BG_COLOR)
            env.draw(screen)
            worm.draw(screen)
            
            # Display stats
            font = pygame.font.SysFont(None, 24)
            score_text = font.render(f"Score: {worm.score}", True, (255, 255, 255))
            reward_text = font.render(f"Total Reward: {total_reward:.2f}", True, (255, 255, 255))
            alive_text = font.render(f"Alive: {worm.alive}", True, (255, 255, 255))
            
            screen.blit(score_text, (10, 10))
            screen.blit(reward_text, (10, 40))
            screen.blit(alive_text, (10, 70))
            
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
                        state = worm.sense_environment(env)
                        total_reward = 0
            
            # Check if worm is dead
            if not worm.alive:
                font = pygame.font.SysFont(None, 48)
                dead_text = font.render("Worm is dead! Press R to restart", True, (255, 0, 0))
                text_rect = dead_text.get_rect(center=(WIDTH//2, HEIGHT//2))
                screen.blit(dead_text, text_rect)
                pygame.display.flip()
    
    pygame.quit()

# Run the simulation
if __name__ == "__main__":
    run_simulation(train_mode=True, display_mode=True, max_episodes=500)