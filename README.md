# Space Invaders Deep Q-Learning Project

## Overview
This project implements a Deep Q-Learning (DQN) agent to play the classic Atari 2600 game Space Invaders using reinforcement learning techniques inspired by DeepMind's original research.

## Project Structure
- `space_invaders_dqn.py`: Main training script for the Deep Q-Learning agent
- `models/`: Directory to save trained model checkpoints
- `videos/`: Directory for recording gameplay videos during training

## Technical Approach

### Deep Q-Network Architecture
The neural network is designed based on DeepMind's original DQN paper, featuring:
- Input: 4 stacked grayscale frames (84x84 pixels)
- 3 Convolutional layers for feature extraction
- 1 Dense layer with 512 neurons
- Output layer predicting Q-values for game actions

### Key Components
- **Experience Replay**: Stores and samples past experiences for training
- **Target Network**: Stabilizes learning by using a separate network for Q-value targets
- **Epsilon-Greedy Exploration**: Balances exploration and exploitation

## Hyperparameters
- **Discount Factor (γ)**: 0.99
- **Initial Exploration Rate (ε)**: 1.0
- **Minimum Exploration Rate**: 0.1
- **Batch Size**: 32
- **Learning Rate**: 0.00025

## Training Process
- Environment: Atari 2600 Space Invaders via Gymnasium
- Preprocessing: Frame stacking, grayscale conversion
- Training Stopping Conditions:
  - Reaching a running reward threshold
  - Maximum episodes
  - Maximum total frames

## Requirements
- Python 3.8+
- TensorFlow
- Gymnasium
- ale-py
- NumPy

## Installation
```bash
pip install gymnasium[atari] tensorflow numpy ale-py
```

## Running the Training
```bash
python space_invaders_dqn.py
```

## Model Saving
- Models are automatically saved during training
- Checkpoints saved at regular intervals
- Final model saved when training completes

## Performance Tracking
- Episode rewards tracked
- Running reward calculated
- Exploration rate decays over time

## Challenges and Considerations
- Computationally expensive training
- Complex game environment
- Balancing exploration and exploitation

## Future Improvements
- Hyperparameter tuning
- Advanced exploration strategies
- Extended training duration

## References
- DeepMind's DQN Nature Paper (2015)
- Gymnasium Atari Environments Documentation

## Acknowledgements
Developed as part of a Deep Learning course assignment.

## Learning Process & Algorithm Details

### Deep Q-Learning Algorithm
This project implements Deep Q-Learning (DQN), a reinforcement learning algorithm that combines Q-learning with deep neural networks. The algorithm:

1. **Observes Environment**: Processes 4 stacked frames of Space Invaders gameplay
2. **Takes Actions**: Selects actions using an epsilon-greedy strategy
3. **Stores Experiences**: Saves transitions (state, action, reward, next_state) in replay memory
4. **Learns from Past**: Randomly samples experiences to update the Q-network

### Key Algorithm Components

#### Double Network Architecture
- **Action Network**: Makes current action decisions
- **Target Network**: Provides stable Q-value targets
- Target network updated every 10,000 frames to reduce training instability

#### Experience Replay Buffer
- Stores last 100,000 game transitions
- Randomly samples batches of 32 experiences
- Breaks correlations between consecutive samples
- Enables learning from rare experiences multiple times

#### Epsilon-Greedy Exploration
- Starts at 100% random actions (ε = 1.0)
- Linearly decays to 10% random actions
- Decay occurs over 1,000,000 frames
- Balances exploration of new strategies with exploitation of learned knowledge

### Training Process Details

#### State Preprocessing
1. Convert frames to grayscale
2. Resize to 84x84 pixels
3. Stack 4 consecutive frames
4. Normalize pixel values

#### Learning Updates
- Occurs every 4 frames
- Uses Huber loss for gradient stability
- Clips gradients to prevent explosive updates
- Learning rate: 0.00025 with Adam optimizer

#### Reward Structure
- Immediate rewards from game score
- Discounted future rewards (γ = 0.99)
- No reward shaping or modifications

### Performance Metrics

#### Training Monitoring
- Episode rewards tracked and averaged
- Running reward over last 100 episodes
- Exploration rate decay progress
- Model saved at performance milestones

#### Success Criteria
- Target running reward: 800 points
- Maximum training frames: 10 million
- Early stopping if target achieved