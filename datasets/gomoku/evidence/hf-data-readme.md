# Gomoku (Five in a Row) AI Training Dataset

This dataset contains board states and moves from self-played Gomoku games generated using the WinePy AI.

## Dataset Information

- **Number of games**: 875
- **Number of examples**: 26378
- **Board size**: 15×15
- **Game format**: Gomoku (Five in a Row)
- **Train/Test split**: 80%/20% (split by complete games)

## Data Organization

This dataset is organized into train and test sets, with each containing both full board and sequence formats:

```
gomoku_dataset/
├── train/                  # Training set (80% of games)
│   ├── full_board/         # Board state as 15×15 grid
│   └── sequence/           # Sparse representation
├── test/                   # Test set (20% of games)
│   ├── full_board/         # Board state as 15×15 grid
│   └── sequence/           # Sparse representation
├── full/                   # Combined data (convenience)
├── sequence/               # Combined data (convenience)
├── samples/                # Sample board visualizations
```

## Data Formats

This dataset is provided in two formats:

### 1. Full Board Format (in `full_board/` directories)

- `board_states.npy`: Numpy array of shape (N, 15, 15) containing board states
  - Each board state is a 15×15 grid where 0=empty, 1=black, -1=white
- `next_moves_coords.npy`: Numpy array of shape (N, 2) containing (x,y) coordinates for each next move
- `next_moves_players.npy`: Numpy array of shape (N,) containing the player (1=black, -1=white) for each move
- `moves.csv`: CSV file containing all moves with columns x, y, player

### 2. Sequence Format (in `sequence/` directories)

- `board_states.json`/`.pkl`: List of board states in sparse representation
  - Each board state is a list of integers where positive=black pieces, negative=white pieces
  - Numbers encode positions in base-15 format: `y * 15 + x`
- `next_moves.json`/`.pkl`: List of next moves (one per board state)
- `complete_games.json`/`.pkl`: List of complete games as sequences of moves

## Train/Test Split

The dataset is split into training (80%) and test (20%) sets. The split is performed at the **game level**, meaning entire games are assigned to either the training or test set. This approach ensures that the test set evaluates the model's ability to generalize to completely unseen game situations.

Random seed 42 was used for reproducibility of the train/test split.

## Usage

### Loading data in Python

```python
# Loading NumPy arrays (full board format)
import numpy as np

# Training data
X_train = np.load('train/full_board/board_states.npy')
y_train_coords = np.load('train/full_board/next_moves_coords.npy')
y_train_players = np.load('train/full_board/next_moves_players.npy')

# Test data
X_test = np.load('test/full_board/board_states.npy')
y_test_coords = np.load('test/full_board/next_moves_coords.npy')

# Loading sequence format
import json
with open('train/sequence/complete_games.json', 'r') as f:
    train_games = json.load(f)
```

## Data Generation

This dataset was generated using the WinePy AI, a Python implementation of the Wine Gomoku AI engine.
The data was generated through self-play using an alpha-beta search algorithm with pattern recognition.

## License

This dataset is licensed under the MIT License.
