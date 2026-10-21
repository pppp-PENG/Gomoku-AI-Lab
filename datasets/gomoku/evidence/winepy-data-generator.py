import multiprocessing
import numpy as np
import json
import os
import time
import random
from datetime import datetime
from ai import AI, Pos

def coord_to_base15(x, y):
    """Convert (x,y) coordinates to a base-15 number"""
    return y * 15 + x

def base15_to_coord(num):
    """Convert a base-15 number back to (x,y) coordinates"""
    return num % 15, num // 15

def generate_game_data(game_id, max_moves=225):
    """Generate training data from a single game"""
    # Initialize AI
    ai = AI()
    ai.set_size(15)
    
    # Lists to store moves and board states
    moves = []
    board_states = []
    current_player = 1  # 1 for black (first player), -1 for white
    
    # Run a complete game
    for move_num in range(max_moves):
        # Get current board state before move
        board_state = []
        for i in range(4, 19):  # 15x15 board with 4 offset
            for j in range(4, 19):
                piece = ai.cell[i][j].piece
                if piece == 0:  # White
                    board_state.append(-coord_to_base15(i-4, j-4))
                elif piece == 1:  # Black
                    board_state.append(coord_to_base15(i-4, j-4))
        
        # Store current state
        if move_num > 0:  # Skip first empty board
            board_states.append(board_state.copy())
        
        # Get AI's move
        result = ai.get_best_move()
        x, y = result.x, result.y
        
        # Convert move to base-15 and store
        move_base15 = coord_to_base15(x, y)
        if current_player == -1:  # White's move
            move_base15 = -move_base15
            
        if move_num > 0:  # Skip first move for labels
            moves.append(move_base15)
            
        # Make the move
        ai.put_chess(result)
        
        # Check for win
        if ai.check_win():
            print(f"Game {game_id}: Player {2 - current_player} wins after {move_num+1} moves")
            break
            
        current_player *= -1  # Switch player
    
    # Create training examples: (previous_moves -> next_move)
    training_data = []
    for i in range(len(board_states)):
        # Skip last board state if there's no corresponding move
        if i < len(moves):
            example = {
                "board_state": board_states[i],
                "next_move": moves[i]
            }
            training_data.append(example)
    
    return training_data

def worker(args):
    """Worker function for parallel processing"""
    game_id, output_dir, worker_id = args
    try:
        # Add some randomness to initial moves
        if random.random() < 0.3:
            # Different worker seeds to ensure diverse games
            random.seed(time.time() + worker_id)
            
        data = generate_game_data(game_id)
        
        # Save to individual JSON file
        filename = os.path.join(output_dir, f"game_{game_id}.json")
        with open(filename, 'w') as f:
            json.dump(data, f)
            
        return game_id, len(data)
    except Exception as e:
        print(f"Error in game {game_id}: {str(e)}")
        return game_id, 0

def generate_training_data(num_games=100, output_dir="training_data", num_processes=None):
    """Generate training data from multiple games in parallel"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Use available CPU cores if not specified
    if num_processes is None:
        num_processes = multiprocessing.cpu_count()
    
    print(f"Generating {num_games} games using {num_processes} processes...")
    
    # Prepare arguments for workers
    args_list = [(i, output_dir, i % num_processes) for i in range(num_games)]
    
    # Set up multiprocessing pool
    pool = multiprocessing.Pool(processes=num_processes)
    
    # Start time measurement
    start_time = time.time()
    
    # Process games in parallel
    results = pool.map(worker, args_list)
    
    # Close the pool
    pool.close()
    pool.join()
    
    # Calculate statistics
    total_examples = sum(count for _, count in results)
    elapsed_time = time.time() - start_time
    games_per_second = num_games / elapsed_time
    examples_per_second = total_examples / elapsed_time
    
    print(f"\nGeneration complete!")
    print(f"Generated {num_games} games with {total_examples} training examples")
    print(f"Time elapsed: {elapsed_time:.2f} seconds")
    print(f"Performance: {games_per_second:.2f} games/sec, {examples_per_second:.2f} examples/sec")
    
    # Create metadata file
    metadata = {
        "num_games": num_games,
        "total_examples": total_examples,
        "date_generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "generation_time": elapsed_time
    }
    
    with open(os.path.join(output_dir, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nData saved to {output_dir}/")
    
    return total_examples

def convert_to_numpy_format(input_dir="training_data", output_file="gomoku_dataset"):
    """Convert JSON data to numpy arrays for ML training using full board representation"""
    all_moves = []
    
    # Initialize count of examples
    total_examples = 0
    
    # List all JSON files (excluding metadata)
    files = [f for f in os.listdir(input_dir) if f.endswith('.json') and f != "metadata.json"]
    
    print(f"Converting {len(files)} files to numpy format...")
    
    # First pass: count total examples to pre-allocate arrays
    for filename in files:
        with open(os.path.join(input_dir, filename), 'r') as f:
            game_data = json.load(f)
            total_examples += len(game_data)
    
    # Pre-allocate arrays for efficiency (using full board representation)
    # 15x15=225 positions, each can be -1 (white), 0 (empty), or 1 (black)
    X = np.zeros((total_examples, 15, 15), dtype=np.int8)
    Y = np.zeros(total_examples, dtype=np.int16)
    
    # Second pass: fill the arrays
    example_index = 0
    for filename in files:
        with open(os.path.join(input_dir, filename), 'r') as f:
            game_data = json.load(f)
            
        for example in game_data:
            # Convert sparse representation to full board
            board = np.zeros((15, 15), dtype=np.int8)
            
            for pos in example["board_state"]:
                if pos > 0:  # Black piece
                    x, y_coord = base15_to_coord(pos)
                    board[y_coord][x] = 1
                elif pos < 0:  # White piece
                    x, y_coord = base15_to_coord(-pos)
                    board[y_coord][x] = -1
            
            # Store the board and next move
            X[example_index] = board
            Y[example_index] = example["next_move"]
            
            example_index += 1
    
    # Save arrays
    np.save(f"{output_file}_X.npy", X)
    np.save(f"{output_file}_Y.npy", Y)
    
    print(f"Converted {total_examples} examples to numpy format")
    print(f"X shape: {X.shape}, Y shape: {Y.shape}")
    print(f"Data saved as {output_file}_X.npy and {output_file}_Y.npy")
    
    return total_examples

def convert_to_sparse_format(input_dir="training_data", output_file="gomoku_sparse_dataset"):
    """Convert JSON data to sparse format (Python pickle) to preserve original structure"""
    import pickle
    
    all_states = []
    all_moves = []
    
    # List all JSON files (excluding metadata)
    files = [f for f in os.listdir(input_dir) if f.endswith('.json') and f != "metadata.json"]
    
    print(f"Converting {len(files)} files to sparse format...")
    
    # Process each file
    for filename in files:
        with open(os.path.join(input_dir, filename), 'r') as f:
            game_data = json.load(f)
            
        for example in game_data:
            all_states.append(example["board_state"])
            all_moves.append(example["next_move"])
    
    # Save as pickle (preserves list of lists structure)
    with open(f"{output_file}_X.pkl", 'wb') as f:
        pickle.dump(all_states, f)
    
    with open(f"{output_file}_Y.pkl", 'wb') as f:
        pickle.dump(all_moves, f)
    
    print(f"Converted {len(all_states)} examples to sparse format")
    print(f"Data saved as {output_file}_X.pkl and {output_file}_Y.pkl")
    
    return len(all_states)

if __name__ == "__main__":
    # Example usage:
    num_games = 10  # Adjust based on your needs
    output_dir = "gomoku_training_data"
    
    # Generate data
    generate_training_data(num_games=num_games, output_dir=output_dir)
    
    # Convert to numpy format for ML
    convert_to_numpy_format(input_dir=output_dir, output_file="gomoku_dataset")
    
    # Alternatively, preserve original sparse structure
    convert_to_sparse_format(input_dir=output_dir, output_file="gomoku_sparse")