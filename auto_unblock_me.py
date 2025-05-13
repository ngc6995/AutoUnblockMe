import time
from enum import Enum
from heapq import heappop, heappush
import cv2
import numpy as np
import win32gui
from mousekey import MouseKey
from PIL import ImageGrab

# Type annotations for various data structures used in the game
#
# The top-left coordinate of the game window (x, y)
Point = tuple[int, int]
# A block represented by its position and size: (x, y, width, height)
Block = tuple[int, int, int, int]
# A list of blocks
BlocksList = list[Block]
# Grid coordinates of a block:
# - First element: numpy array of row indices (e.g., [row1, row2, ...])
# - Second element: numpy array of column indices (e.g., [col1, col2, ...])
Coord = tuple[np.ndarray, np.ndarray]
# List of grid coordinates for multiple positions
CoordsList = list[Coord]
# Valid coordinates list:
# - First element: index of the block (block number)
# - Second element: list of new grid coordinates of that block
ValidCoordsList = list[int, CoordsList]
# Movement command:
# - block_num: index of the block to move
# - direction: direction of movement as a string (e.g., 'up', 'down', 'left', 'right')
# - steps: number of steps to move
Move = tuple[int, str, int]

# Enumeration for the game window dimensions
class GameWindow(Enum):
    PRESET_WIDTH = 516   # Width of the game window in pixels
    PRESET_HEIGHT = 809  # Height of the game window in pixels

# Enumeration for grid-related measurements
class Grid(Enum):
    TOPLEFT_X = 24          # X-coordinate of the top-left corner of the grid
    TOPLEFT_Y = 233         # Y-coordinate of the top-left corner of the grid
    WIDTH = 468             # Width of the entire grid in pixels
    HEIGHT = 468            # Height of the entire grid in pixels
    CELL_SIZE = WIDTH // 6  # Size of each cell in the grid (assuming 6 columns)

# Enumeration for color boundaries used in color detection
class ColorBound(Enum):
    ORANGE_LOWER = [15, 238, 207]  # Lower HSV bound for orange color
    ORANGE_UPPER = [18, 255, 247]  # Upper HSV bound for orange color
    RED_LOWER = [0, 243, 182]      # Lower HSV bound for red color
    RED_UPPER = [3, 255, 225]      # Upper HSV bound for red color

# Enumeration for orientation of blocks
class Orientation(Enum):
    HORIZONTAL = 0  # Horizontal orientation
    VERTICAL = 1    # Vertical orientation

# Enumeration for special block types
class Block(Enum):
    TARGET = -1  # Target block, the block to be moved to the goal

def take_screenshot() -> tuple[np.ndarray, Point]:
    # Define the target window's title and text to identify it
    title = 'ApplicationFrameWindow'
    windowtext='Unblock Me Free'
    # Instantiate a MouseKey object for interacting with windows and elements
    mkey = MouseKey()
    # Retrieve all open windows
    windows = mkey.get_all_windows()
    # Initialize a flag to indicate if the target window is found
    unblock_me_found = False
    # Loop through all windows to find the one matching the specified title and text
    for window in windows:
        if window.title == title and window.windowtext == windowtext:
            unblock_me_found = True
            break  # Exit loop once the target window is found
    if unblock_me_found:
        # Resize the game window to preset dimensions
        win32gui.MoveWindow(
            window.hwnd,
            window.coords_win[0],  # x-coordinate of the window's top-left corner
            window.coords_win[2],  # y-coordinate of the window's top-left corner
            GameWindow.PRESET_WIDTH.value,   # desired width
            GameWindow.PRESET_HEIGHT.value,  # desired height
            True  # repaint the window after moving/resizing
        )
        # Pause execution for 1 second to allow the resize to take effect
        time.sleep(1)
        # Get UI elements associated with the window handle
        element_and_family = mkey.get_elements_from_hwnd(window.hwnd)
        window_element = element_and_family['element']
        # Bring the window to the topmost position for visibility
        mkey.activate_topmost(window_element.hwnd)
        # Define the top-left and bottom-right coordinates of the window for screenshot bounding box
        topleft = (window_element.coords_win[0], window_element.coords_win[2])
        bottomright = (window_element.coords_win[1], window_element.coords_win[3])
        # Capture a screenshot of the specified region
        screenshot = ImageGrab.grab([*topleft, *bottomright])
        # Display the screenshot
        #screenshot.show()
        # Save the screenshot to a file
        screenshot.save('screenshot.png')
        # Convert the PIL Image to a NumPy array for further processing
        screenshot = np.array(screenshot)
        # Convert the image color space from RGB to BGR (OpenCV default)
        screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)
        # Return the processed screenshot and the top-left coordinate of the window
        return screenshot, topleft
    else:
        # Return None if the target window was not found
        return None, None

def crop_grid_image(screenshot: np.ndarray) -> np.ndarray:
    # Extract a sub-image of the screenshot that corresponds to the game grid area
    # Using the top-left coordinates and size defined in the Grid enumeration
    grid_image = screenshot[
        Grid.TOPLEFT_Y.value : Grid.TOPLEFT_Y.value + Grid.HEIGHT.value,  # Rows: from TOPLEFT_Y to TOPLEFT_Y + HEIGHT
        Grid.TOPLEFT_X.value : Grid.TOPLEFT_X.value + Grid.WIDTH.value    # Columns: from TOPLEFT_X to TOPLEFT_X + WIDTH
    ]
    # Return the cropped image containing only the grid portion
    return grid_image

def extract_blocks(grid_image: np.ndarray) -> BlocksList:
    # Apply bilateral filter to remove wood grain of blocks
    filtered = cv2.bilateralFilter(grid_image, 30, 80, 50) 
    # Convert the filtered image from BGR to HSV color space for easier color segmentation
    hsv = cv2.cvtColor(filtered, cv2.COLOR_BGR2HSV)
    # Define the color bounds for detecting orange-colored blocks
    lower_bound = np.array(ColorBound.ORANGE_LOWER.value)
    upper_bound = np.array(ColorBound.ORANGE_UPPER.value)
    # Create a mask that isolates orange regions within the specified bounds
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    # Create a kernel for dilation
    kernel = np.ones((3,3), np.uint8)
    # Dilate the mask to connect nearby regions and fill gaps
    mask = cv2.dilate(mask, kernel, iterations=3)
    # Find contours in the orange mask; each contour corresponds to a potential block
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        blocks_list = []
        # Loop through each contour to compute bounding rectangles
        for contour in contours:
            x, y, width, height = cv2.boundingRect(contour)
            # Append the bounding box coordinates to the list
            blocks_list.append((x, y, width, height))
        # Now detect the red target block
        lower_bound = np.array(ColorBound.RED_LOWER.value)
        upper_bound = np.array(ColorBound.RED_UPPER.value)
        # Create a mask for red regions
        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        # Dilate to enhance red regions
        mask = cv2.dilate(mask, kernel, iterations=3)
        # Find contours for red region
        contour, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if contour:
            # Get the bounding rectangle for the red contour
            x, y, width, height = cv2.boundingRect(contour[0])
            # Append the target (red) block's bounding box to the list
            blocks_list.append((x, y, width, height))
        return blocks_list
    else:
        # Return None if no contours are found
        return None

def show_blocks_detected(image: np.ndarray, blocks_list: BlocksList):
    # Loop through each detected block with index starting from 1
    for i, (x, y, width, height) in enumerate(blocks_list, start=1):
        # Assign a color for the rectangle:
        # Green (0, 255, 0) for all except the target block
        # Blue (255, 0, 0) for the target block
        color = (0, 255, 0) if i < len(blocks_list) else (255, 0, 0)
        # Draw a rectangle around each block on the image
        cv2.rectangle(image, (x, y), (x + width, y + height), color, 2)
    # Display the image with detected blocks in a window titled with the total count
    cv2.imshow(f"Total {len(blocks_list)} Blocks Detected", image)
    # Wait indefinitely until a key is pressed
    cv2.waitKey(0)
    # Close all OpenCV windows
    cv2.destroyAllWindows()

def map_blocks_to_grid(blocks_list: BlocksList) -> np.ndarray:
    # Initialize a 6x6 grid with zeros; each cell represents a position in the game grid
    grid = np.zeros((6, 6), dtype=np.int8)
    # Loop through each detected block with index starting from 1
    for i, (x, y, width, height) in enumerate(blocks_list, start=1):
        # Calculate the row index by dividing y-coordinate by the height of each grid cell
        row = round(y / (Grid.HEIGHT.value / 6))
        # Calculate the column index by dividing x-coordinate by the width of each grid cell
        col = round(x / (Grid.WIDTH.value / 6))
        # Determine if the block is horizontal or vertical based on its dimensions
        if width > height:
            # The block is horizontal
            length = round(width / (Grid.WIDTH.value / 6))
            direction = Orientation.HORIZONTAL
        else:
            # The block is vertical
            length = round(height / (Grid.HEIGHT.value / 6))
            direction = Orientation.VERTICAL
        # Map the block onto the grid based on its orientation
        if direction == Orientation.HORIZONTAL:
            # If this is the target block, assign a special identifier -1
            i = Block.TARGET.value if i == len(blocks_list) else i
            # Fill the grid cells horizontally from the starting column
            grid[row, col:col+length] = i
        else:
            # Fill the grid cells vertically from the starting row
            grid[row:row+length, col] = i
    # Return the grid with all blocks mapped to their positions
    return grid

def print_grid(grid: np.ndarray):
    # Loop through each row in the grid with its index
    for i, row in enumerate(grid):
        # For each cell in the row, create a string representation:
        # '.' if the cell value is 0 (empty),
        # 'X' if it represents the target block,
        # or a letter starting from 'A' for other blocks
        output_string = ' '.join([
            '.' if x == 0 else 
            'X' if x == Block.TARGET.value else 
            chr(65 + (x - 1))  # Map block index to uppercase letter
            for x in row
        ])
        # Print the row; if it's the third row (index 2), add a '=>' at the end(the exit)
        if i == 2:
            print(output_string, '=>')
        else:
            print(output_string)

def is_block_in(coord: Coord, coords_list: CoordsList) -> bool:
    # Unpack the coordinate into separate row and column arrays
    rows, cols = coord    
    # Iterate through each coordinate pair in the list
    for _rows, _cols in coords_list:
        # Check if both the row and column arrays are equal to the current coordinate's arrays
        if np.array_equal(rows, _rows) and np.array_equal(cols, _cols):
            # If a match is found, return True
            return True
    # If no match is found after checking all, return False
    return False

def get_valid_moves(grid: np.ndarray, block_num: int) -> ValidCoordsList:
    # Find the positions (row and column indices) of the current block in the grid
    block_coord = np.where(grid == block_num)
    # Determine if the block is horizontal or vertical
    # A horizontal block has all its cells in the same row
    # A vertical block has all its cells in the same column
    is_horizontal = len(np.unique(block_coord[0])) == 1
    # For a horizontal block, n is the row number; for vertical, it's the column number
    n = block_coord[0][0] if is_horizontal else block_coord[1][0]
    # The length of the current block (number of cells it occupies)
    block_length = len(block_coord[0])
    # Initialize a list to store all valid new positions for the block
    block_new_coords = []
    # Loop through possible new positions in the grid for movement
    for i in range(7 - block_length):
        # Create a copy of the grid to simulate movement
        grid_copy = grid.copy()
        # Generate new coordinates for the block in the potential position
        # For horizontal blocks: same row, moving along columns
        # For vertical blocks: same column, moving along rows
        block_new_coord = (
            np.full(block_length, n) if is_horizontal else np.arange(i, i + block_length),
            np.arange(i, i + block_length) if is_horizontal else np.full(block_length, n)
        )
        # Clear the current block position in the copy
        grid_copy[block_coord] = 0
        # Place the block at the new position in the copy
        grid_copy[block_new_coord] = block_num
        # Check if the move is valid by comparing the sum of the original and new grid
        # If sums are equal, the move doesn't create overlaps or gaps
        if np.sum(grid) == np.sum(grid_copy):
            # Add the new position to the list of valid moves
            block_new_coords.append(block_new_coord)
        else:
            # If an invalid move is detected, check if this move overlaps with previous valid moves
            if is_block_in(block_coord, block_new_coords):
                # If it overlaps, break out of the loop
                break
            else:
                # Otherwise, reset the list to discard invalid moves
                block_new_coords = []
    # Filter out the original position of the block from the valid moves
    block_new_coords = [
        new_coord for new_coord in block_new_coords
        if not np.array_equal(new_coord, block_coord)
    ]
    # If there are valid new positions, return the block number and the list of new coordinates
    if block_new_coords:
        return [block_num, block_new_coords]
    # If no valid moves found, return an empty list
    return []

def get_all_valid_moves(grid: np.ndarray) -> list[ValidCoordsList]:
    # Initialize an empty list to store valid moves for all blocks
    valid_moves = []
    # Loop through all block numbers present in the grid (from 1 to max block number)
    for block_num in range(1, np.max(grid) + 1):
        # Get all valid moves for the current block
        moves = get_valid_moves(grid, block_num)
        # If there are any valid moves for this block, add them to the list
        if moves:
            valid_moves.append(moves)
    # Specifically check for valid moves of the target block
    moves = get_valid_moves(grid, Block.TARGET.value)
    # If any valid moves are found for the target block, add them to the list
    if moves:
        valid_moves.append(moves)
    # Return the complete list of all valid moves for all blocks
    return valid_moves

def create_new_grid(grid: np.ndarray, block_num: int, new_coord: Coord) -> np.ndarray:
    # Find the current position of the specified block in the grid
    block_coord = np.where(grid == block_num)
    # Make a copy of the current grid to simulate the move
    grid_copy = grid.copy()
    # Remove the block from its current position by setting those cells to 0
    grid_copy[block_coord] = 0
    # Place the block at the new specified coordinates
    grid_copy[new_coord] = block_num
    # Return the updated grid with the block moved to the new position
    return grid_copy

def get_children_grid(node: dict) -> list[np.ndarray]:
    # Initialize an empty list to store the resulting child grid states
    children_grid = []
    # Retrieve all valid moves for the current grid state stored in the node
    moves = get_all_valid_moves(node['grid'])
    # Loop through each move, which includes the block number and its possible new coordinates
    for move in moves:
        block_num, new_coords = move
        # For each possible new coordinate of the current move
        for new_coord in new_coords:
            # Generate a new grid state with the block moved to the new coordinate
            new_grid = create_new_grid(node['grid'], block_num, new_coord)
            # Add the new grid state to the list of children
            children_grid.append(new_grid)
    # Return the list of all child grid states resulting from valid moves
    return children_grid

def create_node(grid: np.ndarray, g: int = 0, h: int = 0, parent: dict = None) -> dict:
    # Create a node for the A* algorithm.
    # Returns a dictionary containing node information
    return {
        'grid': grid,
        'g': g,
        'h': h,
        'f': g + h,
        'parent': parent
    }

def is_goal(node: dict) -> bool:
    # Check if the target block (row 2, columns 4 to 5) is in its goal position
    return np.array_equal(
        node['grid'][2, 4:6],  # Extract the target block's current position
        np.full(2, Block.TARGET.value)  # Create an array with the target block's target value
    )

def reconstruct_path(node: dict) -> list[np.ndarray]:
    # Initialize an empty list to store the sequence of grid states forming the path
    path = []
    # Traverse back through the parent nodes until reaching the root (no parent)
    while node['parent']:
        # Add the current node's grid state to the path list
        path.append(node['grid'])
        # Move to the parent node for the next iteration
        node = node['parent']
    # After the loop, add the root node's grid state
    path.append(node['grid'])
    # Reverse the path to get the sequence from start to goal
    return path[::-1]

def heuristic(grid: np.ndarray) -> int:
    # Find the position (row and column indices) of the target block in the grid
    block_coord = np.where(grid == Block.TARGET.value)
    # Calculate the horizontal distance from the target block to the exit
    distance = 4 - block_coord[1][0]
    # Count the number of obstacles (non-zero cells) to the right of the target block in row 2
    obstacles = np.count_nonzero(grid[2, block_coord[1][1]+1:])
    # Return the heuristic estimate: distance plus twice the number of obstacles
    # This estimates the effort to move the target block to the goal considering obstacles
    return distance + 2 * obstacles

def find_path_astar(grid: np.ndarray) -> list[np.ndarray]:
    # Initialize start node
    start_node = create_node(grid)
    start_node_id = hash(tuple(start_node['grid'].flatten()))
    # Initialize open list, open dict and closed set
    open_list = [(start_node['f'], start_node_id)]  # Priority queue
    open_dict = {start_node_id: start_node}  # For quick node lookup
    closed_set = set()  # Explored nodes
    # Searching ...
    while open_list:
        # Get node with lowest f value
        _, current_id = heappop(open_list)
        current_node = open_dict[current_id]
        # Check if we've reached the goal
        if is_goal(current_node):
            return reconstruct_path(current_node)
        closed_set.add(current_id)
        # Explore children
        for child_grid in get_children_grid(current_node):
            # Skip if already explored
            child_id = hash(tuple(child_grid.flatten()))
            if child_id in closed_set:
                continue
            # Calculate new path cost
            tentative_g = current_node['g'] + 1
            # Create or update neighbor
            if child_id not in open_dict:
                child_node = create_node(child_grid, tentative_g, heuristic(child_grid), current_node)
                heappush(open_list, (child_node['f'], child_id))
                open_dict[child_id] = child_node
            elif tentative_g < open_dict[child_id]['g']:
                 # Found a better path to this child
                child_node = open_dict[child_id]
                child_node['g'] = tentative_g
                child_node['f'] = tentative_g + child_node['h']
                child_node['parent'] = current_node
    # No path found
    return None

def get_moves(path: list[np.ndarray]) -> list[Move]:
    # Initialize an empty list to store the moves inferred from the path
    moves = []
    # Loop through each pair of consecutive grid states in the path
    for i in range(len(path) - 1):
        grid1 = path[i]       # Current grid state
        grid2 = path[i + 1]   # Next grid state
        # Find the positions where the grid states differ (i.e., moved block cells)
        rows, cols = np.where(grid1 != grid2)
        # Determine the block number that moved:
        # Check the first differing cell; if it's not empty (not zero), use it
        # Otherwise, use the last differing cell
        block_num = grid1[rows[0], cols[0]] if grid1[rows[0], cols[0]] != 0 else grid1[rows[-1], cols[-1]]
        # Determine if the move was horizontal or vertical by checking the uniqueness of row indices
        is_horizontal = len(np.unique(rows)) == 1
        # Get the current and new coordinates of the moved block
        old_coord = np.where(grid1 == block_num)
        new_coord = np.where(grid2 == block_num)
        # If the move was horizontal
        if is_horizontal:
            # Determine direction based on column change
            direction = 'left' if new_coord[1][0] < old_coord[1][0] else 'right'
            # Calculate the number of steps moved horizontally
            steps = abs(new_coord[1][0] - old_coord[1][0])
        else:
            # Vertical move
            # Determine direction based on row change
            direction = 'up' if new_coord[0][0] < old_coord[0][0] else 'down'
            # Calculate the number of steps moved vertically
            steps = abs(new_coord[0][0] - old_coord[0][0])
        
        # Append the move as a tuple: (block number, direction, number of steps)
        moves.append((block_num, direction, steps))
    # Return the list of all inferred moves
    return moves

def move_block(win_topleft: Point, grid: np.ndarray, block_num: int, directrion: str, steps: int):
    # Initialize mouse control object
    mkey = MouseKey()
    # Find the position (row and column) of the specified block in the grid
    block_coord = np.where(grid == block_num)
    # Determine the length of the block (number of cells it occupies)
    block_length = len(block_coord[0])
    # Get the row and column indices of the block
    block_row = block_coord[0][0]
    block_col = block_coord[1][0]
    # Check if the block is horizontal (all in one row) or vertical (all in one column)
    is_horizontal = len(np.unique(block_coord[0])) == 1
    # Extract the top-left point of the window (for coordinate calculations)
    window_topleft_x, window_topleft_y = win_topleft
    # Calculate the pixel coordinates of the top-left corner of the block in the window
    block_topleft_x = (window_topleft_x + Grid.TOPLEFT_X.value +
                       block_col * 78)  # 78 is likely cell size in pixels
    block_topleft_y = (window_topleft_y + Grid.TOPLEFT_Y.value +
                       block_row * 78)
    # Calculate the center point of the block for mouse movement
    if is_horizontal:
        # For horizontal blocks, center x is offset by half the block's width
        block_middle_x = block_topleft_x + (Grid.CELL_SIZE.value * block_length) // 2
        # Center y is at the middle of the cell height
        block_middle_y = block_topleft_y + Grid.CELL_SIZE.value // 2
    else:
        # For vertical blocks, center x is at the middle of the cell width
        block_middle_x = block_topleft_x + Grid.CELL_SIZE.value // 2
        # Center y is offset by half the block's height
        block_middle_y = block_topleft_y + (Grid.CELL_SIZE.value * block_length) // 2
    # Move the mouse cursor to the center of the block in a natural manner
    mkey.move_to_natural(block_middle_x, block_middle_y)
    # Calculate the target position based on the direction and number of steps
    if directrion == 'left':
        block_x = block_middle_x - Grid.CELL_SIZE.value * steps
        block_y = block_middle_y
    elif directrion == 'right':
        block_x = block_middle_x + Grid.CELL_SIZE.value * steps
        block_y = block_middle_y
    elif directrion == 'up':
        block_x = block_middle_x
        block_y = block_middle_y - Grid.CELL_SIZE.value * steps
    else:  # 'down'
        block_x = block_middle_x
        block_y = block_middle_y + Grid.CELL_SIZE.value * steps
    # Click and hold the mouse button to grab the block
    mkey.left_mouse_down()
    # Drag the mouse to the new position
    mkey.move_to_natural(block_x, block_y)
    # Release the mouse button to complete the move
    mkey.left_mouse_up()

def play_moves(win_topleft: Point, path: list[np.ndarray], moves: list[Move]):
    # Iterate through each grid state in the path
    for i in range(len(path)):
        # Display the current grid state
        print_grid(path[i])
        # Check if there is a corresponding move for this state
        if i < len(moves):
            # Unpack move details: block number, direction, and number of steps
            block_num, direction, steps = moves[i]
            # Create a message describing the move
            message = f"{i+1}. Move block " \
                      f"{'X' if block_num == -1 else chr(65 + (block_num - 1))} " \
                      f"{direction} {steps} " \
                      f"{'steps' if steps > 1 else 'step'}.\n"
            # Print the move message
            print(message)
            # Call the function to perform the move visually
            move_block(win_topleft, path[i], block_num, direction, steps)

if __name__ == '__main__':
    print("Model: unblock_me.py for Solving Unblock Me puzzles.")
    print("Please run solver.py")
