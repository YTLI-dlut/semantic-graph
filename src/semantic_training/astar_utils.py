import numpy as np
import heapq

class Node:
    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position
    
    def __lt__(self, other):
        return self.f < other.f

def astar(maze, start, end, max_iterations=5000):
    """
    Returns a list of tuples as a path from the given start to the given end in the given maze
    :param maze: 2D numpy array (0: obstacle/unknown that is unsafe, >0: traversable)
                 Actually, usually 1=obstacle, 255=free, 127=unknown.
                 We need to define what is traversable.
                 Let's assume input maze is processed: 1=Obstacle, 0=Free/Unknown
    :param start: (x, y) tuple
    :param end: (x, y) tuple
    :return: list of (x, y)
    """

    # Create start and end node
    start_node = Node(None, tuple(start))
    start_node.g = start_node.h = start_node.f = 0
    end_node = Node(None, tuple(end))
    end_node.g = end_node.h = end_node.f = 0

    # Initialize both open and closed list
    open_list = []
    closed_list = set()

    # Add the start node
    heapq.heappush(open_list, start_node)

    # Dimensions
    h, w = maze.shape

    # Loop until you find the end
    iterations = 0
    while len(open_list) > 0:
        iterations += 1
        if iterations > max_iterations:
            # print("A* Max Iterations Reached")
            return None # Path too long or stuck

        # Get the current node
        current_node = heapq.heappop(open_list)
        
        if current_node.position in closed_list:
            continue
            
        closed_list.add(current_node.position)

        # Found the goal
        if np.linalg.norm(np.array(current_node.position) - np.array(end_node.position)) < 5.0: # Tolerance
            path = []
            current = current_node
            while current is not None:
                path.append(current.position)
                current = current.parent
            return path[::-1] # Return reversed path

        # Generate children
        children = []
        for new_position in [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]: # Adjacent squares
            
            # Get node position
            node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])

            # Make sure within range
            if node_position[0] > (h - 1) or node_position[0] < 0 or node_position[1] > (w - 1) or node_position[1] < 0:
                continue

            # Make sure walkable terrain
            # Maze convention: 1 = Obstacle.
            if maze[node_position[0]][node_position[1]] == 1:
                continue

            # Create new node
            new_node = Node(current_node, node_position)
            children.append(new_node)

        # Loop through children
        for child in children:
            # Child is on the closed list
            if child.position in closed_list:
                continue

            # Create the f, g, and h values
            # Cost depends on movement (diagonal vs straight)
            dist = np.linalg.norm(np.array(child.position) - np.array(current_node.position))
            child.g = current_node.g + dist
            child.h = np.linalg.norm(np.array(child.position) - np.array(end_node.position))
            child.f = child.g + child.h

            # Child is already in the open list
            # We use a simplified check here for performance, ignoring re-parenting for now if node is in open list with lower cost
            # Ideally we should update.
            # For simplicity in python standard heapq, we just push. Duplicate nodes are handled by closed_set check when popped.
            heapq.heappush(open_list, child)
            
    return None
