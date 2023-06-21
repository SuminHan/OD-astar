import pygame
import random
import math

SCREEN_SIZE = (800, 600)
NODE_SIZE = 10
EDGE_WIDTH = 2

NUM_NODES = 20
MIN_X, MAX_X = 50, 750
MIN_Y, MAX_Y = 50, 550

nodes = []
for i in range(NUM_NODES):
    x = random.randint(MIN_X, MAX_X)
    y = random.randint(MIN_Y, MAX_Y)
    nodes.append((x, y))

edges = []
for i in range(NUM_NODES):
    for j in range(i+1, NUM_NODES):
        if random.random() < 0.3:
            edges.append((i, j))

def draw_node(node, color):
    pygame.draw.circle(screen, color, node, NODE_SIZE)

def draw_edge(node1, node2):
    pygame.draw.line(screen, (0, 0, 0), node1, node2, EDGE_WIDTH)

def distance(node1, node2):
    node1 = nodes[node1]
    node2 = nodes[node2]
    return math.sqrt((node1[0]-node2[0])**2 + (node1[1]-node2[1])**2)

def a_star(start, goal):
    frontier = [(distance(start, goal), start)]
    visited = set()
    parent = {}
    g_score = {start: 0}
    f_score = {start: distance(start, goal)}

    while frontier:
        _, current = min(frontier)
        frontier.remove((_, current))

        if current == goal:
            path = []
            while current in parent:
                path.append(current)
                current = parent[current]
            path.append(start)
            path.reverse()
            return path

        visited.add(current)

        for neighbor in neighbors(current):
            if neighbor in visited:
                continue

            tentative_g_score = g_score[current] + distance(current, neighbor)

            if neighbor not in frontier:
                frontier.append((f_score.get(neighbor, float('inf')), neighbor))
            elif tentative_g_score >= g_score.get(neighbor, float('inf')):
                continue

            parent[neighbor] = current
            g_score[neighbor] = tentative_g_score
            f_score[neighbor] = tentative_g_score + distance(neighbor, goal)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        screen.fill((255, 255, 255))
        for node in nodes:
            draw_node(node, (255, 0, 0))
        for edge in edges:
            node1 = nodes[edge[0]]
            node2 = nodes[edge[1]]
            draw_edge(node1, node2)
        for visited_node in visited:
            draw_node(nodes[visited_node], (0, 255, 0))
        for frontier_node in frontier:
            draw_node(nodes[frontier_node[1]], (0, 0, 255))
        pygame.display.flip()

    return None

def neighbors(node):
    n = []
    for edge in edges:
        if edge[0] == node:
            n.append(edge[1])
        elif edge[1] == node:
            n.append(edge[0])
    return n

pygame.init()
screen = pygame.display.set_mode(SCREEN_SIZE)
while True:

    start = random.randint(0, NUM_NODES-1)
    goal = random.randint(0, NUM_NODES-1)
    while goal == start:
        goal = random.randint(0, NUM_NODES-1)

    path = a_star(start, goal)
    print(f"Path from node {start} to node {goal}: {path}")

    pygame.time.delay(1000)
