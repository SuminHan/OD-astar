import pygame
import random
import math
import heapq
import numpy as np
from settings import *
from osm_open import Map

#SCREEN_SIZE = (800, 600)
NODE_SIZE = 10
EDGE_WIDTH = 2
AGENT_SIZE = 20

MIN_X, MAX_X = 50, 750
MIN_Y, MAX_Y = 50, 550


map = Map()

nodes = map.get_nodes()
roads = map.get_roads()
lte_cells = map.get_grids()
node2idx = {osmid:idx for idx, osmid in enumerate(map.node_gdf['osmid'].tolist())}

grid_count = np.zeros((map.grid_height, map.grid_width), dtype=int)
block_width = SCREEN_WIDTH / map.grid_width
block_height = SCREEN_HEIGHT / map.grid_height

NUM_NODES = len(nodes)

edges_dict = {}

edges = []
for _, item in map.edge_gdf.iterrows():
    edges.append((node2idx[item['u']], node2idx[item['v']]))

    nu, nv = node2idx[item['u']], node2idx[item['v']]
    edges_dict.setdefault(nu, [])
    edges_dict[nu].append(nv)
    edges_dict.setdefault(nv, [])
    edges_dict[nv].append(nu)


def draw_node(node, color):
    pygame.draw.circle(screen, color, node, NODE_SIZE)

def draw_edge(node1, node2):
    pygame.draw.line(screen, (0, 0, 0), node1, node2, EDGE_WIDTH)

def draw_grid(vertices):
    pygame.draw.line(screen, YELLOW, vertices[0], vertices[1])
    pygame.draw.line(screen, YELLOW, vertices[1], vertices[2])
    pygame.draw.line(screen, YELLOW, vertices[2], vertices[3])
    pygame.draw.line(screen, YELLOW, vertices[3], vertices[0])


# Precompute distances between all pairs of nodes
distances = {}
for u in range(NUM_NODES):
    for v in range(NUM_NODES):
        distances[(u, v)] = math.sqrt((nodes[u][0]-nodes[v][0])**2 + (nodes[u][1]-nodes[v][1])**2)

def a_star(start, goal):
    frontier = [(0, start)]
    visited = set()
    parent = {}
    g_score = {start: 0}
    f_score = {start: distances[(start, goal)]}

    while frontier:
        _, current = heapq.heappop(frontier)

        if current == goal:
            path = []
            path.append(goal)
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

            tentative_g_score = g_score[current] + distances[(current, neighbor)]

            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + distances[(neighbor, goal)]
                parent[neighbor] = current
                heapq.heappush(frontier, (f_score[neighbor], neighbor))

    return None


def neighbors(node):
    return edges_dict[node]

pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
speed = 3

while True:

    grid_count.fill(0)
    start = random.randint(0, NUM_NODES-1)
    goal = random.randint(0, NUM_NODES-1)
    while goal == start:
        goal = random.randint(0, NUM_NODES-1)


    path = a_star(start, goal)
    print(f"Path from node {start} to node {goal}: {path}")

    # generate agent at start node
    agent_pos = nodes[start]
    target_pos = nodes[path[0]]

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # move agent towards target node
        dx = target_pos[0] - agent_pos[0]
        dy = target_pos[1] - agent_pos[1]
        dist = math.sqrt(dx**2 + dy**2)
        if dist < speed:
            agent_pos = target_pos
            if len(path) == 1:
                break
            else:
                target_pos = nodes[path.pop(0)]
        else:
            dx = dx * speed / dist
            dy = dy * speed / dist
            agent_pos = (agent_pos[0] + dx, agent_pos[1] + dy)

        grid_x = (agent_pos[0] / block_width).astype(int)
        grid_y = (agent_pos[1] / block_height).astype(int)
        np.add.at(grid_count, (map.grid_height - grid_y - 1, grid_x), 1)

        screen.fill((255, 255, 255))
        for node in nodes:
            draw_node(node, LIGHTGREY)
        draw_node(nodes[start], BLUE)
        draw_node(nodes[goal], GREEN)
        for edge in edges:
            node1 = nodes[edge[0]]
            node2 = nodes[edge[1]]
            draw_edge(node1, node2)
        draw_node(agent_pos, RED)

        for i, vertices in enumerate(lte_cells):
            if grid_count.reshape(-1)[i] > 0:
                draw_grid(vertices)


        pygame.display.flip()

    pygame.time.delay(1000)  # add a delay before generating a new start and goal
