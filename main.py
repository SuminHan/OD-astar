import sys
import pygame as pg
from settings import *
import numpy as np
import math
import osm_open


class Clock:
    """The clock class contains game time."""
    def __init__(self):
        # Create clock and time variables.
        self.clock = pg.time.Clock()
        self.delta_time = 0
        self.irl_time = 0
        self.resource_time = 0

class Main:
    def __init__(self):
        # initialize Pygame
        pg.init()
        pg.display.set_caption(GAME_TITLE)  # Program title
        self.screen = pg.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.is_shift_pressed = False
        self.clock = Clock()
        self.map = osm_open.Map()
        self.buildings = self.map.get_buildings()
        self.roads = self.map.get_roads()
        self.nodes = self.map.get_nodes()
        self.lte_cells = self.map.get_grids()

        self.generate_agents()

    def generate_agents(self):
        agent_speed = 1 

        agent_x = []
        agent_y = []
        num_agents = 0
        for x, y in self.nodes:
            if 0 < x < SCREEN_WIDTH-1 and 0 < y < SCREEN_HEIGHT-1:
                agent_x.append(x)
                agent_y.append(y)
                num_agents += 1


        # create array of random agent locations
        # agent_x = np.random.randint(0, SCREEN_WIDTH-1, size=num_agents)
        # agent_y = np.random.randint(0, SCREEN_HEIGHT-1, size=num_agents)

        # create array of random agent directions
        agent_direction = np.random.uniform(0, 2 * math.pi, size=num_agents)

        # create array of random agent speeds
        agent_dx = agent_speed * np.cos(agent_direction)
        agent_dy = agent_speed * np.sin(agent_direction)

        # combine agent properties into 2D array
        self.agents = np.column_stack((agent_x, agent_y, agent_dx, agent_dy))

        # set up block size
        self.num_blocks_x = self.map.grid_width # map_width // block_width
        self.num_blocks_y = self.map.grid_height # map_height // block_height
        self.block_width = SCREEN_WIDTH / self.num_blocks_x
        self.block_height = SCREEN_HEIGHT / self.num_blocks_y
        # create grid of zeros
        self.grid_count = np.zeros((self.num_blocks_y, self.num_blocks_x), dtype=int)

        
    def run(self):
        while True:
            self.events()
            self.update()
            self.draw()

    def update_clock(self):
        """Updates all the time variables in the game."""
        self.clock.delta_time = self.clock.clock.tick(FPS) / 1000
        self.clock.irl_time += self.clock.delta_time
        self.clock.resource_time += RESOURCE_TICK

    # ================================================= User Events ================================================ #
    def events(self):
        """Catches all events here and calls appropriate function."""
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.exit_game()
            elif event.type == pg.KEYDOWN and event.key == pg.K_LSHIFT:
                self.is_shift_pressed = True
            elif event.type == pg.KEYUP and event.key == pg.K_LSHIFT:
                self.is_shift_pressed = False

    def update(self):
        self.update_clock()
        # update agent locations
        self.agents[:, 0] += self.agents[:, 2]
        self.agents[:, 1] += self.agents[:, 3]
        
        self.agents[self.agents[:, 0] < 0][:, 0] = 0
        self.agents[self.agents[:, 0] > SCREEN_WIDTH-1][:, 0] = SCREEN_WIDTH-1
        self.agents[self.agents[:, 1] < 0][:, 1] = 0
        self.agents[self.agents[:, 1] > SCREEN_HEIGHT-1][:, 1] = SCREEN_HEIGHT-1

        # check for agents going out of bounds
        out_of_bounds = (self.agents[:, 0] <= 0) | (self.agents[:, 0] >= SCREEN_WIDTH-1) | \
                        (self.agents[:, 1] <= 0) | (self.agents[:, 1] >= SCREEN_HEIGHT-1)
        self.agents[out_of_bounds, 2] *= -1  # invert the x velocity of out-of-bounds agents
        self.agents[out_of_bounds, 3] *= -1  # invert the y velocity of out-of-bounds agents
        # clear grid
        self.grid_count.fill(0)
            
        # update grid with agent positions
        grid_x = (self.agents[:, 0] / self.block_width).astype(int)
        grid_y = (self.agents[:, 1] / self.block_height).astype(int)
        np.add.at(self.grid_count, (grid_y, grid_x), 1)
            

    def draw_buildings(self):
        for vertices in self.buildings:
            pg.draw.polygon(self.screen, RED, vertices)

        for vertices in self.roads:
            for i in range(len(vertices)-1):
                pg.draw.line(self.screen, LIGHTGREY, vertices[i], vertices[i+1])

        for x, y in self.nodes:
            pg.draw.circle(self.screen, LIGHTGREY, (x, y), 2, 0)
            # pg.draw.line(self.screen, LIGHTGREY, vertices[i], vertices[i+1])

        for vertices in self.lte_cells:
            pg.draw.line(self.screen, LIGHTGREY, vertices[0], vertices[1])
            pg.draw.line(self.screen, LIGHTGREY, vertices[1], vertices[2])
            pg.draw.line(self.screen, LIGHTGREY, vertices[2], vertices[3])
            pg.draw.line(self.screen, LIGHTGREY, vertices[3], vertices[0])

    def draw_roads(self):
        pass
  
    def draw_agents(self):
        for x, y, dx, dy in self.agents:
            pg.draw.circle(self.screen, GREEN, (x, y), 5, 0)

    def draw_grid(self):
        # calculate number of blocks that fit within map size
        font = pg.font.Font(None, int(min(self.block_height, self.block_width)*0.8))
        if True or self.is_shift_pressed:
            for y in range(self.grid_count.shape[0]):
                for x in range(self.grid_count.shape[1]):
                    rect_x = x*self.block_width
                    rect_y = y*self.block_height
                    text = font.render(f"{self.grid_count[y, x]}", True, WHITE)
                    # get dimensions of text surface
                    text_width = text.get_width()
                    text_height = text.get_height()

                    # calculate dimensions of rectangle
                    rect_width = text_width + 20
                    rect_height = text_height + 20
                    self.screen.blit(text, (rect_x+2, rect_y+2))

                    # create rectangle surface
                    # rect_surface = pg.Surface((rect_width, rect_height))
                    # rect_surface.fill(BLUE)

                    # draw text onto rectangle surface
                    #rect_surface.blit(text, ((rect_width - text_width / 2), (rect_height - text_height / 2)))
                    # rect_surface.blit(text, (10, 10))


                    #pg.draw.rect(self.screen, BLUE, (rect_x, rect_y, rect_width, rect_height))

                    # self.screen.blit(rect_surface, (rect_x+2, rect_y+2))

            


    def draw(self):
        """This draws everything onto screen."""
        # Draws first --> last
        self.screen.fill(BGCOLOR)

        self.draw_buildings()
        self.draw_roads()
        self.draw_agents()
        self.draw_grid()





        pg.display.flip()

    def exit_game(self):
        """Quits game."""
        pg.quit()
        sys.exit()

    # def draw_grid(self):
    #     """Draws the grid locked to map. Tile size multiplied to reduce clutter."""
    #     for x in range(-self.player.rect[0], SCREEN_WIDTH, int(TILESIZE/2)):
    #         pg.draw.line(self.screen, LIGHTGREY, (x, 0), (x, SCREEN_HEIGHT))
    #     for y in range(-self.player.rect[1], SCREEN_HEIGHT, int(TILESIZE/2)):
    #         pg.draw.line(self.screen, LIGHTGREY, (0, y), (SCREEN_WIDTH, y))


# Creates and runs game class.
game = Main()
#game.new()
game.run()
