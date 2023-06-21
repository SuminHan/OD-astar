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
        self.building_list = self.map.get_buildings()
        # [
        #     [(50, 150), (100, 100), (150, 100), (150, 200), (50, 200)],
        #     [(300, 150), (400, 150), (400, 100), (450, 100), (450, 150), (550, 150), (550, 200), (400, 200), (400, 250), (300, 250)],
        #     [(700, 150), (800, 100), (900, 150), (900, 250), (800, 300), (700, 250)],
        #     [(1050, 200), (1100, 150), (1200, 150), (1250, 200), (1250, 250), (1200, 300), (1100, 300)],
        #     [(1400, 200), (1500, 100), (1600, 200), (1600, 300), (1500, 400), (1400, 300)],
        #     [(100, 550), (150, 500), (200, 550), (250, 500), (300, 550), (250, 600), (200, 650), (150, 600)],
        #     [(400, 550), (500, 500), (550, 500), (600, 550), (600, 650), (550, 700), (500, 700), (500, 650), (450, 650), (450, 600), (500, 600), (500, 550), (400, 550)],
        #     [(750, 550), (850, 500), (900, 500), (950, 550), (900, 600), (850, 600), (850, 650), (750, 650)],
        #     [(1100, 600), (1150, 550), (1200, 550), (1250, 600), (1250, 700), (1200, 750), (1150, 750), (1150, 700), (1100, 650), (1100, 600)],
        #     [(1400, 600), (1450, 550), (1550, 550), (1600, 600), (1550, 650), (1450, 650)]
        # ]
        agent_speed = 1 
        num_agents = 1000

        # create array of random agent locations
        agent_x = np.random.randint(0, SCREEN_WIDTH-1, size=num_agents)
        agent_y = np.random.randint(0, SCREEN_HEIGHT-1, size=num_agents)

        # create array of random agent directions
        agent_direction = np.random.uniform(0, 2 * math.pi, size=num_agents)

        # create array of random agent speeds
        agent_dx = agent_speed * np.cos(agent_direction)
        agent_dy = agent_speed * np.sin(agent_direction)

        # combine agent properties into 2D array
        self.agents = np.column_stack((agent_x, agent_y, agent_dx, agent_dy))

        # create grid of zeros
        self.grid = np.zeros((SCREEN_HEIGHT // GRID_HEIGHT, SCREEN_WIDTH // GRID_WIDTH), dtype=int)

        
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
        
        # check for agents going out of bounds
        out_of_bounds = (self.agents[:, 0] < 5) | (self.agents[:, 0] >= SCREEN_WIDTH-5) | \
                        (self.agents[:, 1] < 5) | (self.agents[:, 1] >= SCREEN_HEIGHT-5)
        self.agents[out_of_bounds, 2] *= -1  # invert the x velocity of out-of-bounds agents
        self.agents[out_of_bounds, 3] *= -1  # invert the y velocity of out-of-bounds agents
        # clear grid
        self.grid.fill(0)
            
        # update grid with agent positions
        grid_x = (self.agents[:, 0] / GRID_WIDTH).astype(int)
        grid_y = (self.agents[:, 1] / GRID_HEIGHT).astype(int)
        np.add.at(self.grid, (grid_y, grid_x), 1)
            

    def draw_buildings(self):
        for vertices in self.building_list:
            pg.draw.polygon(self.screen, RED, vertices)

    def draw_roads(self):
        pass
  
    def draw_agents(self):
        for x, y, dx, dy in self.agents:
            pg.draw.circle(self.screen, GREEN, (x, y), 5, 0)

    def draw_grid(self):
        
        # set up map size
        map_width = SCREEN_WIDTH
        map_height = SCREEN_HEIGHT

        # set up block size
        block_width = 50
        block_height = 50

        # calculate number of blocks that fit within map size
        num_blocks_x = map_width // block_width
        num_blocks_y = map_height // block_height

        # create 2D array of block coordinates
        block_coords = np.zeros((num_blocks_y, num_blocks_x, 3), dtype=int)
        coords_list = []
        for y in range(num_blocks_y):
            for x in range(num_blocks_x):
                coords_list.append((x * block_width, y * block_height, x+y))

        font = pg.font.Font(None, 36)
        if self.is_shift_pressed:
            for y in range(self.grid.shape[0]):
                for x in range(self.grid.shape[1]):
                    rect_x = x*GRID_WIDTH
                    rect_y = y*GRID_HEIGHT
                    text = font.render(f"{self.grid[y, x]}", True, WHITE)
                    # get dimensions of text surface
                    text_width = text.get_width()
                    text_height = text.get_height()

                    # calculate dimensions of rectangle
                    rect_width = 46 #text_width + 20
                    rect_height = 46 #text_height + 20

                    # create rectangle surface
                    rect_surface = pg.Surface((rect_width, rect_height))
                    rect_surface.fill(BLUE)

                    # draw text onto rectangle surface
                    #rect_surface.blit(text, ((rect_width - text_width / 2), (rect_height - text_height / 2)))
                    rect_surface.blit(text, (10, 10))

                    #pg.draw.rect(self.screen, BLUE, (rect_x, rect_y, rect_width, rect_height))

                    self.screen.blit(rect_surface, (rect_x+2, rect_y+2))

            


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
