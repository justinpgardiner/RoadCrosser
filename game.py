import csv
import time
import pygame
from game_pieces import *
from neat import *
import neat.nn as neatnn
import torch.nn as nn
import torch.optim as optim
import torch

# File that contains all code relevant to running the game for either human, NEAT, or ANN use


# Static method needed to do the repetitive task of drawing the background of the game (grass, roads, info bar)
def background(screen, title):
    pygame.display.set_caption(title)
    screen.fill((255, 255, 255))
    pygame.draw.rect(screen, (0, 50, 0), pygame.Rect(0, 0, 650, 50))
    pygame.draw.rect(screen, (0, 0, 0), pygame.Rect(0, 50, 650, 250))
    pygame.draw.rect(screen, (0, 50, 0), pygame.Rect(0, 300, 650, 100))
    pygame.draw.rect(screen, (0, 0, 0), pygame.Rect(0, 400, 650, 250))
    pygame.draw.rect(screen, (0, 50, 0), pygame.Rect(0, 650, 650, 50))
    pygame.draw.rect(screen, (100, 100, 100), pygame.Rect(0, 700, 650, 50))


# Basic Game class, parent to AI classes as well as player controlled implementation
class Game:
    def __init__(self):
        self.player = Player()
        self.level = 0
        self.score = 0
        self.count = 199
        self.color_options = ((255, 255, 0), (0, 255, 255), (0, 255, 0), (255, 0, 255))
        self.stages = [([Lane(True, y, y % 100 == 0, i / 10, color=self.color_options[i % len(self.color_options)]) for y in range(400, 650, 50)],
                        [Lane(True, y, y % 100 == 0, i / 10, color=self.color_options[i % len(self.color_options)]) for y in range(50, 300, 50)]) for i in range(10)]

    def loop(self, screen):
        if self.level == 0:
            self.player.boundaries[3] = 650
        else:
            self.player.boundaries[3] = 700
        # Delay in beginning allows cars to populate lanes before the character is allowed to move,
        # preventing the initial time when there are no obstacles
        if self.count > 50:
            font = pygame.font.Font(None, 30)
            screen.blit(font.render(str(self.count // 50), True, (200, 200, 200)),
                        ((650 - font.render(str(self.count // 50), True, (200, 200, 200)).get_width()) / 2, 350))
        else:
            self.count = 0
            self.player.enabled = True
            # Addresses possible bug if player somehow moves outside of window
            # Also is applicable if logs are implemented in the future
            if self.player.x <= -50 or self.player.x >= 700:
                self.player.is_dead = True

            # Checking for collisions, only way to die in this implementation
            for stage in self.stages:
                draw = self.stages.index(stage) == self.level
                for road in stage:
                    for lane in road:
                        for car in lane.props:
                            if car.collides(self.player) and draw:
                                self.player.is_dead = True

            # Lives are implemented if one wanted to expand on the game in the future for humans
            if self.player.lives == 0:
                font = pygame.font.Font(None, 30)
                screen.blit(font.render("GAME OVER", True, (200, 200, 200)), ((650 - font.render("GAME OVER", True, (200, 200, 200)).get_width()) / 2, 310))
                screen.blit(font.render(
                    "SCORE: " + str(self.score), True,
                    (200, 200, 200)), ((650 - font.render("SCORE: " + str(self.score), True, (200, 200, 200)).get_width()) / 2, 370))
                pygame.display.update()
                time.sleep(2)
                return None

        # Update game pieces
        for stage in self.stages:
            draw = self.stages.index(stage) == self.level
            for road in stage:
                for lane in road:
                    lane.update(pygame, screen, draw=draw)
        self.player.update(pygame, screen)

        self.level = self.player.level

        # Display necessary information
        font = pygame.font.Font(None, 40)
        self.score = int((self.level * 10) + ((self.level + 1) * 650 - self.player.y) // 5)
        self.score = self.score - (self.score % 10)
        screen.blit(font.render("SCORE: " + str(self.score), True, (0, 0, 0)), (400,710))
        screen.blit(
            font.render("LVL: " + str(self.level + 1), True, (0, 0, 0)), (10, 710))

        self.count -= 1

        # Calculates the closest cars to the player
        closest = self.player.closest(self.stages[self.level][0], self.stages[self.level][1], 3)
        for car in closest:
            pygame.draw.line(screen, (255, 0, 0), (car[1][0] + car[1][2] // 2, car[1][1] + 25), (self.player.x + 25, self.player.y + 25))
        # Information returned, similar to the inputs the AI would receive
        return self.player.x, self.player.y, self.player.mid_move, closest


# Implementation of the NEAT algorithm for the game, inherits properties from Game class as is similar
class Neat(Game):
    def __init__(self, num_gen, size):
        super().__init__()
        # A new object of this class is created for every generation, so passing this value as a parameter allows the
        # program to keep track of how far into training the algorithm is, as well as providing the option for fitness
        # functions that rely on average values
        self.num_gen = num_gen
        # In order to decrease training time, multiple players are trained in each generation. The size of this
        # generation can be controlled by the "size" parameter in the constructor. In each list created within the
        # self.players list, there is a player object, a list holding values relating to the last 100 y-values of
        # the player at index 1 (this helps with determining when a player is no longer moving productively), and the
        # number of frames the player has stagnated in y-value according to the list in index 1 at index 2. These are
        # actively being used in the training process. The list at index 3 holds the last 13 outputs from the player and
        # the value at index 4 holds the number of frames of stagnation in decision-making according to this list in
        # index 3. This can be used to understand when a player is spamming an output, such as left, right or down, even
        # when they cannot go that way, and penalize them accordingly. This could be used to impact the fitness function
        # to improve performance, however it is not being used in the final implementation as it complicates the fitness
        # function too much.
        self.players = [[Player(), [-1 for _ in range(100)], 0, [-1 for _ in range(13)], 0] for _ in range(size)]

    def loop(self, screen):
        if self.count > 50:
            font = pygame.font.Font(None, 30)
            screen.blit(font.render(str(self.count // 50), True, (200, 200, 200)),
                        ((650 - font.render(str(self.count // 50), True, (200, 200, 200)).get_width()) / 2, 350))
        else:
            self.count = 0
            # Looping through all players to keep every member of the population updated
            for player in self.players:
                # List representing a player is assigned None when they are dead; here we are checking that the player
                # is still alive
                if player is not None:
                    # We only need the Player object to update the player
                    player = player[0]
                    player.enabled = True
                    # Addresses possible bug if player somehow moves outside of window
                    # Also is applicable if logs are implemented in the future
                    if player.x <= -50 or player.x >= 700:
                        player.is_dead = True
                    for stage in self.stages:
                        # Confirming that this player is in the level visible to the viewer
                        # We are only displaying the highest level of a player, so this prevents confusion of the
                        # program drawing "shadow players" from other levels
                        draw = self.stages.index(stage) == player.level
                        for road in stage:
                            for lane in road:
                                for car in lane.props:
                                    if car.collides(player) and draw:
                                        player.is_dead = True
            # Setting the level to be displayed based off of the max level achieved
            self.level = max([player[0].level if player is not None else -1 for player in self.players])

        # Update game pieces
        for stage in self.stages:
            draw = self.stages.index(stage) == self.level
            for road in stage:
                for lane in road:
                    lane.update(pygame, screen, draw=draw)

        # Outputs must be produced as a list, as there are multiple players in each generation
        player_outputs = []
        for player in self.players:
            # Confirming player is not previously dead
            if player is not None:
                # Only displaying players that didn't die in this frame
                if not player[0].is_dead:
                    player[0].update(pygame, screen, draw=player[0].level == self.level)
                # Calculating the closest cars to the player at the instant of the frame
                # This only considers cars present on the level of the player
                closest = player[0].closest(self.stages[player[0].level][0], self.stages[player[0].level][1], 3)
                # Visualizing the closest cars with lines connecting the car to the player
                for car in closest:
                    if player[0].level == self.level:
                        pygame.draw.line(screen, (255, 0, 0), (car[1][0] + car[1][2] // 2, car[1][1] + 25),
                                            (player[0].x + 25, player[0].y + 25))

                # Outputs, or result of the frame, are appended to greater outputs list
                # These are the inputs to the neural network
                # Outputs are:
                # Player x-value and adjusted y-value based on the level it is on
                # The x-value, adjusted y-value based on level, and length in pixels of each car the closest method
                # deemed closest
                player_outputs.append((float(player[0].x), float(650 - player[0].y) + 660 * player[0].level,
                                       float(closest[0][1][0]), float(closest[0][1][1]) + 660 * player[0].level, float(closest[0][1][2] * 50), float(closest[0][1][3]),
                                       float(closest[1][1][0]), float(closest[1][1][1]) + 660 * player[0].level, float(closest[1][1][2]* 50), float(closest[1][1][3]),
                                       float(closest[2][1][0]), float(closest[2][1][1]) + 660 * player[0].level, float(closest[2][1][2] * 50), float(closest[2][1][3])))
            else:
                # Place-holder output values are given to dead players
                player_outputs.append((-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1))

        self.count -= 1

        # Sets maximum score achieved by a player
        self.score = max([player[0].score if player is not None else -1 for player in self.players])

        # pygame.display.update()
        return player_outputs

    def train(self, genomes, config):
        clock = pygame.time.Clock()
        screen_width = 650
        screen_height = 750
        screen = pygame.display.set_mode((screen_width, screen_height))
        background(screen, "NEAT")
        nets = []
        for genome in genomes:
            # Need networks in a list as everything else is in a list regarding the players; works in parallel with
            # self.players list
            nets.append(neatnn.FeedForwardNetwork.create(genome[1], config))
            genome[1].fitness = 0 if genome[1].fitness is None else genome[1].fitness
        run = True
        maximum = 0
        while run:
            background(screen, "NEAT")
            # Runs a frame of the game, extracting necessary inputs
            inputs = self.loop(screen)

            for i in range(len(self.players)):
                player = self.players[i]
                # Confirming that player is still alive
                if player is not None:
                    # Calculating the cumulative differences of past 100 y-values
                    for j in range(1, len(player[1])):
                        player[1][len(player[1]) - j] = player[1][len(player[1]) - j - 1]
                    player[1][0] = player[0].y
                    dif = 0
                    for g in range(1, len(player[1])):
                        dif += player[1][len(player[1]) - g] - player[1][len(player[1]) - g - 1]

                    # Calculates the outputs of the player's respective neural network based off of the player's
                    # respective inputs
                    if self.count == -1 and not player[0].mid_move:
                        output = nets[i].activate(inputs[i])
                        choice = output.index(max(output))
                        # Updating list of past 13 outputs from the network
                        for j in range(1, len(player[3])):
                            player[3][len(player[3]) - j] = player[3][len(player[3]) - j - 1]
                        player[3][0] = choice
                        # Player makes its move based off the output from the network
                        player[0].move(choice)
                    # Setting player's stagnation to 0 if the player moves between more than two lanes
                    if dif > 50 or dif < -50:
                        player[2] = 0
                    # Ends the player's life if they:
                    # 1. Died by getting hit by a car
                    # 2. Were removed due to a stagnating y-value
                    if player[0].is_dead or player[2] == 500 + (player[0].level // 2) * 50:
                        # Fitness is set to score achieved by the player
                        genomes[i][1].fitness = player[0].score

                        # Possible other fitness function which calculates the average score and stagnation in
                        # decision-making that produced worse results
                        # (genomes[i][1].fitness * (self.num_gen - 1)  + player[0].score - player[4]) / self.num_gen

                        # Updating maximum score
                        maximum = player[0].score if player[0].score > maximum else maximum
                        # Setting that failed player's list to None, signifying its death
                        self.players[i] = None

                    # Increases the player's stagnation regardless, will be set to 0 again if it is not stagnating
                    player[2] += 1

                    # Calculating 13 past cumulative decisions made by the player, used in fitness function that is
                    # currently not being used
                    for h in range(2, 4):
                        sum = 0
                        for j in player[3]:
                            sum += 1 if j == h else 0
                        if sum == len(player[3]):
                            player[4] += .1

            # Updates pygame window with some useful information
            font = pygame.font.Font(None, 20)
            screen.blit(font.render("BEST FIT: " + str(int(max([genome[1].fitness for genome in genomes]))) +
                                    ", WORST FIT: " + str(int(min([genome[1].fitness for genome in genomes]))) +
                                    ", SCORE: " + str(maximum) +
                                    ", GENERATION: " + str(self.num_gen), True, (0, 0, 0)), (10, 710))
            pygame.display.update()

            # Calculates when all players have died
            i = 0
            for player in self.players:
                if player is None:
                    i += 1
            if i == len(self.players):
                run = False

            clock.tick(60)
        max_ = -1
        best_genome = None
        for genome in genomes:
            if genome[1].fitness >= max_:
                best_genome = genome
        return best_genome[1]

    def run_genome(self, genomes, config):
        clock = pygame.time.Clock()
        screen_width = 650
        screen_height = 750
        screen = pygame.display.set_mode((screen_width, screen_height))
        background(screen, "NEAT")
        nets = []
        for genome in genomes:
            # Need networks in a list as everything else is in a list regarding the players; works in parallel with
            # self.players list
            nets.append(neatnn.FeedForwardNetwork.create(genome, config))
            genome.fitness = 0 if genome.fitness is None else genome.fitness
        run = True
        maximum = 0
        while run:
            background(screen, "NEAT")
            # Runs a frame of the game, extracting necessary inputs
            inputs = self.loop(screen)

            for i in range(len(self.players)):
                player = self.players[i]
                # Confirming that player is still alive
                if player is not None:
                    # Calculating the cumulative differences of past 100 y-values
                    for j in range(1, len(player[1])):
                        player[1][len(player[1]) - j] = player[1][len(player[1]) - j - 1]
                    player[1][0] = player[0].y
                    dif = 0
                    for g in range(1, len(player[1])):
                        dif += player[1][len(player[1]) - g] - player[1][len(player[1]) - g - 1]

                    # Calculates the outputs of the player's respective neural network based off of the player's
                    # respective inputs
                    if self.count == -1 and not player[0].mid_move:
                        output = nets[i].activate(inputs[i])
                        choice = output.index(max(output))
                        # Updating list of past 13 outputs from the network
                        for j in range(1, len(player[3])):
                            player[3][len(player[3]) - j] = player[3][len(player[3]) - j - 1]
                        player[3][0] = choice
                        # Player makes its move based off the output from the network
                        player[0].move(choice)
                    # Setting player's stagnation to 0 if the player moves between more than two lanes
                    if dif > 50 or dif < -50:
                        player[2] = 0
                    # Ends the player's life if they:
                    # 1. Died by getting hit by a car
                    # 2. Were removed due to a stagnating y-value
                    if player[0].is_dead or player[2] == 500 + (player[0].level // 2) * 50:
                        # Fitness is set to score achieved by the player
                        genomes[i].fitness = player[0].score

                        # Possible other fitness function which calculates the average score and stagnation in
                        # decision-making that produced worse results
                        # (genomes[i][1].fitness * (self.num_gen - 1)  + player[0].score - player[4]) / self.num_gen

                        # Updating maximum score
                        maximum = player[0].score if player[0].score > maximum else maximum
                        # Setting that failed player's list to None, signifying its death
                        self.players[i] = None

                    # Increases the player's stagnation regardless, will be set to 0 again if it is not stagnating
                    player[2] += 1

                    # Calculating 13 past cumulative decisions made by the player, used in fitness function that is
                    # currently not being used
                    for h in range(2, 4):
                        sum = 0
                        for j in player[3]:
                            sum += 1 if j == h else 0
                        if sum == len(player[3]):
                            player[4] += .1



            # Calculates when all players have died
            i = 0
            for player in self.players:
                if player is None:
                    i += 1
            if i == len(self.players):
                run = False

            pygame.display.update()

            clock.tick(60)
        max_ = -1
        best_genome = None
        for genome in genomes:
            if genome.fitness >= max_:
                best_genome = genome
        return best_genome.fitness

# Class for the simple Artificial Neural Network used
# Uses PyTorch module class
class ANN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ANN, self).__init__()
        # Creates an input and hidden layer
        self.fc1 = nn.Linear(input_size, hidden_size)
        # Sets the activation function to relu, the same activation function we are using for NEAT
        self.relu = nn.ReLU()
        # Connects the hidden layer to the output layer
        self.fc2 = nn.Linear(hidden_size, output_size)

    # Pushes information through the neural network, extracting outputs
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class SimpleANN(Game):
    def __init__(self):
        # No need for additional players because only one network is being tested per generation
        super().__init__()

    # Loop method is similar to the one in the parent game class, as we are only working with one player at a time
    def loop(self, screen):
        if self.level == 0:
            self.player.boundaries[3] = 650
        else:
            self.player.boundaries[3] = 700
        if self.count > 50:
            font = pygame.font.Font(None, 30)
            screen.blit(font.render(str(self.count // 50), True, (200, 200, 200)),
                        ((650 - font.render(str(self.count // 50), True, (200, 200, 200)).get_width()) / 2, 350))
        else:
            self.count = 0
            self.player.enabled = True
            if self.player.x <= -50 or self.player.x >= 700:
                self.player.is_dead = True
            for stage in self.stages:
                draw = self.stages.index(stage) == self.level
                for road in stage:
                    for lane in road:
                        for car in lane.props:
                            if car.collides(self.player) and draw:
                                self.player.is_dead = True
            if self.player.y == -50:
                self.player.y = 650
                self.level += 1
            elif self.player.y == 700 and self.level != 0:
                self.player.y = 0
                self.level -= 1
        for stage in self.stages:
            draw = self.stages.index(stage) == self.level
            for road in stage:
                for lane in road:
                    lane.update(pygame, screen, draw=draw)
        self.player.update(pygame, screen)
        self.level = self.player.level
        self.score = int((self.level * 10) + ((self.level + 1) * 650 - self.player.y) // 5)
        self.score = self.score - (self.score % 10)

        self.count -= 1
        closest = self.player.closest(self.stages[self.level][0], self.stages[self.level][1], 3)
        for car in closest:
            pygame.draw.line(screen, (255, 0, 0), (car[1][0] + car[1][2] // 2, car[1][1] + 25),
                             (self.player.x + 25, self.player.y + 25))

        # Same Outputs as NEAT
        # Outputs, or result of the frame, are appended to greater outputs list
        # These are the inputs to the neural network
        # Outputs are:
        # Player x-value and adjusted y-value based on the level it is on
        # The x-value, adjusted y-value based on level, and length in pixels of each car the closest method
        # deemed closest
        # TO DO: Look into adding a binary input representing direction of a car
        return (float(self.player.x), float(650 - self.player.y) + 660 * self.player.level,
                float(closest[0][1][0]), float(closest[0][1][1]) + 660 * self.player.level, float(closest[0][1][2] * 50), float(closest[0][1][3]),
                float(closest[1][1][0]), float(closest[1][1][1]) + 660 * self.player.level, float(closest[1][1][2]* 50), float(closest[1][1][3]),
                float(closest[2][1][0]), float(closest[2][1][1]) + 660 * self.player.level, float(closest[2][1][2] * 50), float(closest[2][1][3]))

    def train(self):
        clock = pygame.time.Clock()
        screen_width = 650
        screen_height = 750
        screen = pygame.display.set_mode((screen_width, screen_height), pygame.RESIZABLE)
        background(screen, "SIMPLE ANN")

        model = ANN(14, 4, 5)
        # Mean Squared Loss criterion, computes the average square difference between the target value and the actual
        # value. This criterion is used to compare the score achieved (fitness) to the threshold fitness
        criterion = nn.MSELoss()
        # SGD Optimizer, a process which minimizes the loss function by adjusting weights in the neural network based
        # on the largest rate of increase, or gradient, of the function. The learning rate affects how large the "steps"
        # or changes the optimizer makes each iteration
        optimizer = optim.SGD(model.parameters(), lr=.5)
        run = True
        # Maximum fitness
        threshold = 690
        i = 0
        fitness = 0
        # i value is the maximum amount of generations and threshold is maximum fitness
        while i < 3000 and fitness < threshold:
            self.player = Player()
            self.level = 0
            self.score = 0
            self.count = 199
            self.stages = [([Lane(True, y, y % 100 == 0, i / 10, color=self.color_options[i % len(self.color_options)])
                             for y in range(400, 650, 50)],
                            [Lane(True, y, y % 100 == 0, i / 10, color=self.color_options[i % len(self.color_options)])
                             for y in range(50, 300, 50)]) for i in range(5)]
            stagnation = 0
            run = True
            y = [-1 for _ in range(100)]
            while run:
                background(screen, "SIMPLE ANN")
                # Runs a frame of the game, extracting necessary inputs
                inputs = self.loop(screen)
                # Calculating the cumulative differences of past 100 y-values
                for j in range(1, len(y)):
                    y[len(y) - j] = y[len(y) - j - 1]
                y[0] = self.player.y
                dif = 0
                for g in range(1, len(y)):
                    dif += y[len(y) - g] - y[len(y) - g - 1]

                if self.count == -1:
                    # Calculates the outputs of the player's neural network based off of the inputs
                    output = model(torch.tensor(inputs, dtype=torch.float))
                    # Making a move based off of the maximum output
                    self.player.move(torch.argmax(output))
                # Resetting stagnation if the player moved between more than two lanes
                if dif > 50 or dif < -50:
                    stagnation = 0
                # Once the player is either dead or no longer making insightful moves, we set the fitness to the score
                # received and iterate again
                if self.player.is_dead or stagnation == 500 + (self.player.level // 2) * 50:
                    fitness = self.score
                    run = False
                    break
                stagnation += 1
                font = pygame.font.Font(None, 40)
                screen.blit(
                    font.render("FITNESS: " + str(int(fitness)) + ", GENERATION: " + str(i + 1), True, (0, 0, 0)),
                    (10, 710))
                pygame.display.update()
                clock.tick(60)
            # Calculating our loss based off of the Mean Squared Loss
            loss = criterion(torch.tensor(fitness, dtype=torch.float, requires_grad=True), torch.tensor(threshold, dtype=torch.float, requires_grad=True))
            # Logging the best fitness so we can track progress as the network learns
            csv_file_path = "best_fitness_ann.csv"
            with open(csv_file_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([i + 1, fitness])
            # Adjusting our model based off of the loss and fitness function
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            i += 1
            print(f"iteration {i} complete")
        return model

    def run_model(self, model):
        clock = pygame.time.Clock()
        screen_width = 650
        screen_height = 750
        screen = pygame.display.set_mode((screen_width, screen_height), pygame.RESIZABLE)
        stagnation = 0
        run = True
        y = [-1 for _ in range(100)]
        while run:
            background(screen, "SIMPLE ANN")
            # Runs a frame of the game, extracting necessary inputs
            inputs = self.loop(screen)
            # Calculating the cumulative differences of past 100 y-values
            for j in range(1, len(y)):
                y[len(y) - j] = y[len(y) - j - 1]
            y[0] = self.player.y
            dif = 0
            for g in range(1, len(y)):
                dif += y[len(y) - g] - y[len(y) - g - 1]

            if self.count == -1:
                # Calculates the outputs of the player's neural network based off of the inputs
                output = model(torch.tensor(inputs, dtype=torch.float))
                # Making a move based off of the maximum output
                self.player.move(torch.argmax(output))
            # Resetting stagnation if the player moved between more than two lanes
            if dif > 50 or dif < -50:
                stagnation = 0
            # Once the player is either dead or no longer making insightful moves, we set the fitness to the score
            # received and iterate again
            if self.player.is_dead or stagnation == 500 + (self.player.level // 2) * 50:
                fitness = self.score
                run = False
                break
            stagnation += 1
            font = pygame.font.Font(None, 40)
            screen.blit(
                font.render("SCORE: " + str(self.score), True, (0, 0, 0)), (10, 710))
            pygame.display.update()
            clock.tick(60)
        return self.score