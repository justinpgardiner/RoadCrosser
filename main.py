from GUI import *
from game import *
from neat import *
import os
import torch
import csv
import pickle

pygame.init()
screen_width = 650
screen_height = 750
pygame.display.set_caption("TRAINING AI")
count = 199

global menu
global playing
global screen
global gen_count
screen = pygame.display.set_mode((screen_width, screen_height), pygame.RESIZABLE)
menu = True
playing = False
gen_count = 1


def background():
    pygame.draw.rect(screen, (0, 50, 0), pygame.Rect(0, 0, 650, 50))
    pygame.draw.rect(screen, (0, 0, 0), pygame.Rect(0, 50, 650, 250))
    pygame.draw.rect(screen, (0, 50, 0), pygame.Rect(0, 300, 650, 100))
    pygame.draw.rect(screen, (0, 0, 0), pygame.Rect(0, 400, 650, 250))
    pygame.draw.rect(screen, (0, 50, 0), pygame.Rect(0, 650, 650, 50))
    pygame.draw.rect(screen, (100, 100, 100), pygame.Rect(0, 700, 650, 50))


# Function to run simple ANN training and save final model
def run_ann():
    final_model = SimpleANN().train()
    torch.save(final_model.state_dict(), 'final_ann.pth')


# Function to run NEAT training and save final model
def run_neat():
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config.txt")
    config = Config(DefaultGenome, DefaultReproduction, DefaultSpeciesSet, DefaultStagnation, config_path)
    # p = Population(config)
    p = Checkpointer.restore_checkpoint("neat-checkpoint-2999")
    p.add_reporter(StdOutReporter(True))
    stats = StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(Checkpointer(300))
    winner = p.run(eval_genomes, 1)
    # print(sum([1 if node.type == 'hidden' else 0 for node in winner.nodes]))
    with open('neat-winner.pkl', 'wb') as file:
        pickle.dump(winner, file)


def run_best_ann():
    model = ANN(14, 4, 5)
    model.load_state_dict(torch.load('final_ann.pth'))
    csv_file_path = "score_of_best_ann.csv"
    for i in range(1, 2):
        score = SimpleANN().run_model(model)
        with open(csv_file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([i, score])


def run_best_neat():
    with open('neat-winner.pkl', 'rb') as f:
        genome = pickle.load(f)
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config.txt")
    config = Config(DefaultGenome, DefaultReproduction, DefaultSpeciesSet, DefaultStagnation, config_path)
    csv_file_path = "scores_of_best_neat.csv"
    for i in range(1, 2):
        score = Neat(0, 1).run_genome([genome], config)
        with open(csv_file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([i, score])


# Function to evaluate genomes needed by run function of the neat-python population class
# Creates a new object from the Neat class and runs through a loop, evaluating the fitness of all genomes/networks in
# the population
def eval_genomes(genomes, config):
    global gen_count
    best = Neat(gen_count, len(genomes)).train(genomes, config)
    csv_file_path = "best_fitness_neat.csv"
    for nodeid in best.nodes:
        print(best.nodes[nodeid])

    with open(csv_file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([gen_count, best.fitness])
    gen_count += 1


def play():
    global playing
    global menu
    playing = True
    menu = False


def return_to_menu():
    global playing
    global menu
    playing = False
    menu = True


clock = pygame.time.Clock()
running = True

title_font = pygame.font.Font(None, 100)
title = title_font.render("Road Crosser", True, (0, 0, 0))

# Fun GUI touches
title = WavyTitle((650 / 2 - title.get_width() / 2, 100), 100, "Road Crosser", 10, 10, color=(200, 200, 200))
play_button = Button((225, 550), (200, 50), play, "Play", font_size=24, color=(200, 200, 200))
option_button = ToggleButton((225, 450), (200, 50), ["Human", "AI (NEAT)", "AI (Traditional ANN)"], font_size=24, color=(200, 200, 200))
game = None
training = False

# run_neat()
# run_ann()
# run_best_neat()
# run_best_ann()


with open('neat-winner.pkl', 'rb') as f:
    genome = pickle.load(f)
print(len(genome.connections))

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill((255, 255, 255))
    background()
    if menu:
        game = Game()
        play_button.update(pygame, screen)
        option_button.update(pygame, screen)
        title.update(screen)
        font = pygame.font.Font(None, 30)
        screen.blit(font.render("Inspired by Frogger and Crossy Road", True, (200, 200, 200)), ((650 - font.render("Inspired by Frogger and Crossy Road", True, (200, 200, 200)).get_width()) / 2, 350))
    elif playing:
        if option_button.chosen == "Human":
            info = game.loop(screen)
            if info is None:
                return_to_menu()
        elif option_button.chosen == "AI (NEAT)":
            run_best_neat()
            return_to_menu()
        elif option_button.chosen == "AI (Traditional ANN)":
            run_best_ann()
            return_to_menu()

    pygame.display.update()
    clock.tick(60)

pygame.quit()
