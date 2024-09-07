import math
import random

# Contains all "game pieces" (components of the game)
# Used in order to make the game file easier to read and consolidate all commented code relevant to the project in that
# file.


class Player:
    def __init__(self):
        self.lives = 1
        self.x = 300
        self.y = 650
        self.boundaries = [0, -50, 600, 650]
        self.color = (255, 0, 0)
        self.radius = 15
        self.score = 0
        self.level = 0
        self.mid_move = False
        self.x_change = 0
        self.y_change = 0
        self.frames_between_moves = 5
        self.movement_frame = 0
        self.pressed = {"UP": False, "DOWN": False, "LEFT": False, "RIGHT": False}
        self.is_dead = False
        self.log_change = 0
        self.enabled = False

    def update(self, pygame, screen, draw=True):
        if self.level == 0:
            self.boundaries[3] = 650
        else:
            self.boundaries[3] = 700
        for life in range(0, self.lives - 1) and draw:
            pygame.draw.circle(screen, self.color, ((50 * life) + 25, 725), 15)
        if self.is_dead and self.radius > 0:
            self.radius -= 1
            if self.radius == 0:
                self.lives -= 1
                return
        elif self.mid_move and self.movement_frame < self.frames_between_moves:
            self.x += self.x_change + self.log_change
            self.y += self.y_change
            self.movement_frame += 1
            if self.movement_frame == self.frames_between_moves:
                self.mid_move = False
        if self.enabled and not self.mid_move and not self.is_dead:
            keys = pygame.key.get_pressed()
            self.x += self.log_change
            self.x_change = 0
            self.y_change = 0
            self.movement_frame = 0
            if (self.y == 0 and self.level == 0):
                self.y = 0
            if not self.pressed["UP"] and keys[pygame.K_UP] and self.y > self.boundaries[1] and not (self.y == 0 and self.level == 9):
                self.y_change = -50 / self.frames_between_moves
                self.mid_move = True
                self.pressed["UP"] = True
                self.log_change = 0
            elif not self.pressed["DOWN"] and keys[pygame.K_DOWN] and self.y < self.boundaries[3]:
                self.y_change = 50 / self.frames_between_moves
                self.mid_move = True
                self.pressed["DOWN"] = True
                self.log_change = 0
            elif not self.pressed["LEFT"] and keys[pygame.K_LEFT] and self.x > self.boundaries[0]:
                self.x_change = -50 / self.frames_between_moves
                self.mid_move = True
                self.pressed["LEFT"] = True
                self.log_change = 0
            elif not self.pressed["RIGHT"] and keys[pygame.K_RIGHT] and self.x < self.boundaries[2]:
                self.x_change = 50 / self.frames_between_moves
                self.mid_move = True
                self.pressed["RIGHT"] = True
                self.log_change = 0
            self.pressed["UP"] = keys[pygame.K_UP]
            self.pressed["DOWN"] = keys[pygame.K_DOWN]
            self.pressed["LEFT"] = keys[pygame.K_LEFT]
            self.pressed["RIGHT"] = keys[pygame.K_RIGHT]
        if self.lives != 0 and draw:
            pygame.draw.circle(screen, self.color, (self.x + 25, self.y + 25), self.radius)

        self.score = int((self.level * 10) + ((self.level + 1) * 650 - self.y) // 5)
        self.score = self.score - (self.score % 10)

        if self.y == -50:
            self.level += 1
            self.y = 650
        elif self.y <= 0 and self.level == 4:
            self.y = 0
        elif self.y == 700 and self.level != 0:
            self.y = 0
            self.level -= 1

    def get_points(self):
        return (self.x + 25 - self.radius, self.y + 25 - self.radius), \
               (self.x + 25 + self.radius, self.y + 25 - self.radius), \
               (self.x + 25 + self.radius, self.y + 25 + self.radius), \
               (self.x + 25 - self.radius, self.y + 25 - self.radius)

    def move_up(self):
        if not self.mid_move and self.y > self.boundaries[1]:
            self.y_change = -50 / self.frames_between_moves
            self.mid_move = True
            self.pressed["UP"] = True
            self.log_change = 0
            return True
        return False

    def move_down(self):
        if not self.mid_move and self.y != self.boundaries[3]:
            self.y_change = 50 / self.frames_between_moves
            self.mid_move = True
            self.pressed["DOWN"] = True
            self.log_change = 0
            return True
        return False

    def move_left(self):
        if not self.mid_move and self.x != self.boundaries[0]:
            self.x_change = -50 / self.frames_between_moves
            self.mid_move = True
            self.pressed["LEFT"] = True
            self.log_change = 0
            return True
        return False

    def move_right(self):
        if not self.mid_move and self.x != self.boundaries[2]:
            self.x_change = 50 / self.frames_between_moves
            self.mid_move = True
            self.pressed["RIGHT"] = True
            self.log_change = 0
            return True
        return False

    def move(self, pos):
        if pos == 0:
            # print('still')
            pass
        elif pos == 1:
            self.move_up()
            # print('up')
        elif pos == 2:
            self.move_down()
            # print('down')
        elif pos == 3:
            self.move_left()
            # print('left')
        else:
            self.move_right()
            # print('right')

    def closest(self, road1, road2, num):
        props = []
        for lane in road1:
            for prop in lane.props:
                dists = []
                for point in prop.get_points():
                    dists.append(math.sqrt(
                        (abs(self.x + 25 - point[0]) ** 2) + (abs(self.y + 25 - point[1]) ** 2)))
                d = min(dists)
                index = 0
                while index < len(props) and d > props[index][0]:
                    index += 1
                props.insert(index, (d, [prop.x, prop.y, prop.length, prop.velocity]))
        for lane in road2:
            for prop in lane.props:
                dists = []
                for point in prop.get_points():
                    dists.append(math.sqrt(
                        (abs(self.x + 25 - point[0]) ** 2) + (abs(self.y + 25 - point[1]) ** 2)))
                d = min(dists)
                index = 0
                while index < len(props) and d > props[index][0]:
                    index += 1
                props.insert(index, (d, [prop.x, prop.y, prop.length, prop.velocity]))
        closest = []
        k = 0
        for i in range(0, num if len(props) >= num else len(props)):
            closest.append(props[i])
            k += 1
        while k < num:
            # closest.append((10000, 10000, -10000, 3, 0, 0))
            closest.append((1000, [10000, -10000, 3, 0]))
            k += 1
        return closest


class MovingProp:
    def __init__(self, start_y, start_left, num_links, color):
        self.start_left = start_left
        self.x = -50 - 50 * num_links if self.start_left else 700 + 50 * num_links
        self.y = start_y
        self.length = num_links * 50
        self.velocity = 3 if self.start_left else -3
        self.color = color
        self.type = "prop"

    def update(self, pygame, screen, draw=True):
        self.x += self.velocity
        if draw:
            pygame.draw.rect(screen, self.color, pygame.Rect(self.x + 5, self.y + 5, self.length - 10, 40))

    def out_of_bounds(self):
        if self.start_left:
            return self.x > 650
        else:
            return self.x < -self.length

    def get_points(self):
        return (self.x + 5, self.y + 5), \
               (self.x + 5 + self.length - 10, self.y + 5), \
               (self.x + 5 + self.length, self.y + 45), \
               (self.x + 5, self.y + 45)

    def collides(self, player):
        pass


class Car(MovingProp):
    def __init__(self, start_y, start_left, num_links, color):
        super().__init__(start_y, start_left, num_links, color)
        self.type = "car"

    def collides(self, player):
        self_points = self.get_points()
        for point in player.get_points():
            if (self_points[0][0] <= point[0] <= self_points[1][0]) and (self_points[0][1] <= point[1] <= self_points[3][1]):
                return True
        return False


class Log(MovingProp):
    def __init__(self, start_y, start_left, num_links, color):
        super().__init__(start_y, start_left, num_links, color)
        self.velocity = 1 if start_left else -1
        self.type = "log"

    def collides(self, player):
        self_points = self.get_points()
        for point in player.get_points():
            if (self_points[0][0] <= point[0] <= self_points[1][0]) and self.y == player.y:
                return True
        return False


class Lane:
    def __init__(self, is_cars, y, start_left, difficulty, color=(0, 255, 0), is_multi=False):
        self.is_cars = is_cars
        self.color_options = ((255, 255, 0), (0, 255, 255), (0, 255, 0))
        self.y = y
        self.start_left = start_left
        self.color = color
        self.is_multi = is_multi
        self.num_links_current = 0
        self.props = []
        self.create_new_prop()
        self.frames_per_link = abs(50 // self.props[0].velocity)
        self.clock = 0
        self.wait = self.frames_per_link * random.randint(1, 3)
        self.waiting = self.wait != 0
        self.frequency = int(20 / (difficulty/5 + 1))
        if self.waiting:
            self.props.pop(0)

    def create_new_prop(self):
        self.num_links_current = random.randint(1, 3) if self.is_cars else random.randint(2, 3)
        color = self.color_options[random.randint(0, 2)] if self.is_multi else self.color
        prop = Car(self.y, self.start_left, self.num_links_current, color) if self.is_cars else Log(self.y, self.start_left, self.num_links_current, color)
        self.props.append(prop)

    def update(self, pygame, screen, draw=True):
        if self.waiting:
            if self.clock == self.wait:
                self.create_new_prop()
                self.wait = 0
                self.waiting = False
        else:
            if (self.clock % self.frames_per_link) == self.num_links_current:
                self.wait = self.frames_per_link * random.randint(4, self.frequency if self.frequency > 5 else 6)
                self.clock = 0
                self.waiting = True
        props_to_delete = []
        for prop in self.props:
            prop.update(pygame, screen, draw)
            if prop.out_of_bounds():
                props_to_delete.append(prop)
        for prop in props_to_delete:
            self.props.remove(prop)
        self.clock += 1