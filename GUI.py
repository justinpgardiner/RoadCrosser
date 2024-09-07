import pygame as pygame
import math


class Button:
    def __init__(self, location, size, action, text="", font_size=-1, color=(150, 150, 150)):
        self.size = size
        self.location = location
        self.action = action
        self.color = color
        self.pressed = False
        self.hover_color = tuple([y + 100 if y < 155 else 255 for y in self.color])
        self.text_render = None
        self.text_location = None
        self.font_size = font_size
        self.set_text(text, font_size)

    # W = F(2/3)
    # H = F(1/2)
    def set_text(self, text, font_size=-1):
        self.text = text
        self.font_size = font_size
        if font_size != -1:
            font = pygame.font.Font(None, font_size)
        else:
            font = pygame.font.Font(None,
                                    int(2 * ((self.size[0] - (self.size[0] / 16)) / len(text))) if len(
                                        text) != 0 else 0)
            self.text_render = font.render(text, True, (255, 255, 255) if sum(self.color) < 375 else (0, 0, 0))
            if self.text_render.get_height() > self.size[1]:
                font = pygame.font.Font(None,
                                        int(1.5 * ((self.size[1] - (self.size[1] / 16)) / len(text))) if len(
                                            text) != 0 else 0)
        self.text_render = font.render(text, True, (255, 255, 255) if sum(self.color) < 375 else (0, 0, 0))
        self.text_location = ((self.location[0] + self.size[0] / 2) - (self.text_render.get_width() / 2),
                              (self.location[1] + self.size[1] / 2) - (self.text_render.get_height() / 2))

    def in_range(self, mouse_pos):
        in_x_range = self.location[0] < mouse_pos[0] < self.location[0] + self.size[0]
        in_y_range = self.location[1] < mouse_pos[1] < self.location[1] + self.size[1]
        return in_x_range and in_y_range

    def update(self, pyg, scr):
        display_color = self.hover_color if self.in_range(pyg.mouse.get_pos()) else self.color
        display_size = (self.location[0], self.location[1], self.size[0], self.size[1],)
        pyg.draw.rect(scr, display_color, display_size)
        scr.blit(self.text_render, self.text_location)
        if pygame.mouse.get_pressed()[0] and self.in_range(pyg.mouse.get_pos()) and not self.pressed:
            self.action()
            self.pressed = True
        if not pygame.mouse.get_pressed()[0] and self.pressed:
            self.pressed = False


class ToggleButton(Button):
    def __init__(self, location, size, choices, font_size=-1, color=(150, 150, 150)):
        self.choices = choices
        self.chosen = self.choices[0]
        self.index = 0
        super().__init__(location, size, self.toggle, choices[0], font_size, color)

    def toggle(self):
        self.index = self.index + 1 if self.index + 1 < len(self.choices) else 0
        self.chosen = self.choices[self.index]
        super().set_text(str(self.chosen), self.font_size)

    def set_location(self, location, screen, py):
        self.location = location
        self.set_text(self.text, self.font_size)


class WavyTitle:
    def __init__(self, start_location, font_size, text, wave_range, multiplier, color=(0, 0, 0)):
        self.start_location = start_location
        font = pygame.font.Font(None, font_size)
        self.character_renders = [font.render(char, True, color) for char in text]
        self.wave_range = wave_range
        self.multiplier = multiplier
        self.y = start_location[1]
        self.x = 1

    def update(self, screen):
        x = self.start_location[0]
        char_gen = (char_render for char_render in self.character_renders)
        for char_x in range(self.x, self.x + self.multiplier * len(self.character_renders), self.multiplier):
            char = char_gen.__next__()
            char_x /= 10
            screen.blit(char, (x, self.wave_range * math.sin(char_x) + self.start_location[1]))
            x += char.get_width()
        char_gen.close()
        self.x += 1