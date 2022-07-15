from enum import Enum
import numpy as np
from Logic import Engine, Map


class ActionSpace(Enum):
    left = 0
    right = 1
    up = 2
    down = 3
    stay = 4
    place_bomb = 5
    place_trap_left = 6
    place_trap_right = 7
    place_trap_up = 8
    place_trap_down = 9


class Player:
    field_dimensions = 15
    status_vector_length = 6

    def __init__(self, initial_x, initial_y, health, bomb_range, trap_count):
        self.x, self.y = initial_x, initial_y
        self.health = health
        self.opponent_health = health
        # power ups are treated as the same
        self.field_of_vision = np.zeros((Player.status_vector_length, 15, 15), dtype=np.float32)
        self.bomb_range = bomb_range
        self.trap_count = trap_count
        self.traps = set()

    def __call__(self, bomb_range_max, map_height, map_width):
        return [self.field_of_vision,  # tiles,
                self.x / (map_height - 1),  # x/height,
                self.y / (map_width - 1),  # y/width,
                self.health / 10.0,  # hp/10,
                self.opponent_health / 10.0,  # opponent hp/10,
                self.bomb_range / bomb_range_max,  # exp_range/exp_max
                self.trap_count and 1]  # traps/10 # refactor: int(bool(traps))

    def up(self):
        self.field_of_vision = np.insert(self.field_of_vision[:, :Player.field_dimensions - 1, :], 0,
                                         np.zeros(Player.field_dimensions, dtype=np.float32), axis=1)

    def down(self):
        self.field_of_vision = np.insert(self.field_of_vision[:, 1:, :], Player.field_dimensions - 1,
                                         np.zeros(Player.field_dimensions, dtype=np.float32), axis=1)

    def right(self):
        self.field_of_vision = np.insert(self.field_of_vision[:, :, 1:], Player.field_dimensions - 1,
                                         np.zeros(Player.field_dimensions, dtype=np.float32), axis=2)

    def left(self):
        self.field_of_vision = np.insert(self.field_of_vision[:, :, :Player.field_dimensions - 1], 0,
                                         np.zeros(Player.field_dimensions, dtype=np.float32), axis=2)

    def update_tile_status(self, x, y, tile_mask):
        masks = np.array([1, 4, 8, 16, 32, 256])
        for i in range(len(masks)):
            self.field_of_vision[i, x, y] = (tile_mask & masks[i]) and 1
        # fire == bomb
        if tile_mask & 2:
            self.field_of_vision[3, x, y] = 1.0
        # Power ups are the same
        if tile_mask & 64:
            self.field_of_vision[4, x, y] = 1.0
        if tile_mask & 128:
            self.field_of_vision[4, x, y] = 1.0


class Environment:
    def __init__(self):
        self.engine = Engine.Engine()
        health = self.engine.map.player_initial_health
        bomb_range = self.engine.map.player_initial_bomb_range
        trap_count = self.engine.map.player_initial_trap_count
        self.players = [
            Player(
                self.engine.map.player1_initial_x, self.engine.map.player1_initial_y, health, bomb_range, trap_count),
            Player(
                self.engine.map.player2_initial_x, self.engine.map.player2_initial_y, health, bomb_range, trap_count)
        ]
        self.vision_range = self.engine.map.player_vision
        self.turn = self.engine.turn

    def reset(self):
        del self.engine, self.players, self.turn
        self.__init__()
        self.update(ActionSpace.stay.value, 0)
        return \
            self.player(self.engine.map.max_bomb_range,
                        self.engine.map.height, self.engine.map.width), \
            self.players[1 - self.turn](self.engine.map.max_bomb_range,
                                        self.engine.map.height, self.engine.map.width)

    def expand_bomb(self, x, y, dx, dy, exp_range):
        if Player.field_dimensions <= x or \
                Player.field_dimensions <= y or \
                x < 0 or y < 0:
            return 0, 0
        if self.player.field_of_vision[2, x, y]:  # wall
            return 0, 0
        if self.player.field_of_vision[1, x, y]:  # box
            return 1, 0
        if not exp_range:
            return 0, 0

        self.player.field_of_vision[3, x, y] = 1.0
        op_endangered = 0
        if self.player.field_of_vision[5, x, y]:  # check for opponent
            op_endangered = 1

        broken_boxes = 0
        if dx == -1 or dx == dy:
            res = self.expand_bomb(x - 1, y, -1, 0, exp_range - 1)
            broken_boxes += res[0]
            op_endangered += res[1]
        if dx == 1 or dx == dy:
            res = self.expand_bomb(x + 1, y, 1, 0, exp_range - 1)
            broken_boxes += res[0]
            op_endangered += res[1]
        if dy == -1 or dx == dy:
            res = self.expand_bomb(x, y - 1, 0, -1, exp_range - 1)
            broken_boxes += res[0]
            op_endangered += res[1]
        if dy == 1 or dx == dy:
            res = self.expand_bomb(x, y + 1, 0, 1, exp_range - 1)
            broken_boxes += res[0]
            op_endangered += res[1]

        return broken_boxes, op_endangered

    def examine_traps(self, player_x, player_y, latest_action, is_rejected):
        """
    place_trap_left = 6
    place_trap_right = 7
    place_trap_up = 8
    place_trap_down = 9
        """
        trap_coordinates = {
            6: (player_x, player_y - 1),
            7: (player_x, player_y + 1),
            8: (player_x - 1, player_y),
            9: (player_x + 1, player_y),
        }
        if latest_action in trap_coordinates and not is_rejected:
            return self.player.traps.add(trap_coordinates[latest_action])

    def update(self, latest_action, latest_action_rejected):
        center_of_field = Player.field_dimensions // 2
        bombs = set()

        x_ = self.player.x
        y_ = self.player.y
        self.player.x = self.engine.players[self.turn].x
        self.player.y = self.engine.players[self.turn].y
        player_x = self.player.x
        player_y = self.player.y

        other_x = self.engine.players[1 - self.turn].x
        other_y = self.engine.players[1 - self.turn].y

        self.player.health = self.engine.players[self.turn].getHealth()

        if abs(player_x - other_x) + abs(player_y - other_y) <= self.vision_range:
            self.player.opponent_health = self.engine.players[1 - self.turn].getHealth()

        # shift plane of sight
        self.shift_field(x_, y_)

        # account for traps
        self.examine_traps(player_x, player_y, latest_action, latest_action_rejected)

        # to render latest bomb and calculate broken boxes
        latest_planted_bomb = None

        for i in range(-self.vision_range, self.vision_range + 1):
            for j in range(-self.vision_range, self.vision_range + 1):
                if abs(i) + abs(j) > self.vision_range:
                    continue

                # transform to map coordinates
                x = player_x + i
                y = player_y + j

                # get status from engine
                tile = self.engine.map.getTileState(x, y)

                # Remove unobservable states
                tile = Map.remove_state(tile, Map.Tile_State.trap)
                if Map.has_state(tile, Map.Tile_State.box):
                    tile = Map.remove_state(tile, Map.Tile_State.upgrade_trap)
                    tile = Map.remove_state(tile, Map.Tile_State.upgrade_health)
                    tile = Map.remove_state(tile, Map.Tile_State.upgrade_range)

                # render and add bombs to set
                if Map.has_state(tile, Map.Tile_State.bomb):
                    if not latest_planted_bomb and (x, y) == (player_x, player_y) and \
                            latest_action == ActionSpace.place_bomb.value and not latest_action_rejected:
                        latest_planted_bomb = (x - player_x + center_of_field, y - player_y + center_of_field)
                    else:
                        bombs.add((x - player_x + center_of_field, y - player_y + center_of_field))

                # revert to fixed center
                x, y = x - player_x + center_of_field, y - player_y + center_of_field

                # finalizing
                self.player.update_tile_status(x, y, tile)
        # removing the current player
        self.player.field_of_vision[5, 7, 7] = 0.0

        boxes_broken = 0
        opponent_endangered = 0
        if latest_planted_bomb:
            boxes_broken, opponent_endangered = self.expand_bomb(*latest_planted_bomb, 0, 0, self.player.bomb_range)
            bombs.add(latest_planted_bomb)

        for bomb in bombs:
            self.expand_bomb(*bomb, 0, 0, self.engine.map.max_bomb_range)

        # remove a known trap if player triggered it
        if (self.player.x, self.player.y) in self.player.traps:
            self.player.traps.remove((self.player.x, self.player.y))

        for trap in self.player.traps:
            # put traps in field of vision
            delta_x = trap[0] - player_x + center_of_field
            delta_y = trap[1] - player_y + center_of_field
            if delta_x <= Player.field_dimensions - 1 and delta_y <= Player.field_dimensions - 1:
                self.player.field_of_vision[3, delta_x, delta_y] = 1

        return boxes_broken, opponent_endangered

    def clear_hazards(self):
        self.player.field_of_vision[3, :, :] = 0.0

    def change_turns(self):
        self.engine.changeTurn()
        self.engine.nextStep()
        self.turn = self.engine.turn

    def shift_field(self, x, y):
        x, y = self.player.x - x, self.player.y - y

        shift = {
            (0, 0): lambda: True,  # stayed
            (0, -1): self.player.left,  # moved left
            (0, 1): self.player.right,  # moved right
            (-1, 0): self.player.up,  # moved up
            (1, 0): self.player.down,  # moved down
        }
        shift[(x, y)]()

    def reward(self, trap_count_, bomb_range_, boxes_broken, opponent_endangered):
        # player_health, op_health, trap_count, bomb_range, boxes_broken, endangered, op_endangered
        weight = np.array([1, -1, 1, 1, -1, 1, -1, 1])
        params = np.array([self.player.health,
                           self.player.opponent_health,
                           0 if self.player.trap_count < trap_count_ else self.player.trap_count - trap_count_,
                           self.player.bomb_range,
                           bomb_range_,
                           boxes_broken,
                           self.player.field_of_vision[3, 7, 7],
                           opponent_endangered])
        return np.dot(weight, params)

    def step(self, action):
        self.clear_hazards()

        old_trap_count = self.player.trap_count
        old_bomb_range = self.player.bomb_range

        term = self.engine.step(Engine.Action(action)) != -1

        rejected = self.engine.lastAction[self.turn] != Engine.Action(action)
        boxes_broken, opponent_endangered = self.update(action, rejected)

        ret = \
            self.player(self.engine.map.max_bomb_range, self.engine.map.height, self.engine.map.width), \
            self.reward(old_trap_count, old_bomb_range, boxes_broken, opponent_endangered), term, self.turn

        self.change_turns()

        return ret

    @property
    def player(self):
        return self.players[self.turn]


if __name__ == '__main__':
    raise Warning('This module is not meant to be executed directly.')
