from typing import Sequence
from TMM.domains.box_push.simulator import BoxPushSimulator, Coord
from TMM.domains.box_push_truck.transition import transition_mixed


class BoxPushSimulatorV2(BoxPushSimulator):

  def init_game(self, x_grid: int, y_grid: int, a1_init: Coord, a2_init: Coord,
                boxes: Sequence[Coord], goals: Sequence[Coord],
                walls: Sequence[Coord], drops: Sequence[Coord],
                wall_dir: Sequence[int], box_types: Sequence[int], **kwargs):
    self.box_types = box_types
    super().init_game(x_grid, y_grid, a1_init, a2_init, boxes, goals, walls,
                      drops, wall_dir, **kwargs)

  def _get_transition_distribution(self, a1_action, a2_action):
    return transition_mixed(self.box_states, self.a1_pos, self.a2_pos,
                            a1_action, a2_action, self.boxes, self.goals,
                            self.walls, self.drops, self.x_grid, self.y_grid,
                            self.box_types, self.a1_init, self.a2_init)
