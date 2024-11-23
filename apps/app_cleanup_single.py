from typing import Hashable, Tuple
from apps.app import AppInterface
from TMM.domains.box_push import EventType, BoxState, conv_box_idx_2_state
from TMM.domains.cleanup_single.simulator import CleanupSingleSimulator
from TMM.domains.cleanup_single.agent import Agent_CleanupSingle
from TMM.domains.cleanup_single.policy import Policy_CleanupSingle
from TMM.domains.cleanup_single.mdp import MDPCleanupSingle
from TMM.domains.cleanup_single.maps import MAP_SINGLE_V1

GAME_MAP = MAP_SINGLE_V1


class BoxPushApp(AppInterface):

  def __init__(self) -> None:
    super().__init__()

  def _init_game(self):
    'define game related variables and objects'
    self.x_grid = GAME_MAP["x_grid"]
    self.y_grid = GAME_MAP["y_grid"]
    self.game = CleanupSingleSimulator(False)
    self.game.max_steps = 150

    mdp = MDPCleanupSingle(**GAME_MAP)
    policy = Policy_CleanupSingle(mdp, 0.3)
    agent = Agent_CleanupSingle(policy)

    self.game.init_game(**GAME_MAP)
    self.game.set_autonomous_agent(agent)

  def _init_gui(self):
    self.main_window.title("Box Push")
    self.canvas_width = 300
    self.canvas_height = 300
    super()._init_gui()

  def _conv_key_to_agent_event(self,
                               key_sym) -> Tuple[Hashable, Hashable, Hashable]:
    agent_id = None
    action = None
    value = None
    # agent1 move
    if key_sym == "Left":
      action = EventType.LEFT
    elif key_sym == "Right":
      action = EventType.RIGHT
    elif key_sym == "Up":
      action = EventType.UP
    elif key_sym == "Down":
      action = EventType.DOWN
    elif key_sym == "p":
      action = EventType.HOLD
    elif key_sym == "backslash":
      action = EventType.STAY

    return (agent_id, action, value)

  def _conv_mouse_to_agent_event(
      self, is_left: bool,
      cursor_pos: Tuple[float, float]) -> Tuple[Hashable, Hashable, Hashable]:
    return (None, None, None)

  def _update_canvas_scene(self):
    data = self.game.get_env_info()
    box_states = data["box_states"]
    boxes = data["boxes"]
    drops = data["drops"]
    goals = data["goals"]
    walls = data["walls"]
    agent_pos = data["agent_pos"]

    x_unit = int(self.canvas_width / self.x_grid)
    y_unit = int(self.canvas_height / self.y_grid)

    self.clear_canvas()
    for coord in boxes:
      self.create_rectangle(coord[0] * x_unit, coord[1] * y_unit,
                            (coord[0] + 1) * x_unit, (coord[1] + 1) * y_unit,
                            "gray")

    for coord in goals:
      self.create_rectangle(coord[0] * x_unit, coord[1] * y_unit,
                            (coord[0] + 1) * x_unit, (coord[1] + 1) * y_unit,
                            "gold")

    for coord in walls:
      self.create_rectangle(coord[0] * x_unit, coord[1] * y_unit,
                            (coord[0] + 1) * x_unit, (coord[1] + 1) * y_unit,
                            "black")

    for coord in drops:
      self.create_rectangle(coord[0] * x_unit, coord[1] * y_unit,
                            (coord[0] + 1) * x_unit, (coord[1] + 1) * y_unit,
                            "gray")

    agent_hold = False
    for bidx, sidx in enumerate(box_states):
      state = conv_box_idx_2_state(sidx, len(drops), len(goals))
      box = None
      box_color = "green2"
      if state[0] == BoxState.Original:
        box = boxes[bidx]
      elif state[0] == BoxState.WithAgent1:
        box = agent_pos
        agent_hold = True
        box_color = "green4"
      elif state[0] == BoxState.OnDropLoc:
        box = drops[state[1]]

      if box is not None:
        self.create_rectangle(box[0] * x_unit, box[1] * y_unit,
                              (box[0] + 1) * x_unit, (box[1] + 1) * y_unit,
                              box_color)

    a1_color = "blue"
    if agent_hold:
      a1_color = "dark slate blue"
    self.create_circle((agent_pos[0] + 0.5) * x_unit,
                       (agent_pos[1] + 0.5) * y_unit, x_unit * 0.5, a1_color)

  def _update_canvas_overlay(self):

    x_unit = int(self.canvas_width / self.x_grid)
    y_unit = int(self.canvas_height / self.y_grid)

    self.create_text((self.game.agent_pos[0] + 0.5) * x_unit,
                     (self.game.agent_pos[1] + 0.5) * y_unit,
                     str(self.game.agent.get_current_latent()))

  def _on_game_end(self):
    self.game.reset_game()
    self._update_canvas_scene()
    self._update_canvas_overlay()
    self._on_start_btn_clicked()


if __name__ == "__main__":
  app = BoxPushApp()
  app.run()
