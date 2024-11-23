from typing import Hashable, Mapping, Sequence
import os
import numpy as np
from TMM.domains.simulator import Simulator
from TMM.domains.agent import SimulatorAgent, InteractiveAgent
from TMM.domains.rescue import (E_EventType, Route, Location, Work, Place,
                                T_Connections, is_work_done, get_score)
from TMM.domains.rescue.transition import transition


class RescueSimulator(Simulator):
  AGENT1 = 0
  AGENT2 = 1

  def __init__(self, id: Hashable = None) -> None:
    super().__init__(id)
    self.agent_1 = None
    self.agent_2 = None
    self.max_steps = 30

  def init_game(self, places: Sequence[Place], routes: Sequence[Route],
                connections: Mapping[int, T_Connections],
                work_locations: Sequence[Location], work_info: Sequence[Work],
                a1_init, a2_init, **kwargs):

    self.routes = routes
    self.places = places
    self.connections = connections
    self.work_locations = work_locations
    self.work_info = work_info
    self.a1_init = a1_init
    self.a2_init = a2_init

    self.reset_game()

  def get_current_state(self):
    return [self.work_states, self.a1_pos, self.a2_pos]

  @classmethod
  def get_state_action_from_history_item(cls, history_item):
    'return state and a tuple of joint action'
    step, score, wstt, a1pos, a2pos, a1act, a2act, a1lat, a2lat = history_item
    return (wstt, a1pos, a2pos), (a1act, a2act)

  def set_autonomous_agent(self,
                           agent1: SimulatorAgent = InteractiveAgent(),
                           agent2: SimulatorAgent = InteractiveAgent()):
    self.agent_1 = agent1
    self.agent_2 = agent2

    self.agents = [agent1, agent2]

    # order can be important as Agent2 state may include Agent1's mental state,
    # or vice versa. here we assume agent2 updates its mental state later
    self.agent_1.init_latent(self.get_current_state())
    self.agent_2.init_latent(self.get_current_state())

  def reset_game(self):
    self.score = 0
    self.current_step = 0
    self.history = []
    self.a1_pos = self.a1_init
    self.a2_pos = self.a2_init
    self.work_states = [1] * len(self.work_locations)

    if self.agent_1 is not None:
      self.agent_1.init_latent(self.get_current_state())
    if self.agent_2 is not None:
      self.agent_2.init_latent(self.get_current_state())
    self.changed_state = set()

  def update_score(self):
    self.score = get_score(self.work_states, self.work_info, self.places)

  def get_score(self):
    return self.score

  def take_a_step(self, map_agent_2_action: Mapping[Hashable,
                                                    Hashable]) -> None:
    a1_action = None
    if self.AGENT1 in map_agent_2_action:
      a1_action = map_agent_2_action[self.AGENT1]
    a2_action = None
    if self.AGENT2 in map_agent_2_action:
      a2_action = map_agent_2_action[self.AGENT2]

    if a1_action is None and a2_action is None:
      return

    if a1_action is None:
      a1_action = E_EventType.Stay
    if a2_action is None:
      a2_action = E_EventType.Stay

    a1_lat = self.agent_1.get_current_latent()
    if a1_lat is None:
      a1_lat = "None"

    a2_lat = self.agent_2.get_current_latent()
    if a2_lat is None:
      a2_lat = "None"

    a1_cur_state = tuple(self.get_current_state())
    a2_cur_state = tuple(self.get_current_state())

    state = [
        self.current_step, self.score, self.work_states, self.a1_pos,
        self.a2_pos, a1_action, a2_action, a1_lat, a2_lat
    ]
    self.history.append(state)

    self._transition(a1_action, a2_action)
    self.current_step += 1
    self.changed_state.add("current_step")

    self.update_score()
    self.changed_state.add("score")

    if self.is_finished():
      return

    # update mental model
    tuple_actions = (a1_action, a2_action)
    self.agent_1.update_mental_state(a1_cur_state, tuple_actions,
                                     self.get_current_state())
    self.agent_2.update_mental_state(a2_cur_state, tuple_actions,
                                     self.get_current_state())
    self.changed_state.add("a1_latent")
    self.changed_state.add("a2_latent")

  def _transition(self, a1_action, a2_action):
    list_next_env = self._get_transition_distribution(a1_action, a2_action)

    list_prop = []
    for item in list_next_env:
      list_prop.append(item[0])

    idx_c = np.random.choice(range(len(list_next_env)), 1, p=list_prop)[0]
    _, work_states, a1_pos, a2_pos = list_next_env[idx_c]
    self.a1_pos = a1_pos
    self.a2_pos = a2_pos
    self.work_states = work_states

    self.changed_state.add("a1_pos")
    self.changed_state.add("a2_pos")
    self.changed_state.add("work_states")

  def _get_transition_distribution(self, a1_action, a2_action):
    return transition(self.work_states, self.a1_pos, self.a2_pos, a1_action,
                      a2_action, self.routes, self.connections,
                      self.work_locations, self.work_info)

  def get_num_agents(self):
    return 2

  def event_input(self, agent: Hashable, event_type: Hashable, value):
    if (agent is None) or (event_type is None):
      return

    if agent == self.AGENT1:
      if event_type != E_EventType.Set_Latent:
        self.agent_1.set_action(event_type)
      else:
        self.agent_1.set_latent(value)
        self.changed_state.add("a1_latent")
    elif agent == self.AGENT2:
      if event_type != E_EventType.Set_Latent:
        self.agent_2.set_action(event_type)
      else:
        self.agent_2.set_latent(value)
        self.changed_state.add("a2_latent")

  def get_joint_action(self) -> Mapping[Hashable, Hashable]:

    map_a2a = {}
    map_a2a[self.AGENT1] = self.agent_1.get_action(self.get_current_state())
    map_a2a[self.AGENT2] = self.agent_2.get_action(self.get_current_state())

    return map_a2a

  def get_env_info(self):
    return {
        "work_states": self.work_states,
        "a1_pos": self.a1_pos,
        "a2_pos": self.a2_pos,
        "work_locations": self.work_locations,
        "routes": self.routes,
        "places": self.places,
        "connections": self.connections,
        "work_info": self.work_info,
        "a1_latent": self.agent_1.get_current_latent(),
        "a2_latent": self.agent_2.get_current_latent(),
        "current_step": self.current_step,
        "score": self.score
    }

  def get_changed_objects(self):
    dict_changed_obj = {}
    for state in self.changed_state:
      if state == "a1_latent":
        dict_changed_obj[state] = self.agent_1.get_current_latent()
      elif state == "a2_latent":
        dict_changed_obj[state] = self.agent_2.get_current_latent()
      else:
        dict_changed_obj[state] = getattr(self, state)
    self.changed_state = set()
    return dict_changed_obj

  def save_history(self, file_name, header):
    dir_path = os.path.dirname(file_name)
    if dir_path != '' and not os.path.exists(dir_path):
      os.makedirs(dir_path)

    with open(file_name, 'w', newline='') as txtfile:
      # sequence
      txtfile.write(header)
      txtfile.write('\n')
      txtfile.write('# cur_step, score, work_states, a1_pos, a2_pos, ' +
                    'a1_act, a2_act, a1_latent, a2_latent\n')

      for step, score, wstt, a1pos, a2pos, a1act, a2act, a1lat, a2lat in self.history:  # noqa: E501
        txtfile.write('%d; ' % (step, ))  # cur step
        txtfile.write('%d; ' % (score, ))  # score
        # work states
        for idx in range(len(wstt) - 1):
          txtfile.write('%d, ' % (wstt[idx], ))
        txtfile.write('%d; ' % (wstt[-1], ))

        txtfile.write('%s; ' % a1pos)
        txtfile.write('%s; ' % a2pos)

        txtfile.write('%s; %s; ' % (a1act.name, a2act.name))

        txtfile.write('%s; ' % a1lat)
        txtfile.write('%s; ' % a2lat)
        txtfile.write('\n')

      # last state
      txtfile.write('%d; ' % (self.current_step, ))  # cur step
      txtfile.write('%d; ' % (self.score, ))  # score
      # work states
      for idx in range(len(self.work_states) - 1):
        txtfile.write('%d, ' % (self.work_states[idx], ))
      txtfile.write('%d; ' % (self.work_states[-1], ))

      txtfile.write('%s; ' % self.a1_pos)
      txtfile.write('%s; ' % self.a2_pos)
      txtfile.write('\n')

  def is_finished(self) -> bool:
    if super().is_finished():
      return True

    for idx in range(len(self.work_states)):
      if not is_work_done(idx, self.work_states,
                          self.work_info[idx].coupled_works):
        return False

    return True

  @classmethod
  def read_file(cls, file_name):
    traj = []
    with open(file_name, newline='') as txtfile:
      lines = txtfile.readlines()
      i_start = 0
      for i_r, row in enumerate(lines):
        if row == ('# cur_step, score, work_states, a1_pos, a2_pos, ' +
                   'a1_act, a2_act, a1_latent, a2_latent\n'):
          i_start = i_r
          break

      for i_r in range(i_start + 1, len(lines)):
        line = lines[i_r]
        states = line.rstrip()[:-1].split("; ")
        if len(states) < 9:
          for dummy in range(9 - len(states)):
            states.append(None)
        step, score, wstate, a1pos, a2pos, a1act, a2act, a1lat, a2lat = states
        score = int(score)
        work_state = tuple([int(elem) for elem in wstate.split(", ")])
        a1_pos = Location.from_str(a1pos)
        a2_pos = Location.from_str(a2pos)
        if a1act is None:
          a1_act = None
        else:
          a1_act = E_EventType[a1act]
        if a2act is None:
          a2_act = None
        else:
          a2_act = E_EventType[a2act]
        if a1lat is None or a1lat == "None":
          a1_lat = None
        else:
          a1_lat = int(a1lat)
        if a2lat is None or a2lat == "None":
          a2_lat = None
        else:
          a2_lat = int(a2lat)
        traj.append(
            [score, work_state, a1_pos, a2_pos, a1_act, a2_act, a1_lat, a2_lat])

    return traj
