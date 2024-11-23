from typing import Sequence, Mapping, Tuple
import itertools
import numpy as np
from TMM.models.mdp import LatentMDP, StateSpace
from TMM.domains.rescue_v2 import (Route, E_EventType, E_Type, Location, Work,
                                   Place, T_Connections, AGENT_ACTIONSPACE,
                                   is_work_done)
from TMM.domains.rescue_v2.transition import transition


class MDP_Rescue(LatentMDP):

  def __init__(self, routes: Sequence[Route], places: Sequence[Place],
               connections: Mapping[int, T_Connections],
               work_locations: Sequence[Location], work_info: Sequence[Work],
               **kwarg):
    self.routes = routes
    self.places = places
    self.connections = connections
    self.work_locations = work_locations
    self.work_info = work_info
    super().__init__(use_sparse=True)

  def _transition_impl(self, work_states, a1_pos, a2_pos, a3_pos, a1_action,
                       a2_action, a3_action):
    return transition(work_states, a1_pos, a2_pos, a3_pos, a1_action, a2_action,
                      a3_action, self.routes, self.connections,
                      self.work_locations, self.work_info)

  def map_to_str(self):
    BASE36 = 36
    num_place = len(self.places)
    num_route = len(self.routes)

    str_map = np.base_repr(num_place, BASE36) + np.base_repr(num_route, BASE36)
    str_place = ","
    for place_id in range(num_place):
      str_node = ""
      for connection in self.connections[place_id]:
        if connection[0] == E_Type.Route:
          str_node += np.base_repr(connection[1], BASE36)
        else:
          str_node += np.base_repr(connection[1] + num_route, BASE36)
      str_place += str_node + "_"
    str_map += str_place[:-1]

    str_route = ","
    for route in self.routes:
      str_route += (np.base_repr(route.start, BASE36) +
                    np.base_repr(route.end, BASE36) +
                    np.base_repr(route.length, BASE36) + "_")

    str_map += str_route[:-1]

    return str_map

  def init_statespace(self):
    '''
    To disable dummy states, set self.dummy_states = None
    '''

    self.dict_factored_statespace = {}

    list_locations = []
    for place_id in self.connections:
      if len(self.connections[place_id]) > 0:
        list_locations.append(Location(E_Type.Place, place_id))

    for route_id, route in enumerate(self.routes):
      for idx in range(route.length):
        list_locations.append(Location(E_Type.Route, route_id, idx))

    self.pos1_space = StateSpace(statespace=list_locations)
    self.pos2_space = StateSpace(statespace=list_locations)
    self.pos3_space = StateSpace(statespace=list_locations)

    num_works = len(self.work_locations)
    self.work_states_space = StateSpace(
        statespace=list(itertools.product([0, 1], repeat=num_works)))

    self.dict_factored_statespace = {
        0: self.pos1_space,
        1: self.pos2_space,
        2: self.pos3_space,
        3: self.work_states_space
    }

    self.dummy_states = None

  def is_terminal(self, state_idx):
    factored_state_idx = self.conv_idx_to_state(state_idx)
    work_states = self.work_states_space.idx_to_state[factored_state_idx[-1]]

    for idx in range(len(work_states)):
      if not is_work_done(idx, work_states, self.work_info[idx].coupled_works):
        return False

    return True

  def legal_actions(self, state_idx):
    if self.is_terminal(state_idx):
      return []

    return super().legal_actions(state_idx)

  def conv_sim_states_to_mdp_sidx(self, tup_states) -> int:
    work_states, pos1, pos2, pos3 = tup_states

    pos1_idx = self.pos1_space.state_to_idx[pos1]
    pos2_idx = self.pos2_space.state_to_idx[pos2]
    pos3_idx = self.pos3_space.state_to_idx[pos3]
    work_states_idx = self.work_states_space.state_to_idx[tuple(work_states)]

    return self.conv_state_to_idx(
        (pos1_idx, pos2_idx, pos3_idx, work_states_idx))

  def conv_mdp_sidx_to_sim_states(
      self, state_idx) -> Tuple[Sequence, Location, Location, Location]:
    state_vec = self.conv_idx_to_state(state_idx)
    pos1 = self.pos1_space.idx_to_state[state_vec[0]]
    pos2 = self.pos2_space.idx_to_state[state_vec[1]]
    pos3 = self.pos3_space.idx_to_state[state_vec[2]]
    work_states = self.work_states_space.idx_to_state[state_vec[3]]

    return work_states, pos1, pos2, pos3

  def conv_mdp_aidx_to_sim_actions(self, action_idx):
    vector_aidx = self.conv_idx_to_action(action_idx)
    list_actions = []
    for idx, aidx in enumerate(vector_aidx):
      list_actions.append(
          self.dict_factored_actionspace[idx].idx_to_action[aidx])

    return tuple(list_actions)

  def conv_sim_actions_to_mdp_aidx(self, tuple_actions):
    list_aidx = []
    for idx, act in enumerate(tuple_actions):
      list_aidx.append(self.dict_factored_actionspace[idx].action_to_idx[act])

    return self.np_action_to_idx[tuple(list_aidx)]


class MDP_Rescue_Agent(MDP_Rescue):

  def init_actionspace(self):
    self.my_act_space = AGENT_ACTIONSPACE
    self.dict_factored_actionspace = {0: self.my_act_space}

  def legal_actions(self, state_idx):
    if self.is_terminal(state_idx):
      return []

    work_states, pos1, pos2, pos3 = self.conv_mdp_sidx_to_sim_states(state_idx)

    if pos1.type == E_Type.Route:
      move_actions = [
          self.conv_sim_actions_to_mdp_aidx((E_EventType(idx), ))
          for idx in range(2)
      ]
      stay_actions = [
          self.conv_sim_actions_to_mdp_aidx((E_EventType.Stay, )),
          self.conv_sim_actions_to_mdp_aidx((E_EventType.Rescue, )),
      ]
      return move_actions + stay_actions
    else:
      move_actions = [
          self.conv_sim_actions_to_mdp_aidx((E_EventType(idx), ))
          for idx in range(len(self.connections[pos1.id]))
      ]
      stay_actions = [
          self.conv_sim_actions_to_mdp_aidx((E_EventType.Stay, )),
          self.conv_sim_actions_to_mdp_aidx((E_EventType.Rescue, )),
      ]
      return move_actions + stay_actions

  def init_latentspace(self):
    num_works = len(self.work_locations)
    latent_states = list(range(num_works))
    self.latent_space = StateSpace(latent_states)

  def transition_model(self, state_idx: int, action_idx: int) -> np.ndarray:
    if self.is_terminal(state_idx):
      return np.array([[1.0, state_idx]])

    (work_states, my_pos, mate_pos1,
     mate_pos2) = self.conv_mdp_sidx_to_sim_states(state_idx)
    my_act, = self.conv_mdp_aidx_to_sim_actions(action_idx)

    # assume teammates have the same possible actions as me
    list_p_next_env = []
    for teammate_act1 in AGENT_ACTIONSPACE.actionspace:
      for teammate_act2 in AGENT_ACTIONSPACE.actionspace:
        list_p_next_env = list_p_next_env + self._transition_impl(
            work_states, my_pos, mate_pos1, mate_pos2, my_act, teammate_act1,
            teammate_act2)

    list_next_p_state = []
    map_next_state = {}
    for p, work_states_n, my_pos_n, mate_pos1_n, mate_pos2_n in list_p_next_env:
      sidx_n = self.conv_sim_states_to_mdp_sidx(
          [work_states_n, my_pos_n, mate_pos1_n, mate_pos2_n])
      # assume teammates choose an action uniformly
      map_next_state[sidx_n] = (map_next_state.get(sidx_n, 0) +
                                p / AGENT_ACTIONSPACE.num_actions**2)
    for key in map_next_state:
      val = map_next_state[key]
      list_next_p_state.append([val, key])

    return np.array(list_next_p_state)

  def reward(self, latent_idx: int, state_idx: int, action_idx: int) -> float:
    if self.is_terminal(state_idx):
      return 0

    _, my_pos, _, _ = self.conv_mdp_sidx_to_sim_states(state_idx)
    my_act, = self.conv_mdp_aidx_to_sim_actions(action_idx)
    latent = self.latent_space.idx_to_state[latent_idx]

    panelty = -1

    work_loc = self.work_locations[latent]
    if my_pos == work_loc and my_act == E_EventType.Rescue:
      return 1

    return panelty


class MDP_Rescue_Task(MDP_Rescue):

  def init_actionspace(self):
    self.a1_a_space = AGENT_ACTIONSPACE
    self.a2_a_space = AGENT_ACTIONSPACE
    self.a3_a_space = AGENT_ACTIONSPACE
    self.dict_factored_actionspace = {
        0: self.a1_a_space,
        1: self.a2_a_space,
        2: self.a3_a_space
    }

  def transition_model(self, state_idx: int, action_idx: int) -> np.ndarray:
    if self.is_terminal(state_idx):
      return np.array([[1.0, state_idx]])

    work_states, a1_pos, a2_pos, a3_pos = self.conv_mdp_sidx_to_sim_states(
        state_idx)

    act1, act2, act3 = self.conv_mdp_aidx_to_sim_actions(action_idx)

    list_p_next_env = self._transition_impl(work_states, a1_pos, a2_pos, a3_pos,
                                            act1, act2, act3)
    list_next_p_state = []
    map_next_state = {}
    for p, work_states_n, a1_pos_n, a2_pos_n, a3_pos_n in list_p_next_env:
      sidx_n = self.conv_sim_states_to_mdp_sidx(
          [work_states_n, a1_pos_n, a2_pos_n, a3_pos_n])
      map_next_state[sidx_n] = map_next_state.get(sidx_n, 0) + p

    for key in map_next_state:
      val = map_next_state[key]
      list_next_p_state.append([val, key])

    return np.array(list_next_p_state)

  def legal_actions(self, state_idx):
    if self.is_terminal(state_idx):
      return []

    work_states, pos1, pos2, pos3 = self.conv_mdp_sidx_to_sim_states(state_idx)

    if pos1.type == E_Type.Route:
      a1_actions = [E_EventType(idx) for idx in range(2)]
      a1_actions.append(E_EventType.Stay)
      a1_actions.append(E_EventType.Rescue)
    else:
      a1_actions = [
          E_EventType(idx) for idx in range(len(self.connections[pos1.id]))
      ]
      a1_actions.append(E_EventType.Stay)
      a1_actions.append(E_EventType.Rescue)

    if pos2.type == E_Type.Route:
      a2_actions = [E_EventType(idx) for idx in range(2)]
      a2_actions.append(E_EventType.Stay)
      a2_actions.append(E_EventType.Rescue)
    else:
      a2_actions = [
          E_EventType(idx) for idx in range(len(self.connections[pos2.id]))
      ]
      a2_actions.append(E_EventType.Stay)
      a2_actions.append(E_EventType.Rescue)

    if pos3.type == E_Type.Route:
      a3_actions = [E_EventType(idx) for idx in range(2)]
      a3_actions.append(E_EventType.Stay)
      a3_actions.append(E_EventType.Rescue)
    else:
      a3_actions = [
          E_EventType(idx) for idx in range(len(self.connections[pos3.id]))
      ]
      a3_actions.append(E_EventType.Stay)
      a3_actions.append(E_EventType.Rescue)

    list_actions = []
    for tuple_actions in itertools.product(a1_actions, a2_actions, a3_actions):
      list_actions.append(self.conv_sim_actions_to_mdp_aidx(tuple_actions))

    return list_actions

  def init_latentspace(self):
    return super().init_latentspace()
