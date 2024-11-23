import numpy as np
from TMM.models.mdp import StateSpace
from TMM.domains.box_push.mdp import BoxPushMDP
from TMM.domains.box_push_truck import BoxState, EventType, AGENT_ACTIONSPACE
from TMM.domains.box_push_truck import get_possible_latent_states
from TMM.domains.box_push_truck.transition import transition_mixed


class MDP_BoxPushV2(BoxPushMDP):

  def __init__(self, x_grid, y_grid, boxes, goals, walls, drops, box_types,
               a1_init, a2_init, **kwargs):
    self.box_types = box_types
    self.a1_init = a1_init
    self.a2_init = a2_init
    super().__init__(x_grid, y_grid, boxes, goals, walls, drops, **kwargs)

  def init_latentspace(self):
    latent_states = get_possible_latent_states(len(self.boxes), len(self.drops),
                                               len(self.goals))
    self.latent_space = StateSpace(latent_states)

  def _transition_impl(self, box_states, a1_pos, a2_pos, a1_action, a2_action):
    return transition_mixed(box_states, a1_pos, a2_pos, a1_action, a2_action,
                            self.boxes, self.goals, self.walls, self.drops,
                            self.x_grid, self.y_grid, self.box_types,
                            self.a1_init, self.a2_init)

  def map_to_str(self):
    str_map = super().map_to_str()
    boxes_w_type = list(zip(self.boxes, self.box_types))
    boxes_w_type.sort()
    _, sorted_box_types = zip(*boxes_w_type)
    np_btype = np.array(sorted_box_types) - 1
    str_btype = "".join(np_btype.astype(str))
    int_btype = int(str_btype, base=2)
    base36_btype = np.base_repr(int_btype, base=36)

    return str_map + "_" + base36_btype

  def legal_actions(self, state_idx):
    if self.is_terminal(state_idx):
      return []

    len_s_space = len(self.dict_factored_statespace)
    state_vec = self.conv_idx_to_state(state_idx)

    a1_pos = self.pos1_space.idx_to_state[state_vec[0]]
    a2_pos = self.pos2_space.idx_to_state[state_vec[1]]

    box_states = []
    for idx in range(2, len_s_space):
      box_sidx = state_vec[idx]
      box_state = self.dict_factored_statespace[idx].idx_to_state[box_sidx]
      box_states.append(box_state)

    a1_hold = False
    a2_hold = False
    both_hold = False
    for idx, bstate in enumerate(box_states):
      if bstate[0] == BoxState.WithBoth:
        both_hold = True
      elif bstate[0] == BoxState.WithAgent1:
        a1_hold = True  # noqa: F841
      elif bstate[0] == BoxState.WithAgent2:
        a2_hold = True  # noqa: F841

    if both_hold and a1_pos != a2_pos:  # illegal state
      return []

    # if not (a1_hold or both_hold) and a1_pos in self.goals:  # illegal state
    #   return []

    # if not (a2_hold or both_hold) and a2_pos in self.goals:  # illegal state
    #   return []

    return super().legal_actions(state_idx)


class MDP_BoxPushV2_Agent(MDP_BoxPushV2):

  def init_actionspace(self):
    self.dict_factored_actionspace = {}
    self.my_act_space = AGENT_ACTIONSPACE
    self.dict_factored_actionspace = {0: self.my_act_space}

  def transition_model(self, state_idx: int, action_idx: int) -> np.ndarray:
    if self.is_terminal(state_idx):
      return np.array([[1.0, state_idx]])

    box_states, my_pos, teammate_pos = self.conv_mdp_sidx_to_sim_states(
        state_idx)
    my_act, = self.conv_mdp_aidx_to_sim_actions(action_idx)

    # assume a2 has the same possible actions as a1
    list_p_next_env = []
    for teammate_act in self.my_act_space.actionspace:
      list_p_next_env = list_p_next_env + self._transition_impl(
          box_states, my_pos, teammate_pos, my_act, teammate_act)

    list_next_p_state = []
    map_next_state = {}
    for p, box_states_list, my_pos_n, teammate_pos_n in list_p_next_env:
      sidx_n = self.conv_sim_states_to_mdp_sidx(
          [box_states_list, my_pos_n, teammate_pos_n])
      # assume a2 choose an action uniformly
      map_next_state[sidx_n] = (map_next_state.get(sidx_n, 0) +
                                p / self.num_actions)

    for key in map_next_state:
      val = map_next_state[key]
      list_next_p_state.append([val, key])

    return np.array(list_next_p_state)

  def reward(self, latent_idx: int, state_idx: int, action_idx: int) -> float:
    if self.is_terminal(state_idx):
      return 0

    len_s_space = len(self.dict_factored_statespace)
    state_vec = self.conv_idx_to_state(state_idx)

    my_pos = self.pos1_space.idx_to_state[state_vec[0]]
    # teammate_pos = self.pos2_space.idx_to_state[state_vec[1]]

    box_states = []
    for idx in range(2, len_s_space):
      box_sidx = state_vec[idx]
      box_state = self.dict_factored_statespace[idx].idx_to_state[box_sidx]
      box_states.append(box_state)

    my_act, = self.conv_mdp_aidx_to_sim_actions(action_idx)
    latent = self.latent_space.idx_to_state[latent_idx]

    holding_box = -1
    for idx, bstate in enumerate(box_states):
      if bstate[0] in [BoxState.WithMe, BoxState.WithBoth]:
        holding_box = idx

    panelty = -1

    if latent[0] == "pickup":
      # if already holding a box, set every action but stay as illegal
      if holding_box >= 0:
        if my_act != EventType.HOLD:
          return 0
        else:
          return -np.inf
      else:
        idx = latent[1]
        bstate = box_states[idx]
        box_pos = None
        if bstate[0] == BoxState.Original:
          box_pos = self.boxes[latent[1]]
        elif bstate[0] == BoxState.OnDropLoc:
          box_pos = self.drops[bstate[1]]

        if my_pos == box_pos and my_act == EventType.HOLD:
          return 0
        if my_pos != box_pos and my_act == EventType.HOLD:
          return -np.inf
    elif holding_box >= 0:  # not "pickup" and holding a box --> drop the box
      desired_loc = None
      if latent[0] == "origin":
        desired_loc = self.boxes[holding_box]
      elif latent[0] == "drop":
        desired_loc = self.drops[latent[1]]
      else:  # latent[0] == "goal"
        desired_loc = self.goals[latent[1]]

      if my_pos == desired_loc and my_act == EventType.UNHOLD:
        return 0
      if my_pos != desired_loc and my_act == EventType.UNHOLD:
        return -np.inf
    else:  # "drop the box" but not having a box (illegal state)
      if my_act != EventType.HOLD:
        return 0
      else:
        return -np.inf

    return panelty


class MDP_BoxPushV2_Task(MDP_BoxPushV2):

  def init_actionspace(self):
    self.dict_factored_actionspace = {}
    self.a1_a_space = AGENT_ACTIONSPACE
    self.a2_a_space = AGENT_ACTIONSPACE
    self.dict_factored_actionspace = {0: self.a1_a_space, 1: self.a2_a_space}

  def transition_model(self, state_idx: int, action_idx: int) -> np.ndarray:
    if self.is_terminal(state_idx):
      return np.array([[1.0, state_idx]])

    box_states, a1_pos, a2_pos = self.conv_mdp_sidx_to_sim_states(state_idx)

    act1, act2 = self.conv_mdp_aidx_to_sim_actions(action_idx)

    list_p_next_env = self._transition_impl(box_states, a1_pos, a2_pos, act1,
                                            act2)
    list_next_p_state = []
    map_next_state = {}
    for p, box_states_list, a1_pos_n, a2_pos_n in list_p_next_env:
      sidx_n = self.conv_sim_states_to_mdp_sidx(
          [box_states_list, a1_pos_n, a2_pos_n])
      map_next_state[sidx_n] = map_next_state.get(sidx_n, 0) + p

    for key in map_next_state:
      val = map_next_state[key]
      list_next_p_state.append([val, key])

    return np.array(list_next_p_state)


class MDP_Movers_Agent(MDP_BoxPushV2_Agent):

  def get_possible_box_states(self):
    box_states = [(BoxState.Original, None), (BoxState.WithBoth, None)]
    num_drops = len(self.drops)
    num_goals = len(self.goals)
    if num_drops != 0:
      for idx in range(num_drops):
        box_states.append((BoxState.OnDropLoc, idx))
    for idx in range(num_goals):
      box_states.append((BoxState.OnGoalLoc, idx))
    return box_states


class MDP_Cleanup_Agent(MDP_BoxPushV2_Agent):

  def get_possible_box_states(self):
    box_states = [(BoxState.Original, None), (BoxState.WithAgent1, None),
                  (BoxState.WithAgent2, None)]
    num_drops = len(self.drops)
    num_goals = len(self.goals)
    if num_drops != 0:
      for idx in range(num_drops):
        box_states.append((BoxState.OnDropLoc, idx))
    for idx in range(num_goals):
      box_states.append((BoxState.OnGoalLoc, idx))
    return box_states


class MDP_Movers_Task(MDP_BoxPushV2_Task):

  def get_possible_box_states(self):
    box_states = [(BoxState.Original, None), (BoxState.WithBoth, None)]
    num_drops = len(self.drops)
    num_goals = len(self.goals)
    if num_drops != 0:
      for idx in range(num_drops):
        box_states.append((BoxState.OnDropLoc, idx))
    for idx in range(num_goals):
      box_states.append((BoxState.OnGoalLoc, idx))
    return box_states


class MDP_Cleanup_Task(MDP_BoxPushV2_Task):

  def get_possible_box_states(self):
    box_states = [(BoxState.Original, None), (BoxState.WithAgent1, None),
                  (BoxState.WithAgent2, None)]
    num_drops = len(self.drops)
    num_goals = len(self.goals)
    if num_drops != 0:
      for idx in range(num_drops):
        box_states.append((BoxState.OnDropLoc, idx))
    for idx in range(num_goals):
      box_states.append((BoxState.OnGoalLoc, idx))
    return box_states
