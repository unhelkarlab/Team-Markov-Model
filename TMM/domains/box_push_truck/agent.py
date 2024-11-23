from typing import Sequence
import numpy as np
from TMM.models.policy import CachedPolicyInterface
from TMM.domains.agent import (AIAgent_PartialObs, AIAgent_Abstract,
                               BTILCachedAgentModel)
from TMM.domains.agent.cached_agent import NoMindCachedPolicy
from TMM.domains.box_push_truck import (conv_box_idx_2_state,
                                        conv_box_state_2_idx, BoxState)
from TMM.domains.box_push_truck.mdp import MDP_BoxPushV2
from TMM.domains.box_push_truck.agent_model import (
    AM_BoxPushV2, AM_BoxPushV2_Cleanup, AM_BoxPushV2_Movers,
    assumed_initial_mental_distribution)


class BoxPushAIAgent_PartialObs(AIAgent_PartialObs):

  def observed_states(self, tup_states):
    box_states, a1_pos, a2_pos = tup_states

    mdp = self.agent_model.get_reference_mdp()  # type: MDP_BoxPushV2
    num_drops = len(mdp.drops)
    num_goals = len(mdp.goals)
    assert num_goals == 1
    prev_box_states, prev_a1_pos, prev_a2_pos = self.assumed_tup_states

    def max_dist(my_pos, mate_pos):
      return max(abs(my_pos[0] - mate_pos[0]), abs(my_pos[1] - mate_pos[1]))

    def assumed_state(prev_box_states, prev_my_pos, prev_mate_pos, box_states,
                      my_pos, mate_pos, e_boxstate_with_me: BoxState,
                      e_boxstate_with_mate: BoxState):
      agent_dist = max_dist(my_pos, mate_pos)

      assumed_box_states = list(prev_box_states)

      assumed_my_pos = my_pos
      assumed_mate_pos = prev_mate_pos

      if agent_dist <= 1:
        assumed_mate_pos = mate_pos
      else:
        prev_agent_dist = max_dist(my_pos, prev_mate_pos)
        if prev_agent_dist <= 1:
          possible_coords = [(prev_mate_pos[0], prev_mate_pos[1]),
                             (prev_mate_pos[0] - 1, prev_mate_pos[1]),
                             (prev_mate_pos[0] + 1, prev_mate_pos[1]),
                             (prev_mate_pos[0], prev_mate_pos[1] - 1),
                             (prev_mate_pos[0], prev_mate_pos[1] + 1)]
          for crd in possible_coords:
            if crd[0] < 0 or crd[0] >= mdp.x_grid:
              continue
            if crd[1] < 0 or crd[1] >= mdp.y_grid:
              continue
            if crd in mdp.walls:
              continue
            dist_tmp = max_dist(my_pos, crd)
            if dist_tmp > 1:
              assumed_mate_pos = crd
              break

      for idx, coord in enumerate(mdp.boxes):
        if max_dist(my_pos, coord) <= 1:
          bstate = conv_box_idx_2_state(box_states[idx], num_drops, num_goals)
          if bstate[0] != BoxState.Original:
            # if not at the original location, first assume it's at goal
            assumed_box_states[idx] = conv_box_state_2_idx(
                (BoxState.OnGoalLoc, 0), num_drops)

      for idx, bidx in enumerate(box_states):
        # get box position (coordinates). if box is at goal, it will be None
        bpos = None
        bstate = conv_box_idx_2_state(bidx, num_drops, num_goals)
        if bstate[0] == BoxState.Original:
          bpos = mdp.boxes[idx]
        elif bstate[0] == e_boxstate_with_me:
          bpos = my_pos
        elif bstate[0] == e_boxstate_with_mate:
          bpos = mate_pos
        elif bstate[0] == BoxState.WithBoth:
          bpos = my_pos
        elif bstate[0] == BoxState.OnDropLoc:
          bpos = mdp.drops[bstate[1]]

        prev_bstate = conv_box_idx_2_state(prev_box_states[idx], num_drops,
                                           num_goals)
        # if previously box was with me but now it's at the goal --> goal
        if (prev_bstate[0] in [e_boxstate_with_me, BoxState.WithBoth]
            and bpos is None):
          assumed_box_states[idx] = conv_box_state_2_idx(
              (BoxState.OnGoalLoc, 0), num_drops)
        # if previously box was at goal held by friend, i'm close to the goal.
        # now it's placed at goal
        elif (agent_dist <= 1 and prev_mate_pos == mdp.goals[0]
              and prev_bstate[0] == e_boxstate_with_mate and bpos is None):
          assumed_box_states[idx] = conv_box_state_2_idx(
              (BoxState.OnGoalLoc, 0), num_drops)
        # box position is not at goal. update only if it's visible by me
        elif bpos is not None:
          if max_dist(my_pos, bpos) <= 1:
            assumed_box_states[idx] = bidx

      return tuple(assumed_box_states), assumed_my_pos, assumed_mate_pos

    if self.agent_idx == 0:
      assumed_box_states, assumed_a1_pos, assumed_a2_pos = assumed_state(
          prev_box_states, prev_a1_pos, prev_a2_pos, box_states, a1_pos, a2_pos,
          BoxState.WithAgent1, BoxState.WithAgent2)
    else:
      assumed_box_states, assumed_a2_pos, assumed_a1_pos = assumed_state(
          prev_box_states, prev_a2_pos, prev_a1_pos, box_states, a2_pos, a1_pos,
          BoxState.WithAgent2, BoxState.WithAgent1)

    return assumed_box_states, assumed_a1_pos, assumed_a2_pos

  def observed_actions(self, tup_actions, tup_nxt_state) -> tuple:
    bstate, a1_pos, a2_pos = self.assumed_tup_states
    observed_actions = [None, None]
    if self.agent_idx == 0:
      observed_actions[0] = tup_actions[0]
      if max(abs(a1_pos[0] - a2_pos[0]), abs(a1_pos[1] - a2_pos[1])) <= 1:
        observed_actions[1] = tup_actions[1]
    else:
      observed_actions[1] = tup_actions[1]
      if max(abs(a1_pos[0] - a2_pos[0]), abs(a1_pos[1] - a2_pos[1])) <= 1:
        observed_actions[0] = tup_actions[0]

    return tuple(observed_actions)


class BoxPushAIAgent_PO_Team(BoxPushAIAgent_PartialObs):

  def __init__(self,
               init_tup_states,
               policy_model: CachedPolicyInterface,
               has_mind: bool = True,
               agent_idx: int = 0) -> None:
    super().__init__(init_tup_states, policy_model, has_mind, agent_idx)

  def _create_agent_model(self,
                          policy_model: CachedPolicyInterface) -> AM_BoxPushV2:
    return AM_BoxPushV2_Movers(agent_idx=self.agent_idx,
                               policy_model=policy_model)


class BoxPushAIAgent_PO_Indv(BoxPushAIAgent_PartialObs):

  def __init__(self,
               init_tup_states,
               policy_model: CachedPolicyInterface,
               has_mind: bool = True,
               agent_idx: int = 0) -> None:
    super().__init__(init_tup_states, policy_model, has_mind, agent_idx)

  def _create_agent_model(self,
                          policy_model: CachedPolicyInterface) -> AM_BoxPushV2:
    return AM_BoxPushV2_Cleanup(self.agent_idx, policy_model)


class BoxPushAIAgent_Team(AIAgent_Abstract):

  def __init__(self,
               policy_model: CachedPolicyInterface,
               has_mind: bool = True,
               agent_idx: int = 0) -> None:
    super().__init__(policy_model, has_mind, agent_idx)

  def _create_agent_model(self,
                          policy_model: CachedPolicyInterface) -> AM_BoxPushV2:
    return AM_BoxPushV2_Movers(agent_idx=self.agent_idx,
                               policy_model=policy_model)


class BoxPushAIAgent_Indv(AIAgent_Abstract):

  def __init__(self,
               policy_model: CachedPolicyInterface,
               has_mind: bool = True,
               agent_idx: int = 0) -> None:
    super().__init__(policy_model, has_mind, agent_idx)

  def _create_agent_model(self,
                          policy_model: CachedPolicyInterface) -> AM_BoxPushV2:
    return AM_BoxPushV2_Cleanup(self.agent_idx, policy_model)


class BoxPushAIAgent_BTIL(AIAgent_Abstract):

  def __init__(self,
               np_tx: np.ndarray,
               mask_sas: Sequence[bool],
               policy_model: CachedPolicyInterface,
               agent_idx: int = 0,
               np_bx: np.ndarray = None,
               np_coach: np.ndarray = None) -> None:
    self.np_tx = np_tx
    self.mask_sas = mask_sas
    self.np_bx = np_bx
    self.np_coach = np_coach
    super().__init__(policy_model, True, agent_idx)

  def _create_agent_model(self, policy_model: CachedPolicyInterface):

    def init_latents(obstate_idx):
      if self.np_bx is None:
        return assumed_initial_mental_distribution(self.agent_idx, obstate_idx,
                                                   policy_model.mdp)
      else:
        if self.np_coach is not None:
          xidx = self.get_coaching_x(obstate_idx)
          np_bx = np.zeros(self.np_coach.shape[1 + self.agent_idx])
          np_bx[xidx] = 1.0
          return np_bx

        return self.np_bx[obstate_idx]

    return BTILCachedAgentModel(init_latents, self.np_tx, self.mask_sas,
                                policy_model)

  def get_coaching_x(self, obstate_idx):
    max_idx = self.np_coach[obstate_idx].reshape(-1).argmax()
    max_coords = np.unravel_index(max_idx, self.np_coach.shape[1:])
    xidx = max_coords[self.agent_idx]
    return xidx

  def update_mental_state(self, tup_cur_state, tup_actions, tup_nxt_state):
    'tup_actions: tuple of actions'

    if self.np_coach is not None:
      mdp = self.agent_model.get_reference_mdp()  # type: LatentMDP
      sidx_nxt = mdp.conv_sim_states_to_mdp_sidx(tup_nxt_state)
      xidx = self.get_coaching_x(sidx_nxt)
      self.agent_model.set_init_mental_state_idx(None, init_latent=xidx)
    else:
      super().update_mental_state(tup_cur_state, tup_actions, tup_nxt_state)


class BoxPushAIAgent_BTIL_ABS(AIAgent_Abstract):

  def __init__(self,
               np_tx: np.ndarray,
               mask_sas: Sequence[bool],
               policy_model: CachedPolicyInterface,
               agent_idx: int = 0,
               np_bx: np.ndarray = None,
               np_abs: np.ndarray = None,
               np_coach: np.ndarray = None) -> None:
    self.np_tx = np_tx
    self.mask_sas = mask_sas
    self.np_bx = np_bx
    self.np_abs = np_abs
    self.np_coach = np_coach
    self.cur_abs = None
    super().__init__(policy_model, True, agent_idx)

  def _create_agent_model(self, policy_model: CachedPolicyInterface):

    def init_latents(obstate_idx):
      if self.np_bx is None:
        return assumed_initial_mental_distribution(self.agent_idx, obstate_idx,
                                                   policy_model.mdp)
      else:
        if self.np_coach is not None:
          max_idx = self.np_coach[self.cur_abs].reshape(-1).argmax()
          max_coords = np.unravel_index(max_idx, self.np_coach.shape[1:])
          xidx = max_coords[self.agent_idx]
          np_bx = np.zeros(self.np_coach.shape[1 + self.agent_idx])
          np_bx[xidx] = 1.0
          return np_bx

        return self.np_bx[obstate_idx]

    return BTILCachedAgentModel(init_latents, self.np_tx, self.mask_sas,
                                policy_model)

  def conv_obstate_to_abstate(self, obstate_idx):
    TOP_1 = True
    TOP_3 = False
    if TOP_1:
      abstate = np.argmax(self.np_abs[obstate_idx])
    elif TOP_3:
      ind = np.argpartition(self.np_abs[obstate_idx], -3)[-3:]
      np_new_dist = self.np_abs[obstate_idx][ind]
      print(np_new_dist)
      np_new_dist = np_new_dist / np.sum(np_new_dist)[..., None]
      abstate = np.random.choice(ind, p=np_new_dist)
    else:
      abstate = np.random.choice(self.np_abs.shape[-1],
                                 p=self.np_abs[obstate_idx])
    return abstate

  def init_latent(self, tup_states):
    mdp = self.agent_model.get_reference_mdp()  # type: LatentMDP
    sidx = mdp.conv_sim_states_to_mdp_sidx(tup_states)
    self.cur_abs = self.conv_obstate_to_abstate(sidx)
    self.agent_model.set_init_mental_state_idx(self.cur_abs)

  def get_current_latent(self):
    if self.agent_model.is_current_latent_valid():
      return self.conv_idx_to_latent(self.agent_model.current_latent)
    else:
      return None

  def get_action(self, tup_states):
    if self.manual_action is not None:
      next_action = self.manual_action
      self.manual_action = None
      return next_action

    if self.np_coach is not None:
      max_idx = self.np_coach[self.cur_abs].reshape(-1).argmax()
      max_coords = np.unravel_index(max_idx, self.np_coach.shape[1:])
      xidx = max_coords[self.agent_idx]
      self.agent_model.set_init_mental_state_idx(None, xidx)
    # mdp = self.agent_model.get_reference_mdp()  # type: LatentMDP
    # sidx = mdp.conv_sim_states_to_mdp_sidx(tup_states)
    # self.cur_abs = self.conv_obstate_to_abstate(sidx)
    tup_aidx = self.agent_model.get_action_idx(self.cur_abs)
    return self.agent_model.policy_model.conv_idx_to_action(tup_aidx)[0]

  def update_mental_state(self, tup_cur_state, tup_actions, tup_nxt_state):
    'tup_actions: tuple of actions'

    mdp = self.agent_model.get_reference_mdp()  # type: LatentMDP
    sidx_cur = mdp.conv_sim_states_to_mdp_sidx(tup_cur_state)
    sidx_nxt = mdp.conv_sim_states_to_mdp_sidx(tup_nxt_state)

    list_aidx = []
    for idx, act in enumerate(tup_actions):
      if act is None:
        list_aidx.append(None)
      else:
        list_aidx.append(mdp.dict_factored_actionspace[idx].action_to_idx[act])

    prev_abs = self.cur_abs
    self.cur_abs = self.conv_obstate_to_abstate(sidx_nxt)
    if self.np_coach is None:
      self.agent_model.update_mental_state_idx(prev_abs, tuple(list_aidx),
                                               self.cur_abs)

  def set_latent(self, latent):
    xidx = self.conv_latent_to_idx(latent)
    self.agent_model.set_init_mental_state_idx(None, xidx)

  def set_action(self, action):
    self.manual_action = action

  def get_action_distribution(self, state_idx, latent_idx):
    return self.agent_model.policy_model.policy(
        self.conv_obstate_to_abstate(state_idx), latent_idx)

  def get_next_latent_distribution(self, latent_idx, state_idx,
                                   tuple_action_idx, next_state_idx):
    return self.agent_model.transition_mental_state(
        latent_idx, self.conv_obstate_to_abstate(state_idx), tuple_action_idx,
        self.conv_obstate_to_abstate(next_state_idx))

  def get_initial_latent_distribution(self, state_idx):
    return self.agent_model.initial_mental_distribution(
        self.conv_obstate_to_abstate(state_idx))

  def conv_idx_to_latent(self, latent_idx):
    return self.agent_model.policy_model.conv_idx_to_latent(latent_idx)

  def conv_latent_to_idx(self, latent):
    return self.agent_model.policy_model.conv_latent_to_idx(latent)


class AIAgent_NoMind(AIAgent_Abstract):

  def __init__(self,
               policy_model: NoMindCachedPolicy,
               agent_idx: int = 0) -> None:
    super().__init__(policy_model, False, agent_idx)
    self.policy_model = policy_model

  def _create_agent_model(self, policy_model: CachedPolicyInterface):
    return None

  def init_latent(self, tup_states):
    return

  def get_current_latent(self):
    return None

  def get_action(self, tup_states):
    if self.manual_action is not None:
      next_action = self.manual_action
      self.manual_action = None
      return next_action

    mdp = self.policy_model.mdp
    sidx = mdp.conv_sim_states_to_mdp_sidx(tup_states)
    tup_aidx = self.policy_model.get_action(sidx, None)
    return self.policy_model.conv_idx_to_action(tup_aidx)[0]

  def update_mental_state(self, tup_cur_state, tup_actions, tup_nxt_state):
    'tup_actions: tuple of actions'
    pass

  def set_latent(self, latent):
    pass

  def get_action_distribution(self, state_idx, latent_idx):
    return self.agent_model.policy_model.policy(state_idx, latent_idx)

  def get_next_latent_distribution(self, latent_idx, state_idx,
                                   tuple_action_idx, next_state_idx):
    return None

  def get_initial_latent_distribution(self, state_idx):
    return None

  def conv_idx_to_latent(self, latent_idx):
    return None

  def conv_latent_to_idx(self, latent):
    return None
