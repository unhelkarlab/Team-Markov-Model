from typing import Sequence, Optional
import numpy as np
from TMM.models.agent_model import AgentModel
from TMM.models.policy import PolicyInterface, CachedPolicyInterface
from TMM.domains.agent import AIAgent_PartialObs, AIAgent_Abstract
from TMM.domains.agent.cached_agent import BTILCachedAgentModel
from TMM.domains.rescue import is_work_done, Location, E_Type
from TMM.domains.rescue.mdp import MDP_Rescue


def assumed_initial_mental_distribution(agent_idx: int, obstate_idx: int,
                                        mdp: MDP_Rescue):
  '''
      obstate_idx: absolute (task-perspective) state representation.
  '''
  work_states, _, _ = mdp.conv_mdp_sidx_to_sim_states(obstate_idx)

  np_work_states = np.array(work_states)
  np_bx = np_work_states / np.sum(np_work_states)

  return np_bx


class RescueAM(AgentModel):

  def __init__(self,
               agent_idx: int,
               policy_model: Optional[PolicyInterface] = None) -> None:
    super().__init__(policy_model)
    self.agent_idx = agent_idx

  def initial_mental_distribution(self, obstate_idx: int) -> np.ndarray:
    '''
        assume agent1 and 2 has the same policy
        state_idx: absolute (task-perspective) state representation.
                    For here, we assume agent1 state and task state is the same
        '''
    mdp = self.get_reference_mdp()  # type: MDP_Rescue
    return assumed_initial_mental_distribution(self.agent_idx, obstate_idx, mdp)

  def transition_mental_state(self, latstate_idx: int, obstate_idx: int,
                              tuple_action_idx: Sequence[int],
                              obstate_next_idx: int) -> np.ndarray:
    '''
    obstate_idx: absolute (task-perspective) state representation.
          Here, we assume agent1 state space and task state space is the same
    '''
    mdp = self.get_reference_mdp()  # type: MDP_Rescue

    work_states_cur, a1_pos, a2_pos = mdp.conv_mdp_sidx_to_sim_states(
        obstate_idx)
    work_states_nxt, _, _ = mdp.conv_mdp_sidx_to_sim_states(obstate_next_idx)

    mate_loc = a2_pos if self.agent_idx == 0 else a1_pos

    if work_states_cur != work_states_nxt:
      if is_work_done(latstate_idx, work_states_nxt,
                      mdp.work_info[latstate_idx].coupled_works):
        np_work_states_nxt = np.array(work_states_nxt)
        for idx, _ in enumerate(work_states_nxt):
          if is_work_done(idx, work_states_nxt,
                          mdp.work_info[idx].coupled_works):
            np_work_states_nxt[idx] = 0

        return np_work_states_nxt / np.sum(np_work_states_nxt)

    P_CHANGE = 0.1
    if mate_loc in mdp.work_locations:
      widx = mdp.work_locations.index(mate_loc)
      wstate_n = work_states_nxt[widx]
      workload = mdp.work_info[widx].workload
      if wstate_n != 0 and workload > 1:
        np_Tx = np.zeros(len(work_states_cur))
        np_Tx[widx] += P_CHANGE
        np_Tx[latstate_idx] += 1 - P_CHANGE
        return np_Tx

    np_Tx = np.zeros(len(work_states_cur))
    np_Tx[latstate_idx] = 1
    return np_Tx


class AIAgent_Rescue_PartialObs(AIAgent_PartialObs):

  def __init__(self,
               init_tup_states,
               agent_idx,
               policy_model: CachedPolicyInterface,
               has_mind: bool = True) -> None:
    super().__init__(init_tup_states, policy_model, has_mind, agent_idx)

  def _create_agent_model(self,
                          policy_model: CachedPolicyInterface) -> RescueAM:
    'Should be implemented at inherited method'
    return RescueAM(agent_idx=self.agent_idx, policy_model=policy_model)

  def observed_states(self, tup_states):
    work_states, a1_pos, a2_pos = tup_states

    mdp = self.agent_model.get_reference_mdp()  # type: MDP_Rescue

    prev_work_states, prev_a1_pos, prev_a2_pos = self.assumed_tup_states
    asm_work_states = list(prev_work_states)
    asm_a1_pos = prev_a1_pos
    asm_a2_pos = prev_a2_pos

    asm_work_states = work_states

    visible_plaec_locs = []
    for idx, place in enumerate(mdp.places):
      if place.visible:
        visible_plaec_locs.append(Location(E_Type.Place, idx))

    if self.agent_idx == 0:
      asm_a1_pos = a1_pos

      if a1_pos == a2_pos:
        asm_a2_pos = a2_pos
      elif a2_pos in visible_plaec_locs:
        asm_a2_pos = a2_pos
    else:
      asm_a2_pos = a2_pos

      if a1_pos == a2_pos:
        asm_a1_pos = a1_pos
      elif a1_pos in visible_plaec_locs:
        asm_a1_pos = a1_pos

    return asm_work_states, asm_a1_pos, asm_a2_pos

  def observed_actions(self, tup_actions, tup_nxt_state):
    observed_actions = [None, None]
    if self.agent_idx == 0:
      observed_actions[0] = tup_actions[0]
    else:
      observed_actions[1] = tup_actions[1]

    return tuple(observed_actions)


class AIAgent_Rescue_BTIL(AIAgent_Abstract):

  def __init__(self,
               np_tx: np.ndarray,
               mask_sas: Sequence[bool],
               policy_model: CachedPolicyInterface,
               agent_idx: int = 0,
               np_bx: np.ndarray = None) -> None:
    self.np_tx = np_tx
    self.mask_sas = mask_sas
    self.np_bx = np_bx
    super().__init__(policy_model, True, agent_idx)

  def _create_agent_model(self, policy_model: CachedPolicyInterface):

    def init_latents(obstate_idx):
      if self.np_bx is None:
        return assumed_initial_mental_distribution(self.agent_idx, obstate_idx,
                                                   policy_model.mdp)
      else:
        return self.np_bx[obstate_idx]

    return BTILCachedAgentModel(init_latents, self.np_tx, self.mask_sas,
                                policy_model)
