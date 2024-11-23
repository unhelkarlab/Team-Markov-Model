from typing import Sequence, Optional, Callable
import numpy as np
from TMM.models.policy import PolicyInterface
from TMM.models.agent_model import AgentModel
from TMM.models.mdp import LatentMDP, StateSpace


class BTILCachedPolicy(PolicyInterface):

  def __init__(self, np_policy: np.ndarray, task_mdp: LatentMDP, agent_idx: int,
               latent_space: StateSpace) -> None:
    super().__init__(task_mdp)
    self.agent_idx = agent_idx
    self.np_policy = np_policy
    self.latent_space = latent_space

  def policy(self, obstate_idx: int, latstate_idx: int) -> np.ndarray:
    return self.np_policy[latstate_idx, obstate_idx]

  def conv_idx_to_action(self, tuple_aidx: Sequence[int]):
    aidx = tuple_aidx[0]
    return self.mdp.dict_factored_actionspace[
        self.agent_idx].idx_to_action[aidx],

  def conv_action_to_idx(self, tuple_actions: Sequence) -> Sequence[int]:
    action = tuple_actions[0]
    return self.mdp.dict_factored_actionspace[
        self.agent_idx].action_to_idx[action],

  def get_num_actions(self):
    return self.mdp.dict_factored_actionspace[self.agent_idx].num_actions

  def get_num_latent_states(self):
    return self.latent_space.num_states

  def conv_idx_to_latent(self, latent_idx: int):
    return self.latent_space.idx_to_state[latent_idx]

  def conv_latent_to_idx(self, latent_state):
    return self.latent_space.state_to_idx[latent_state]


class BTILCachedAgentModel(AgentModel):

  def __init__(self,
               cb_bx: Callable,
               np_tx: np.ndarray,
               mask_sas: Sequence[bool],
               policy_model: Optional[PolicyInterface] = None) -> None:
    '''
    mask_xsas: tuple of bools that tells on which arguments the transition model
              depends among (x, s, a_1, ..., a_n, s').
              For example, if the 1st, 3rd, and the last elements are true, then 
              the transition model only depend on x, a_1, and s'.
    '''
    super().__init__(policy_model)

    self.np_tx = np_tx
    self.cb_bx = cb_bx
    self.mask_sas = mask_sas

  def transition_mental_state(self, latstate_idx: int, obstate_idx: int,
                              tuple_action_idx: Sequence[int],
                              obstate_next_idx: int) -> np.ndarray:
    tuple_idx = (obstate_idx, ) + tuple_action_idx + (obstate_next_idx, )
    list_input = [latstate_idx]
    for idx, masked in enumerate(self.mask_sas):
      if masked:
        list_input.append(tuple_idx[idx])
    # with np.printoptions(precision=3, suppress=True):
    #   print(self.np_tx[tuple(list_input)])
    return self.np_tx[tuple(list_input)]

  def initial_mental_distribution(self, obstate_idx: int) -> np.ndarray:
    return self.cb_bx(obstate_idx)


class NoMindCachedPolicy(BTILCachedPolicy):

  def __init__(self,
               np_policy: np.ndarray,
               task_mdp: LatentMDP,
               agent_idx: int,
               np_abs: Optional[np.ndarray] = None) -> None:
    super().__init__(np_policy, task_mdp, agent_idx, StateSpace())
    self.np_abs = np_abs

  def policy(self, obstate_idx: int, latstate_idx: int) -> np.ndarray:
    obstate = obstate_idx
    if self.np_abs is not None:
      obstate = np.argmax(self.np_abs[obstate_idx])

    return self.np_policy[obstate]
