from abc import abstractmethod
from TMM.models.policy import CachedPolicyInterface
from TMM.domains.agent import AIAgent_Abstract


class AIAgent_PartialObs(AIAgent_Abstract):

  def __init__(self,
               init_tup_states,
               policy_model: CachedPolicyInterface,
               has_mind: bool = True,
               agent_idx: int = 0) -> None:
    super().__init__(policy_model, has_mind, agent_idx)
    self.init_tup_states = init_tup_states
    self.assumed_tup_states = init_tup_states

  @abstractmethod
  def observed_actions(self, tup_actions, tup_nxt_state):
    pass

  @abstractmethod
  def observed_states(self, tup_states):
    # TODO: might need revision.
    pass

  def init_latent(self, tup_state):
    # TODO: check if this is correct as it is not using the input argument
    self.assumed_tup_states = self.init_tup_states
    return super().init_latent(self.assumed_tup_states)

  def get_action(self, tup_state):
    return super().get_action(self.assumed_tup_states)

  def update_mental_state(self, tup_cur_state, tup_actions, tup_nxt_state):
    prev_tuple_states = self.assumed_tup_states
    observed_actions = self.observed_actions(tup_actions, tup_nxt_state)
    self.assumed_tup_states = self.observed_states(tup_nxt_state)

    return super().update_mental_state(prev_tuple_states, observed_actions,
                                       self.assumed_tup_states)
