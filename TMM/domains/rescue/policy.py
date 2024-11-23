import os
from TMM.models.policy import CachedPolicyInterface
from TMM.domains.box_push.policy import PolicyFromIdenticalAgentMDP
from TMM.domains.rescue.mdp import MDP_Rescue

policy_rescue_list = []


class Policy_Rescue(PolicyFromIdenticalAgentMDP):

  def __init__(self, task_mdp: MDP_Rescue, agent_mdp: MDP_Rescue,
               temperature: float, agent_idx: int) -> None:
    super().__init__(task_mdp, agent_idx)

    cur_dir = os.path.dirname(__file__)
    str_fileprefix = os.path.join(cur_dir, "data/qval_rescue_")
    str_fileprefix += task_mdp.map_to_str() + "_"
    # In this cached policy, states are represented w.r.t agent1.
    # We need to convert states in task-mdp into states in agent1-mdp.
    self.agent_policy = CachedPolicyInterface(agent_mdp, str_fileprefix,
                                              policy_rescue_list, temperature)

  def _convert_task_state_2_agent_state(self, obstate_idx):
    work_states, a1_pos, a2_pos = self.mdp.conv_mdp_sidx_to_sim_states(
        obstate_idx)

    pos_1 = a1_pos
    pos_2 = a2_pos
    if self.agent_idx == 1:
      pos_1 = a2_pos
      pos_2 = a1_pos
    return self.agent_policy.mdp.conv_sim_states_to_mdp_sidx(
        [work_states, pos_1, pos_2])
