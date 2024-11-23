import os
from TMM.models.policy import CachedPolicyInterface
from TMM.domains.box_push.policy import PolicyFromIdenticalAgentMDP_BoxPush
from TMM.domains.box_push_truck.mdp import MDP_BoxPushV2

policy_movers_list = []
policy_cleanup_list = []


class Policy_Movers(PolicyFromIdenticalAgentMDP_BoxPush):

  def __init__(self, task_mdp: MDP_BoxPushV2, agent_mdp: MDP_BoxPushV2,
               temperature: float, agent_idx: int) -> None:
    super().__init__(task_mdp, agent_idx)

    cur_dir = os.path.dirname(__file__)
    str_fileprefix = os.path.join(cur_dir, "data/qval_movers_")
    str_fileprefix += task_mdp.map_to_str() + "_"
    # In this cached policy, states are represented w.r.t agent1.
    # We need to convert states in task-mdp into states in agent1-mdp.
    self.agent_policy = CachedPolicyInterface(agent_mdp, str_fileprefix,
                                              policy_movers_list, temperature)


class Policy_Cleanup(PolicyFromIdenticalAgentMDP_BoxPush):

  def __init__(self, task_mdp: MDP_BoxPushV2, agent_mdp: MDP_BoxPushV2,
               temperature: float, agent_idx: int) -> None:
    super().__init__(task_mdp, agent_idx)

    cur_dir = os.path.dirname(__file__)
    str_fileprefix = os.path.join(cur_dir, "data/qval_cleanup_")
    str_fileprefix += task_mdp.map_to_str() + "_"
    # In this cached policy, states are represented w.r.t agent1.
    # We need to convert states in task-mdp into states in agent1-mdp.
    self.agent_policy = CachedPolicyInterface(agent_mdp, str_fileprefix,
                                              policy_cleanup_list, temperature)
