from typing import Tuple, Sequence
from TMM.domains.box_push_truck import (BoxState, EventType,
                                        conv_box_state_2_idx,
                                        conv_box_idx_2_state)
from TMM.domains.box_push.transition import (get_box_idx_impl,
                                             get_moved_coord_impl,
                                             hold_state_impl,
                                             update_dropped_box_state_impl,
                                             is_opposite_direction)

Coord = Tuple[int, int]


def transition_mixed(box_states: list, a1_pos: Coord, a2_pos: Coord,
                     a1_act: EventType, a2_act: EventType,
                     box_locations: Sequence[Coord], goals: Sequence[Coord],
                     walls: Sequence[Coord], drops: Sequence[Coord],
                     x_bound: int, y_bound: int, box_types: Sequence[int],
                     a1_init: Coord, a2_init: Coord):
  # num_goals = len(goals)
  num_drops = len(drops)
  num_goals = len(goals)

  # methods
  def get_box_idx(coord):
    return get_box_idx_impl(coord, box_states, a1_pos, a2_pos, box_locations,
                            goals, drops)

  def get_moved_coord(coord, action, box_states=None, holding_box=False):
    coord_new = get_moved_coord_impl(coord, action, x_bound, y_bound, walls,
                                     box_states, a1_pos, a2_pos, box_locations,
                                     goals, drops)
    if coord_new in goals and not holding_box:
      return coord
    else:
      return coord_new

  def hold_state():
    return hold_state_impl(box_states, drops, goals)

  def update_dropped_box_state(boxidx, coord, box_states_new):
    res = update_dropped_box_state_impl(boxidx, coord, box_states_new,
                                        box_locations, drops, goals)
    return res, conv_box_idx_2_state(box_states_new[boxidx], num_drops,
                                     num_goals)

  P_MOVE = 1.0
  list_next_env = []
  hold = hold_state()
  # both do not hold anything
  if hold == "None":
    if (a1_act == EventType.HOLD and a2_act == EventType.HOLD
        and a1_pos == a2_pos):
      bidx = get_box_idx(a1_pos)
      if bidx >= 0:
        if box_types[bidx] == 2:
          state = (BoxState.WithBoth, None)
          box_states_new = list(box_states)
          box_states_new[bidx] = conv_box_state_2_idx(state, num_drops)
          list_next_env.append((1.0, box_states_new, a1_pos, a2_pos))
        elif box_types[bidx] == 1:
          state1 = (BoxState.WithAgent1, None)
          state2 = (BoxState.WithAgent2, None)
          box_states_new = list(box_states)
          box_states_new[bidx] = conv_box_state_2_idx(state1, num_drops)
          list_next_env.append((0.5, box_states_new, a1_pos, a2_pos))
          box_states_new = list(box_states)
          box_states_new[bidx] = conv_box_state_2_idx(state2, num_drops)
          list_next_env.append((0.5, box_states_new, a1_pos, a2_pos))
        else:
          raise ValueError("Box types other than 1 or 2 are not implemented")
      else:
        list_next_env.append((1.0, box_states, a1_pos, a2_pos))

    else:
      a1_pos_new = a1_pos
      box_states_new = list(box_states)
      if a1_act == EventType.HOLD:
        bidx = get_box_idx(a1_pos)
        if bidx >= 0:
          if box_types[bidx] == 1:
            state = (BoxState.WithAgent1, None)
            box_states_new[bidx] = conv_box_state_2_idx(state, num_drops)
      else:
        a1_pos_new = get_moved_coord(a1_pos, a1_act, None, False)

      a2_pos_new = a2_pos
      if a2_act == EventType.HOLD:
        bidx = get_box_idx(a2_pos)
        if bidx >= 0:
          if box_types[bidx] == 1:
            state = (BoxState.WithAgent2, None)
            box_states_new[bidx] = conv_box_state_2_idx(state, num_drops)
      else:
        a2_pos_new = get_moved_coord(a2_pos, a2_act, None, False)

      list_next_env.append((1.0, box_states_new, a1_pos_new, a2_pos_new))
  # both hold the same box
  elif hold == "Both":
    bidx = get_box_idx(a1_pos)
    assert bidx >= 0
    assert box_types[bidx] == 2
    # invalid case
    if a1_pos != a2_pos:
      list_next_env.append((1.0, box_states, a1_pos, a2_pos))
      return list_next_env

    # both try to drop the box
    if a1_act == EventType.UNHOLD and a2_act == EventType.UNHOLD:
      box_states_new = list(box_states)
      _, bstate = update_dropped_box_state(bidx, a1_pos, box_states_new)
      a1_pos_new = a1_pos
      a2_pos_new = a2_pos
      # respawn agents if box is dropped at the goal
      if bstate[0] == BoxState.OnGoalLoc:
        a1_pos_new = a1_init
        a2_pos_new = a2_init
      list_next_env.append((1.0, box_states_new, a1_pos_new, a2_pos_new))
    # only agent1 try to unhold
    elif a1_act == EventType.UNHOLD:
      a2_pos_new = get_moved_coord(a2_pos, a2_act, box_states, True)
      a1_pos_new = a2_pos_new
      if a2_pos_new == a2_pos:
        list_next_env.append((1.0, box_states, a1_pos, a2_pos))
      else:
        list_next_env.append((0.5, box_states, a1_pos_new, a2_pos_new))
        list_next_env.append((0.5, box_states, a1_pos, a2_pos))
    elif a2_act == EventType.UNHOLD:
      a1_pos_new = get_moved_coord(a1_pos, a1_act, box_states, True)
      a2_pos_new = a1_pos_new
      if a1_pos_new == a1_pos:
        list_next_env.append((1.0, box_states, a1_pos, a2_pos))
      else:
        list_next_env.append((0.5, box_states, a1_pos_new, a2_pos_new))
        list_next_env.append((0.5, box_states, a1_pos, a2_pos))
    else:
      if (is_opposite_direction(a1_act, a2_act)
          or (a1_act == EventType.STAY and a2_act == EventType.STAY)):
        list_next_env.append((1.0, box_states, a1_pos, a2_pos))
      elif a1_act == a2_act:
        a1_pos_new = get_moved_coord(a1_pos, a1_act, box_states, True)
        list_next_env.append((1.0, box_states, a1_pos_new, a1_pos_new))
      else:
        a1_pos_new = get_moved_coord(a1_pos, a1_act, box_states, True)
        a2_pos_new = get_moved_coord(a2_pos, a2_act, box_states, True)
        if a1_pos_new == a2_pos_new:
          list_next_env.append((1.0, box_states, a1_pos_new, a1_pos_new))
        else:
          list_next_env.append((0.5, box_states, a1_pos_new, a1_pos_new))
          list_next_env.append((0.5, box_states, a2_pos_new, a2_pos_new))
  elif hold == "Each":
    a1_pos_new = a1_pos
    a2_pos_new = a2_pos

    # if more than one remains on the same grid
    if (a1_act in [EventType.UNHOLD, EventType.STAY]
        or a2_act in [EventType.UNHOLD, EventType.STAY]):
      box_states_new = list(box_states)

      p_a1_success = 0
      bidx1 = get_box_idx(a1_pos)
      if bidx1 >= 0 and a1_pos in goals and a1_act == EventType.UNHOLD:
        _, bstate = update_dropped_box_state(bidx1, a1_pos, box_states_new)
        p_a1_success = 1
        if bstate[0] == BoxState.OnGoalLoc:
          a1_pos_new = a1_init

      p_a2_success = 0
      bidx2 = get_box_idx(a2_pos)
      if bidx2 >= 0 and a2_pos in goals and a2_act == EventType.UNHOLD:
        _, bstate = update_dropped_box_state(bidx2, a2_pos, box_states_new)
        p_a2_success = 1
        if bstate[0] == BoxState.OnGoalLoc:
          a2_pos_new = a2_init

      if a1_act != EventType.UNHOLD:
        a1_pos_new = get_moved_coord(a1_pos, a1_act, box_states_new, True)
        if a1_pos_new != a1_pos:
          p_a1_success = P_MOVE

      if a2_act != EventType.UNHOLD:
        a2_pos_new = get_moved_coord(a2_pos, a2_act, box_states_new, True)
        if a2_pos_new != a2_pos:
          p_a2_success = P_MOVE

      p_ss = p_a1_success * p_a2_success
      p_sf = p_a1_success * (1 - p_a2_success)
      p_fs = (1 - p_a1_success) * p_a2_success
      p_ff = (1 - p_a1_success) * (1 - p_a2_success)
      if p_ss > 0:
        list_next_env.append((p_ss, box_states_new, a1_pos_new, a2_pos_new))
      if p_sf > 0:
        list_next_env.append((p_sf, box_states_new, a1_pos_new, a2_pos))
      if p_fs > 0:
        list_next_env.append((p_fs, box_states_new, a1_pos, a2_pos_new))
      if p_ff > 0:
        list_next_env.append((p_ff, box_states_new, a1_pos, a2_pos))
    # when both try to move
    else:
      agent_dist = (abs(a1_pos[0] - a2_pos[0]) + abs(a1_pos[1] - a2_pos[1]))
      if agent_dist > 2:
        p_a1_success = 0
        a1_pos_new = get_moved_coord(a1_pos, a1_act, box_states, True)
        if a1_pos_new != a1_pos:
          p_a1_success = P_MOVE

        p_a2_success = 0
        a2_pos_new = get_moved_coord(a2_pos, a2_act, box_states, True)
        if a2_pos_new != a2_pos:
          p_a2_success = P_MOVE

        p_ss = p_a1_success * p_a2_success
        p_sf = p_a1_success * (1 - p_a2_success)
        p_fs = (1 - p_a1_success) * p_a2_success
        p_ff = (1 - p_a1_success) * (1 - p_a2_success)
        if p_ss > 0:
          list_next_env.append((p_ss, box_states, a1_pos_new, a2_pos_new))
        if p_sf > 0:
          list_next_env.append((p_sf, box_states, a1_pos_new, a2_pos))
        if p_fs > 0:
          list_next_env.append((p_fs, box_states, a1_pos, a2_pos_new))
        if p_ff > 0:
          list_next_env.append((p_ff, box_states, a1_pos, a2_pos))
      elif agent_dist == 2:
        p_a1_success = 0
        a1_pos_new = get_moved_coord(a1_pos, a1_act, box_states, True)
        if a1_pos_new != a1_pos:
          p_a1_success = P_MOVE

        p_a2_success = 0
        a2_pos_new = get_moved_coord(a2_pos, a2_act, box_states, True)
        if a2_pos_new != a2_pos:
          p_a2_success = P_MOVE

        p_ss = p_a1_success * p_a2_success
        p_sf = p_a1_success * (1 - p_a2_success)
        p_fs = (1 - p_a1_success) * p_a2_success
        p_ff = (1 - p_a1_success) * (1 - p_a2_success)

        if a1_pos_new == a2_pos_new:
          if p_ss > 0:
            list_next_env.append((p_ss * 0.5, box_states, a1_pos_new, a2_pos))
            list_next_env.append((p_ss * 0.5, box_states, a1_pos, a2_pos_new))
        else:
          if p_ss > 0:
            list_next_env.append((p_ss, box_states, a1_pos_new, a2_pos_new))

        if p_sf > 0:
          list_next_env.append((p_sf, box_states, a1_pos_new, a2_pos))
        if p_fs > 0:
          list_next_env.append((p_fs, box_states, a1_pos, a2_pos_new))
        if p_ff > 0:
          list_next_env.append((p_ff, box_states, a1_pos, a2_pos))
      else:  # agent_dst == 1
        a1_pos_new = get_moved_coord(a1_pos, a1_act, box_states, True)
        a2_pos_new = get_moved_coord(a2_pos, a2_act, box_states, True)
        if a1_pos_new != a1_pos and a2_pos_new != a2_pos:
          if P_MOVE > 0:
            list_next_env.append(
                (P_MOVE * P_MOVE, box_states, a1_pos_new, a2_pos_new))
          if P_MOVE > 0 and P_MOVE < 1:
            list_next_env.append(
                (P_MOVE * (1 - P_MOVE), box_states, a1_pos_new, a2_pos))
            list_next_env.append(
                (P_MOVE * (1 - P_MOVE), box_states, a1_pos, a2_pos_new))
          if P_MOVE < 1:
            list_next_env.append(
                ((1 - P_MOVE) * (1 - P_MOVE), box_states, a1_pos, a2_pos))
        elif a1_pos_new != a1_pos:
          a2_pos_new2 = get_moved_coord(a2_pos, a2_act, None, True)
          if a2_pos_new2 == a1_pos:
            if P_MOVE > 0:
              list_next_env.append(
                  (P_MOVE * P_MOVE, box_states, a1_pos_new, a1_pos))
            if P_MOVE > 0 and P_MOVE < 1:
              list_next_env.append(
                  (P_MOVE * (1 - P_MOVE), box_states, a1_pos_new, a2_pos))
            if P_MOVE < 1:
              list_next_env.append(((1 - P_MOVE), box_states, a1_pos, a2_pos))
          else:
            if P_MOVE > 0:
              list_next_env.append((P_MOVE, box_states, a1_pos_new, a2_pos))
            if P_MOVE < 1:
              list_next_env.append((1 - P_MOVE, box_states, a1_pos, a2_pos))
        elif a2_pos_new != a2_pos:
          a1_pos_new2 = get_moved_coord(a1_pos, a1_act, None, True)
          if a1_pos_new2 == a2_pos:
            if P_MOVE > 0:
              list_next_env.append(
                  (P_MOVE * P_MOVE, box_states, a2_pos, a2_pos_new))
            if P_MOVE > 0 and P_MOVE < 1:
              list_next_env.append(
                  ((1 - P_MOVE) * P_MOVE, box_states, a1_pos, a2_pos_new))
            if P_MOVE < 1:
              list_next_env.append(((1 - P_MOVE), box_states, a1_pos, a2_pos))
          else:
            if P_MOVE > 0:
              list_next_env.append((P_MOVE, box_states, a1_pos, a2_pos_new))
            if P_MOVE < 1:
              list_next_env.append((1 - P_MOVE, box_states, a1_pos, a2_pos))
        else:
          list_next_env.append((1.0, box_states, a1_pos, a2_pos))
  # only a1 holds a box
  elif hold == "A1":
    box_states_new = list(box_states)
    p_a1_success = 0
    a1_pos_new = a1_pos
    if a1_act == EventType.UNHOLD:
      bidx = get_box_idx(a1_pos)
      assert bidx >= 0
      assert box_types[bidx] == 1
      _, bstate = update_dropped_box_state(bidx, a1_pos, box_states_new)
      p_a1_success = 1.0
      # respawn
      if bstate[0] == BoxState.OnGoalLoc:
        a1_pos_new = a1_init
    else:
      a1_pos_new = get_moved_coord(a1_pos, a1_act, box_states_new, True)
      if a1_pos_new != a1_pos:
        p_a1_success = P_MOVE

    a2_pos_new = a2_pos
    if a2_act == EventType.HOLD and a2_pos != a1_pos:
      bidx = get_box_idx(a2_pos)
      if bidx >= 0 and box_types[bidx] == 1:
        box_states_new[bidx] = conv_box_state_2_idx((BoxState.WithAgent2, None),
                                                    num_drops)
    else:
      a2_pos_new = get_moved_coord(a2_pos, a2_act, None, False)

    if p_a1_success > 0:
      list_next_env.append(
          (p_a1_success, box_states_new, a1_pos_new, a2_pos_new))
    if 1 - p_a1_success > 0:
      list_next_env.append(
          (1 - p_a1_success, box_states_new, a1_pos, a2_pos_new))
  # only a2 holds a box
  else:  # hold == "A2":
    box_states_new = list(box_states)
    p_a2_success = 0
    a2_pos_new = a2_pos
    if a2_act == EventType.UNHOLD:
      bidx = get_box_idx(a2_pos)
      assert bidx >= 0
      assert box_types[bidx] == 1
      _, bstate = update_dropped_box_state(bidx, a2_pos, box_states_new)
      p_a2_success = 1.0
      # respawn
      if bstate[0] == BoxState.OnGoalLoc:
        a2_pos_new = a2_init
    else:
      a2_pos_new = get_moved_coord(a2_pos, a2_act, box_states_new, True)
      if a2_pos_new != a2_pos:
        p_a2_success = P_MOVE

    a1_pos_new = a1_pos
    if a1_act == EventType.HOLD and a1_pos != a2_pos:
      bidx = get_box_idx(a1_pos)
      if bidx >= 0 and box_types[bidx] == 1:
        box_states_new[bidx] = conv_box_state_2_idx((BoxState.WithAgent1, None),
                                                    num_drops)
    else:
      a1_pos_new = get_moved_coord(a1_pos, a1_act, None, False)

    if p_a2_success > 0:
      list_next_env.append(
          (p_a2_success, box_states_new, a1_pos_new, a2_pos_new))
    if 1 - p_a2_success > 0:
      list_next_env.append(
          (1 - p_a2_success, box_states_new, a1_pos_new, a2_pos))

  return list_next_env
