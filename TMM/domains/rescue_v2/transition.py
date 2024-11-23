from typing import Sequence, Mapping, Union
from TMM.domains.rescue_v2 import (Route, Location, E_Type, Work, E_EventType,
                                   T_Connections)


def find_location_index(list_locations: Sequence[Location], location: Location):
  for idx, item in enumerate(list_locations):
    if location.type == E_Type.Place:
      if item.type == location.type and item.id == location.id:
        return idx
    else:  # location.type == E_Type.Route
      if (item.type == location.type and item.id == location.id
          and item.index == location.index):
        return idx

  return None


def transition(work_states: Sequence[int], a1_location: Location,
               a2_location: Location, a3_location: Location,
               a1_action: E_EventType, a2_action: E_EventType,
               a3_action: E_EventType, routes: Sequence[Route],
               connections: Mapping[int, T_Connections],
               work_locations: Sequence[Location], work_info: Sequence[Work]):
  list_next_env = []
  i_work1 = find_location_index(work_locations, a1_location)
  i_work2 = find_location_index(work_locations, a2_location)
  i_work3 = find_location_index(work_locations, a3_location)

  # both are at the same place where a work exists.
  new_work_states = list(work_states)
  if a1_location == a2_location:
    if i_work1 is not None and work_states[i_work1] != 0:  # work not done
      if a1_action == E_EventType.Rescue and a2_action == E_EventType.Rescue:
        if work_info[i_work1].workload <= 2:
          new_work_states[i_work1] -= 1  # work done

  if a1_location == a3_location:
    if i_work1 is not None and new_work_states[i_work1] != 0:  # work not done
      if a1_action == E_EventType.Rescue and a3_action == E_EventType.Rescue:
        if work_info[i_work1].workload <= 2:
          new_work_states[i_work1] -= 1  # work done

  if a2_location == a3_location:
    if i_work2 is not None and new_work_states[i_work2] != 0:  # work not done
      if a2_action == E_EventType.Rescue and a3_action == E_EventType.Rescue:
        if work_info[i_work2].workload <= 2:
          new_work_states[i_work2] -= 1  # work done

  def get_updated_state(work_states: Sequence[int], agent_location: Location,
                        action: E_EventType, i_work: Union[int, None]):
    new_work_states = list(work_states)
    new_agent_location = agent_location
    if action == E_EventType.Rescue:
      if i_work is not None and work_states[i_work] != 0:  # work not done
        if work_info[i_work].workload <= 1:
          new_work_states[i_work] -= 1
    else:  # action is not Stay --> move to a new location
      if agent_location.type == E_Type.Place:
        place_id = agent_location.id
        if action.value < len(connections[place_id]):
          new_place = connections[place_id][action.value]
          if new_place[0] == E_Type.Route:
            route_id = new_place[1]
            if routes[route_id].start == place_id:
              new_agent_location = Location(E_Type.Route, route_id, 0)
            else:  # route[route_id].end == place_name
              new_agent_location = Location(E_Type.Route, route_id,
                                            routes[route_id].length - 1)
          else:  # new_place[0] == E_Type.place
            new_agent_location = Location(E_Type.Place, new_place[1])
      else:
        if action.value < 2:
          route_id = agent_location.id
          if action == E_EventType.Option0:  # move toward the end
            new_idx = agent_location.index + 1
            if new_idx < routes[route_id].length:
              new_agent_location = Location(E_Type.Route, route_id, new_idx)
            else:
              new_agent_location = Location(E_Type.Place, routes[route_id].end)
          elif action == E_EventType.Option1:  # move toward the start
            new_idx = agent_location.index - 1
            if new_idx >= 0:
              new_agent_location = Location(E_Type.Route, route_id, new_idx)
            else:
              new_agent_location = Location(E_Type.Place,
                                            routes[route_id].start)
    return new_work_states, new_agent_location

  new_work_states, new_a1_location = get_updated_state(new_work_states,
                                                       a1_location, a1_action,
                                                       i_work1)
  new_work_states, new_a2_location = get_updated_state(new_work_states,
                                                       a2_location, a2_action,
                                                       i_work2)
  new_work_states, new_a3_location = get_updated_state(new_work_states,
                                                       a3_location, a3_action,
                                                       i_work3)

  list_next_env.append(
      (1.0, new_work_states, new_a1_location, new_a2_location, new_a3_location))
  return list_next_env
