from typing import Union, Sequence, Tuple
from enum import Enum
from dataclasses import dataclass, field
from TMM.models.mdp import ActionSpace

T_RouteId = int
T_PlaceId = int


class PlaceName:
  Fire_stateion = "Fire Station"
  City_hall = "City Hall"
  Police_station = "Police Station"
  Bridge_1 = "Bridge 1"
  Campsite = "Campsite"
  Mall = "Mall"
  Bridge_2 = "Bridge 2"
  Hospital = "Hospital"
  Intersection1 = "Intersection 1"
  Intersection2 = "Intersection 2"
  Intersection3 = "Intersection 3"
  Intersection4 = "Intersection 4"
  Intersection5 = "Intersection 5"


class E_EventType(Enum):
  Option0 = 0
  Option1 = 1
  Option2 = 2
  Option3 = 3
  Stay = 4
  Rescue = 5
  Set_Latent = 100


AGENT_ACTIONSPACE = ActionSpace([E_EventType(idx) for idx in range(6)])


class E_Type(Enum):
  Place = 0
  Route = 1


T_Connections = Sequence[Tuple[E_Type, int]]


@dataclass
class Location:
  type: E_Type
  id: Union[T_PlaceId, T_RouteId]
  index: int = 0

  def __repr__(self) -> str:
    return f"{self.type.name}, {self.id}, {self.index}"

  def __eq__(self, other) -> bool:
    if self.type == E_Type.Route:
      return (self.type == other.type and self.id == other.id
              and self.index == other.index)
    else:
      return (self.type == other.type and self.id == other.id)

  def __hash__(self) -> int:
    return hash(repr(self))

  @classmethod
  def from_str(cls, str_loc: str):
    list_loc = str_loc.split(", ")
    e_type = E_Type[list_loc[0]]
    id_place = int(list_loc[1])
    index = int(list_loc[2])

    return Location(e_type, id_place, index)


@dataclass
class Route:
  start: T_PlaceId
  end: T_PlaceId
  length: int
  coords: Sequence[Tuple[float, float]] = field(default_factory=list)


@dataclass
class Work:
  workload: int
  rescue_place: T_PlaceId
  coupled_works: Sequence = field(default_factory=list)


@dataclass
class Place:
  name: str
  coord: Tuple[float, float]
  helps: int = 0
  visible: bool = True


def is_work_done(widx, work_states: Sequence[int], couples: Sequence):
  state = work_states[widx]
  if state == 0:
    return True
  else:
    for couple in couples:
      if work_states[couple] == 0:
        return True

  return False
