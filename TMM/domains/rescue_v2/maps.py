from TMM.domains.rescue_v2 import (Route, Location, E_Type, Work, Place,
                                   PlaceName)

MAP_RESCUE = {
    "name":
    "rescue_3",
    "places": [
        Place(PlaceName.Fire_stateion, (0.312 - 0.05, 0.632)),
        Place(PlaceName.Police_station, (0.763 - 0.05, 0.588)),
        Place(PlaceName.Hospital, (0.430 - 0.05, 0.322)),
        Place(PlaceName.Campsite, (0.343 - 0.05, 0.134), helps=1),
        Place(PlaceName.City_hall, (0.447 - 0.05, 0.952), helps=2),
        Place(PlaceName.Mall, (0.932 - 0.05, 0.676), helps=2),
        Place(PlaceName.Intersection1, (0.268 - 0.05, 0.524), visible=False),
        Place(PlaceName.Intersection2, (0.10 - 0.05, 0.571), visible=False),
        Place(PlaceName.Intersection3, (0.629 - 0.05, 0.696), visible=False),
        Place(PlaceName.Intersection4, (0.696 - 0.05, 0.912), visible=False),
        Place(PlaceName.Intersection5, (0.591 - 0.05, 0.359), visible=False),
    ],
    "routes": [
        Route(start=3,
              end=7,
              length=2,
              coords=[(0.201 - 0.05, 0.248), (0.110 - 0.05, 0.373)]),
        Route(start=7,
              end=4,
              length=2,
              coords=[(0.150 - 0.05, 0.761), (0.275 - 0.05, 0.885)]),
        Route(start=9, end=5, length=1, coords=[(0.837 - 0.05, 0.824)]),
        Route(start=5,
              end=10,
              length=2,
              coords=[(0.921 - 0.05, 0.427), (0.763 - 0.05, 0.282)]),
        Route(start=0, end=8, length=1, coords=[(0.467 - 0.05, 0.737)]),
        Route(start=1, end=10, length=1, coords=[(0.720 - 0.05, 0.443)]),
        Route(start=2, end=6, length=1, coords=[(0.322 - 0.05, 0.410)]),
    ],
    "connections": {
        0: [(E_Type.Place, 6), (E_Type.Route, 4)],
        1: [(E_Type.Place, 8), (E_Type.Route, 5)],
        2: [(E_Type.Place, 10), (E_Type.Route, 6)],
        3: [(E_Type.Route, 0)],
        4: [(E_Type.Route, 1), (E_Type.Place, 9)],
        5: [(E_Type.Route, 2), (E_Type.Route, 3)],
        6: [(E_Type.Place, 7), (E_Type.Place, 0), (E_Type.Route, 6)],
        7: [(E_Type.Route, 0), (E_Type.Route, 1), (E_Type.Place, 6)],
        8: [(E_Type.Route, 4), (E_Type.Place, 9), (E_Type.Place, 1)],
        9: [(E_Type.Place, 4), (E_Type.Route, 2), (E_Type.Place, 8)],
        10: [(E_Type.Place, 2), (E_Type.Route, 5), (E_Type.Route, 3)],
    },
    "work_locations": [
        Location(E_Type.Place, id=3),
        Location(E_Type.Place, id=4),
        Location(E_Type.Place, id=5),
    ],
    "work_info": [
        Work(workload=1, rescue_place=3),
        Work(workload=2, rescue_place=4),
        Work(workload=2, rescue_place=5),
    ],
    "a1_init":
    Location(E_Type.Place, 1),
    "a2_init":
    Location(E_Type.Place, 0),
    "a3_init":
    Location(E_Type.Place, 2),
}
