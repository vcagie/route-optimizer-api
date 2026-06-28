import json
import os
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler

from ortools.constraint_solver import pywrapcp, routing_enums_pb2

MAX_LOCATIONS = 10
ORS_MATRIX_URL = "https://api.openrouteservice.org/v2/matrix/driving-car"
ORS_TIMEOUT_SECONDS = 10
TIME_HORIZON_S = 24 * 3600
TIME_WINDOW_SOLVE_TIME_LIMIT_SECONDS = 2


class InfeasibleError(Exception):
    pass


def fetch_distance_duration_matrix(locations):
    api_key = os.environ.get("ORS_API_KEY")
    if not api_key:
        raise RuntimeError("ORS_API_KEY environment variable is not set")

    body = json.dumps({
        "locations": [[loc["lng"], loc["lat"]] for loc in locations],
        "metrics": ["distance", "duration"],
        "units": "m",
    }).encode("utf-8")

    req = urllib.request.Request(
        ORS_MATRIX_URL,
        data=body,
        method="POST",
        headers={
            "Authorization": api_key,
            "Content-Type": "application/json",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=ORS_TIMEOUT_SECONDS) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"routing API error: {e.code} {e.read().decode(errors='replace')}")
    except urllib.error.URLError as e:
        raise RuntimeError(f"routing API unreachable: {e.reason}")

    return data["distances"], data["durations"]


def validate_request(body):
    if not isinstance(body, dict):
        raise ValueError("Request body must be a JSON object")

    mode = body.get("mode", "round-trip")
    if mode not in ("round-trip", "one-way"):
        raise ValueError('mode must be "round-trip" or "one-way"')

    locations = body.get("locations")
    if not isinstance(locations, list) or len(locations) < 2:
        raise ValueError("locations must be a list with at least 2 items")
    if len(locations) > MAX_LOCATIONS:
        raise ValueError(f"locations must contain at most {MAX_LOCATIONS} items")

    seen_ids = set()
    for loc in locations:
        if not isinstance(loc, dict) or "id" not in loc or "lat" not in loc or "lng" not in loc:
            raise ValueError("each location must have id, lat, lng")
        if loc["id"] in seen_ids:
            raise ValueError(f"duplicate location id: {loc['id']}")
        seen_ids.add(loc["id"])
        lat, lng = loc["lat"], loc["lng"]
        if not isinstance(lat, (int, float)) or not (-90 <= lat <= 90):
            raise ValueError(f"invalid lat for id {loc['id']}")
        if not isinstance(lng, (int, float)) or not (-180 <= lng <= 180):
            raise ValueError(f"invalid lng for id {loc['id']}")

        earliest = loc.get("earliest", 0)
        latest = loc.get("latest", TIME_HORIZON_S)
        service_s = loc.get("service_s", 0)
        if not isinstance(earliest, int) or isinstance(earliest, bool) or earliest < 0:
            raise ValueError(f"invalid earliest for id {loc['id']}")
        if not isinstance(latest, int) or isinstance(latest, bool) or latest < earliest:
            raise ValueError(f"invalid latest for id {loc['id']} (must be >= earliest)")
        if not isinstance(service_s, int) or isinstance(service_s, bool) or service_s < 0:
            raise ValueError(f"invalid service_s for id {loc['id']}")

    return mode, locations


def sum_route_distance_m(distance_m, order_indices, close_loop):
    total = 0.0
    for a, b in zip(order_indices, order_indices[1:]):
        total += distance_m[a][b]
    if close_loop:
        total += distance_m[order_indices[-1]][order_indices[0]]
    return total


def _add_time_dimension(routing, manager, duration_s, service_s_list, time_windows, real_node_count):
    def time_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return round(duration_s[from_node][to_node]) + service_s_list[from_node]

    time_callback_index = routing.RegisterTransitCallback(time_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(time_callback_index)

    routing.AddDimension(time_callback_index, TIME_HORIZON_S, TIME_HORIZON_S, False, "Time")
    time_dimension = routing.GetDimensionOrDie("Time")

    for node in range(real_node_count):
        earliest, latest = time_windows[node]
        time_dimension.CumulVar(manager.NodeToIndex(node)).SetRange(earliest, latest)

    return time_dimension


def _extract_time_window_params(routing, manager, time_dimension, solution, order_indices):
    return [solution.Value(time_dimension.CumulVar(manager.NodeToIndex(node))) for node in order_indices]


def solve_round_trip(distance_m, duration_s, service_s_list=None, time_windows=None):
    n = len(distance_m)
    has_time_windows = time_windows is not None

    manager = pywrapcp.RoutingIndexManager(n, 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    params = pywrapcp.DefaultRoutingSearchParameters()

    if has_time_windows:
        time_dimension = _add_time_dimension(routing, manager, duration_s, service_s_list, time_windows, n)
        time_dimension.CumulVar(routing.Start(0)).SetRange(0, 0)
        routing.AddVariableMinimizedByFinalizer(time_dimension.CumulVar(routing.End(0)))

        params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
        params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        params.time_limit.FromSeconds(TIME_WINDOW_SOLVE_TIME_LIMIT_SECONDS)
    else:
        def duration_callback(from_index, to_index):
            return round(duration_s[manager.IndexToNode(from_index)][manager.IndexToNode(to_index)])

        transit_callback_index = routing.RegisterTransitCallback(duration_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
        params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC

    solution = routing.SolveWithParameters(params)
    if solution is None:
        if has_time_windows:
            raise InfeasibleError("no route satisfies the given time windows")
        raise RuntimeError("solver failed to find a solution")

    order_indices = []
    index = routing.Start(0)
    while not routing.IsEnd(index):
        order_indices.append(manager.IndexToNode(index))
        index = solution.Value(routing.NextVar(index))

    total_distance_m = sum_route_distance_m(distance_m, order_indices, close_loop=True)

    if has_time_windows:
        arrivals_s = _extract_time_window_params(routing, manager, time_dimension, solution, order_indices)
        total_duration_s = solution.Value(time_dimension.CumulVar(routing.End(0)))
        return order_indices, total_distance_m, total_duration_s, arrivals_s

    total_duration_s = solution.ObjectiveValue()
    return order_indices, total_distance_m, total_duration_s, None


def solve_one_way(distance_m, duration_s, service_s_list=None, time_windows=None):
    n = len(distance_m)
    has_time_windows = time_windows is not None

    # Add a dummy "end" node with zero cost to/from every real node,
    # so the solver can terminate the path at whichever real node is optimal.
    dummy = n
    size = n + 1
    full_duration = [[0] * size for _ in range(size)]
    for i in range(n):
        for j in range(n):
            full_duration[i][j] = duration_s[i][j]

    full_service_s_list = None
    if has_time_windows:
        full_service_s_list = service_s_list + [0]

    manager = pywrapcp.RoutingIndexManager(size, 1, [0], [dummy])
    routing = pywrapcp.RoutingModel(manager)

    params = pywrapcp.DefaultRoutingSearchParameters()

    if has_time_windows:
        time_dimension = _add_time_dimension(routing, manager, full_duration, full_service_s_list, time_windows, n)
        time_dimension.CumulVar(routing.Start(0)).SetRange(0, 0)
        routing.AddVariableMinimizedByFinalizer(time_dimension.CumulVar(routing.End(0)))

        params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
        params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        params.time_limit.FromSeconds(TIME_WINDOW_SOLVE_TIME_LIMIT_SECONDS)
    else:
        def duration_callback(from_index, to_index):
            return round(full_duration[manager.IndexToNode(from_index)][manager.IndexToNode(to_index)])

        transit_callback_index = routing.RegisterTransitCallback(duration_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
        params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC

    solution = routing.SolveWithParameters(params)
    if solution is None:
        if has_time_windows:
            raise InfeasibleError("no route satisfies the given time windows")
        raise RuntimeError("solver failed to find a solution")

    order_indices = []
    index = routing.Start(0)
    while not routing.IsEnd(index):
        node = manager.IndexToNode(index)
        if node != dummy:
            order_indices.append(node)
        index = solution.Value(routing.NextVar(index))

    total_distance_m = sum_route_distance_m(distance_m, order_indices, close_loop=False)

    if has_time_windows:
        arrivals_s = _extract_time_window_params(routing, manager, time_dimension, solution, order_indices)
        total_duration_s = solution.Value(time_dimension.CumulVar(routing.End(0)))
        return order_indices, total_distance_m, total_duration_s, arrivals_s

    total_duration_s = solution.ObjectiveValue()
    return order_indices, total_distance_m, total_duration_s, None


def optimize(mode, locations):
    distance_m, duration_s = fetch_distance_duration_matrix(locations)

    has_time_windows = any("earliest" in loc or "latest" in loc or "service_s" in loc for loc in locations)
    service_s_list = None
    time_windows = None
    if has_time_windows:
        service_s_list = [loc.get("service_s", 0) for loc in locations]
        time_windows = [(loc.get("earliest", 0), loc.get("latest", TIME_HORIZON_S)) for loc in locations]

    if mode == "round-trip":
        order_indices, total_distance_m, total_duration_s, arrivals_s = solve_round_trip(
            distance_m, duration_s, service_s_list, time_windows)
    else:
        order_indices, total_distance_m, total_duration_s, arrivals_s = solve_one_way(
            distance_m, duration_s, service_s_list, time_windows)

    order_ids = [locations[i]["id"] for i in order_indices]
    return order_ids, round(total_distance_m / 1000, 3), round(total_duration_s / 60, 2), arrivals_s


class handler(BaseHTTPRequestHandler):
    def _set_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        self.send_response(204)
        self._set_cors_headers()
        self.end_headers()

    def _send_json(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self._set_cors_headers()
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            raw_body = self.rfile.read(content_length) if content_length > 0 else b"{}"
            body = json.loads(raw_body)
        except (ValueError, json.JSONDecodeError):
            self._send_json(400, {"error": "invalid JSON body"})
            return

        try:
            mode, locations = validate_request(body)
        except ValueError as e:
            self._send_json(400, {"error": str(e)})
            return

        try:
            order, total_distance_km, total_duration_min, arrivals_s = optimize(mode, locations)
        except InfeasibleError as e:
            self._send_json(422, {"error": str(e)})
            return
        except RuntimeError as e:
            self._send_json(502, {"error": str(e)})
            return

        self._send_json(200, {
            "mode": mode,
            "order": order,
            "total_distance_km": total_distance_km,
            "total_duration_min": total_duration_min,
            "arrivals_s": arrivals_s,
        })
