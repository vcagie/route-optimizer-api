import json
import os
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler

from ortools.constraint_solver import pywrapcp, routing_enums_pb2

MAX_LOCATIONS = 10
SOLVE_TIME_LIMIT_SECONDS = 3
ORS_MATRIX_URL = "https://api.openrouteservice.org/v2/matrix/driving-car"
ORS_TIMEOUT_SECONDS = 10


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

    return mode, locations


def sum_route_distance_m(distance_m, order_indices, close_loop):
    total = 0.0
    for a, b in zip(order_indices, order_indices[1:]):
        total += distance_m[a][b]
    if close_loop:
        total += distance_m[order_indices[-1]][order_indices[0]]
    return total


def solve_round_trip(distance_m, duration_s):
    n = len(distance_m)

    manager = pywrapcp.RoutingIndexManager(n, 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    def duration_callback(from_index, to_index):
        return round(duration_s[manager.IndexToNode(from_index)][manager.IndexToNode(to_index)])

    transit_callback_index = routing.RegisterTransitCallback(duration_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.time_limit.FromSeconds(SOLVE_TIME_LIMIT_SECONDS)

    solution = routing.SolveWithParameters(params)
    if solution is None:
        raise RuntimeError("solver failed to find a solution")

    order_indices = []
    index = routing.Start(0)
    while not routing.IsEnd(index):
        order_indices.append(manager.IndexToNode(index))
        index = solution.Value(routing.NextVar(index))

    total_duration_s = solution.ObjectiveValue()
    total_distance_m = sum_route_distance_m(distance_m, order_indices, close_loop=True)
    return order_indices, total_distance_m, total_duration_s


def solve_one_way(distance_m, duration_s):
    n = len(distance_m)

    # Add a dummy "end" node with zero cost to/from every real node,
    # so the solver can terminate the path at whichever real node is optimal.
    dummy = n
    size = n + 1
    full_duration = [[0] * size for _ in range(size)]
    for i in range(n):
        for j in range(n):
            full_duration[i][j] = duration_s[i][j]

    manager = pywrapcp.RoutingIndexManager(size, 1, [0], [dummy])
    routing = pywrapcp.RoutingModel(manager)

    def duration_callback(from_index, to_index):
        return round(full_duration[manager.IndexToNode(from_index)][manager.IndexToNode(to_index)])

    transit_callback_index = routing.RegisterTransitCallback(duration_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.time_limit.FromSeconds(SOLVE_TIME_LIMIT_SECONDS)

    solution = routing.SolveWithParameters(params)
    if solution is None:
        raise RuntimeError("solver failed to find a solution")

    order_indices = []
    index = routing.Start(0)
    while not routing.IsEnd(index):
        node = manager.IndexToNode(index)
        if node != dummy:
            order_indices.append(node)
        index = solution.Value(routing.NextVar(index))

    total_duration_s = solution.ObjectiveValue()
    total_distance_m = sum_route_distance_m(distance_m, order_indices, close_loop=False)
    return order_indices, total_distance_m, total_duration_s


def optimize(mode, locations):
    distance_m, duration_s = fetch_distance_duration_matrix(locations)

    if mode == "round-trip":
        order_indices, total_distance_m, total_duration_s = solve_round_trip(distance_m, duration_s)
    else:
        order_indices, total_distance_m, total_duration_s = solve_one_way(distance_m, duration_s)

    order_ids = [locations[i]["id"] for i in order_indices]
    return order_ids, round(total_distance_m / 1000, 3), round(total_duration_s / 60, 2)


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
            order, total_distance_km, total_duration_min = optimize(mode, locations)
        except RuntimeError as e:
            self._send_json(502, {"error": str(e)})
            return

        self._send_json(200, {
            "mode": mode,
            "order": order,
            "total_distance_km": total_distance_km,
            "total_duration_min": total_duration_min,
        })
