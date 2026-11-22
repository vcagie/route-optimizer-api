import json
import os
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler

from ortools.constraint_solver import pywrapcp, routing_enums_pb2

MAX_LOCATIONS = 25
# Held-Karp is O(n^2 * 2^n): 15 stops solves in ~0.25s in CPython, and every
# extra stop roughly doubles that. Above the cutoff, fall back to OR-Tools with
# a time-bounded metaheuristic search.
EXACT_SOLVER_MAX_LOCATIONS = 15
HEURISTIC_TIME_LIMIT_SECONDS = 2
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

    distances, durations = data["distances"], data["durations"]
    # ORS returns null for pairs it cannot route between (e.g. across an ocean).
    if any(v is None for row in durations for v in row):
        raise RuntimeError("no drivable route exists between some of these stops")
    return distances, durations


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

    start_id = body.get("start_id")
    if start_id is None:
        start_id = locations[0]["id"]
    elif start_id not in seen_ids:
        raise ValueError("start_id must match one of the location ids")

    end_id = body.get("end_id")
    if mode == "round-trip":
        end_id = None  # a round trip always ends where it started
    elif end_id is not None:
        if end_id not in seen_ids:
            raise ValueError("end_id must match one of the location ids")
        if end_id == start_id:
            raise ValueError("end_id must differ from start_id for a one-way route")

    return mode, locations, start_id, end_id


def route_total(matrix, order, close_loop):
    total = sum(matrix[a][b] for a, b in zip(order, order[1:]))
    if close_loop:
        total += matrix[order[-1]][order[0]]
    return total


def solve_exact(cost, start, end, close_loop):
    """Held-Karp dynamic programming: returns the provably optimal visiting order.

    dp[mask][j] is the cheapest path that leaves `start`, visits exactly the
    nodes in `mask`, and currently sits at node j.
    """
    n = len(cost)
    inf = float("inf")
    full = (1 << n) - 1
    dp = [[inf] * n for _ in range(1 << n)]
    parent = [[-1] * n for _ in range(1 << n)]
    dp[1 << start][start] = 0.0

    for mask in range(1 << n):
        if not mask & (1 << start):
            continue
        row = dp[mask]
        for j in range(n):
            base = row[j]
            if base == inf:
                continue
            cost_from_j = cost[j]
            for k in range(n):
                bit = 1 << k
                if mask & bit:
                    continue
                next_mask = mask | bit
                # A pinned end may only be entered as the very last stop.
                if k == end and next_mask != full:
                    continue
                candidate = base + cost_from_j[k]
                if candidate < dp[next_mask][k]:
                    dp[next_mask][k] = candidate
                    parent[next_mask][k] = j

    if end is not None:
        last = end
    else:
        def final_cost(j):
            return dp[full][j] + (cost[j][start] if close_loop else 0)
        last = min((j for j in range(n) if j != start), key=final_cost)

    order = []
    mask, node = full, last
    while node != -1:
        order.append(node)
        prev = parent[mask][node]
        mask &= ~(1 << node)
        node = prev
    order.reverse()
    return order


def solve_heuristic(cost, start, end, close_loop):
    """OR-Tools routing with guided local search for routes too large to solve exactly."""
    n = len(cost)

    if close_loop:
        size = n
        manager = pywrapcp.RoutingIndexManager(n, 1, start)
    elif end is not None:
        size = n
        manager = pywrapcp.RoutingIndexManager(n, 1, [start], [end])
    else:
        # Add a dummy "end" node with zero cost to/from every real node,
        # so the solver can terminate the path at whichever real node is optimal.
        size = n + 1
        manager = pywrapcp.RoutingIndexManager(size, 1, [start], [n])

    int_cost = [[0] * size for _ in range(size)]
    for i in range(n):
        for j in range(n):
            int_cost[i][j] = round(cost[i][j])

    routing = pywrapcp.RoutingModel(manager)

    def cost_callback(from_index, to_index):
        return int_cost[manager.IndexToNode(from_index)][manager.IndexToNode(to_index)]

    transit_callback_index = routing.RegisterTransitCallback(cost_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.time_limit.FromSeconds(HEURISTIC_TIME_LIMIT_SECONDS)

    solution = routing.SolveWithParameters(params)
    if solution is None:
        raise RuntimeError("solver failed to find a solution")

    order = []
    index = routing.Start(0)
    while not routing.IsEnd(index):
        order.append(manager.IndexToNode(index))
        index = solution.Value(routing.NextVar(index))
    if end is not None:
        order.append(end)  # the pinned end node is the route's end index, not visited in the loop
    return order


def baseline_order(n, start, end):
    """The user's own order, adjusted only to respect the pinned start/end."""
    order = [start] + [i for i in range(n) if i not in (start, end)]
    if end is not None:
        order.append(end)
    return order


def build_legs(locations, order, distance_m, duration_s, close_loop):
    pairs = list(zip(order, order[1:]))
    if close_loop:
        pairs.append((order[-1], order[0]))
    return [
        {
            "from_id": locations[a]["id"],
            "to_id": locations[b]["id"],
            "distance_m": round(distance_m[a][b], 1),
            "duration_s": round(duration_s[a][b], 1),
        }
        for a, b in pairs
    ]


def optimize(mode, locations, start_id, end_id):
    distance_m, duration_s = fetch_distance_duration_matrix(locations)

    index_of = {loc["id"]: i for i, loc in enumerate(locations)}
    start = index_of[start_id]
    end = index_of[end_id] if end_id is not None else None
    close_loop = mode == "round-trip"

    # Optimize for drive time; distance is reported alongside.
    is_exact = len(locations) <= EXACT_SOLVER_MAX_LOCATIONS
    solver = solve_exact if is_exact else solve_heuristic
    order = solver(duration_s, start, end, close_loop)
    baseline = baseline_order(len(locations), start, end)

    return {
        "mode": mode,
        "order": [locations[i]["id"] for i in order],
        "optimal": is_exact,
        "solver": "held-karp" if is_exact else "or-tools-guided-local-search",
        "total_distance_m": round(route_total(distance_m, order, close_loop), 1),
        "total_duration_s": round(route_total(duration_s, order, close_loop), 1),
        # Kept for backward compatibility with earlier clients.
        "total_distance_km": round(route_total(distance_m, order, close_loop) / 1000, 3),
        "total_duration_min": round(route_total(duration_s, order, close_loop) / 60, 2),
        "legs": build_legs(locations, order, distance_m, duration_s, close_loop),
        "baseline": {
            "order": [locations[i]["id"] for i in baseline],
            "total_distance_m": round(route_total(distance_m, baseline, close_loop), 1),
            "total_duration_s": round(route_total(duration_s, baseline, close_loop), 1),
        },
    }


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
            mode, locations, start_id, end_id = validate_request(body)
        except ValueError as e:
            self._send_json(400, {"error": str(e)})
            return

        try:
            result = optimize(mode, locations, start_id, end_id)
        except RuntimeError as e:
            self._send_json(502, {"error": str(e)})
            return

        self._send_json(200, result)
