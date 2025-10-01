# src/core/solvers/cp.py

from ortools.sat.python import cp_model


def solve_puzzle_cp(puzzle: dict) -> list[tuple[int, int]] | None:
    """
    Solves a Zip puzzle using the CP-SAT constraint solver.
    This implementation uses the "dummy node" technique to model a Hamiltonian
    path problem using a circuit constraint.
    """
    grid = puzzle["grid"]
    walls = puzzle.get("walls", set())
    blocked_cells = puzzle.get("blocked_cells", set())
    height = len(grid)
    width = len(grid[0])

    model = cp_model.CpModel()

    # --- 1. Create variables ---
    # Map (r,c) coordinates to an integer node index
    node_map = {}
    rev_node_map = {}
    for r in range(height):
        for c in range(width):
            if (r, c) not in blocked_cells:
                node_idx = len(node_map)
                node_map[(r, c)] = node_idx
                rev_node_map[node_idx] = (r, c)

    num_real_nodes = len(node_map)
    if num_real_nodes == 0:
        return []

    dummy_node = num_real_nodes

    # Create a boolean literal for each possible directed arc u -> v
    arcs = []
    literal_map = {}
    for r_from in range(height):
        for c_from in range(width):
            pos_from = (r_from, c_from)
            if pos_from not in node_map:
                continue
            u = node_map[pos_from]

            # Arcs to adjacent real nodes
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                r_to, c_to = r_from + dr, c_from + dc
                pos_to = (r_to, c_to)
                if pos_to in node_map:
                    wall_pair = tuple(sorted((pos_from, pos_to)))
                    if wall_pair not in walls:
                        v = node_map[pos_to]
                        literal = model.NewBoolVar(f"arc_{u}_{v}")
                        literal_map[(u, v)] = literal
                        arcs.append([u, v, literal])

    # Arcs to and from the dummy node, representing start/end of the path
    for i in range(num_real_nodes):
        # Arc from dummy to real node (start of path)
        literal_start = model.NewBoolVar(f"start_{i}")
        literal_map[(dummy_node, i)] = literal_start
        arcs.append([dummy_node, i, literal_start])
        # Arc from real node to dummy (end of path)
        literal_end = model.NewBoolVar(f"end_{i}")
        literal_map[(i, dummy_node)] = literal_end
        arcs.append([i, dummy_node, literal_end])

    # --- 2. Add constraints ---
    model.AddCircuit(arcs)

    # Waypoint ordering constraints
    num_map = puzzle["num_map"]

    ranks = {
        i: model.NewIntVar(0, num_real_nodes - 1, f"rank_{i}")
        for i in range(num_real_nodes)
    }
    model.AddAllDifferent(list(ranks.values()))

    # Link ranks to path choices
    for u, v, literal in arcs:
        if u != dummy_node and v != dummy_node:  # Real-to-real arcs
            model.Add(ranks[v] == ranks[u] + 1).OnlyEnforceIf(literal)
        elif u == dummy_node:  # Dummy-to-real arcs (path start)
            model.Add(ranks[v] == 0).OnlyEnforceIf(literal)
        elif v == dummy_node:  # Real-to-dummy arcs (path end)
            model.Add(ranks[u] == num_real_nodes - 1).OnlyEnforceIf(literal)

    # Enforce waypoint order
    if num_map:
        sorted_waypoints = sorted(num_map.keys())
        for i in range(len(sorted_waypoints) - 1):
            p1_pos = num_map[sorted_waypoints[i]]
            p2_pos = num_map[sorted_waypoints[i + 1]]
            if p1_pos in node_map and p2_pos in node_map:
                model.Add(ranks[node_map[p2_pos]] > ranks[node_map[p1_pos]])

        if 1 in num_map and num_map[1] in node_map:
            model.Add(ranks[node_map[num_map[1]]] == 0)

    # --- 3. Solve the model ---
    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    # --- 4. Reconstruct the path ---
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        start_node = -1
        for i in range(num_real_nodes):
            if solver.Value(ranks[i]) == 0:
                start_node = i
                break
        if start_node == -1:
            return None

        successors = {}
        for (u, v), literal in literal_map.items():
            if u != dummy_node and v != dummy_node and solver.Value(literal):
                successors[u] = v

        solution_path = []
        current_node = start_node
        while current_node in successors:
            solution_path.append(rev_node_map[current_node])
            current_node = successors[current_node]
        solution_path.append(rev_node_map[current_node])  # Append the last node

        if len(solution_path) == num_real_nodes:
            return solution_path

    return None
