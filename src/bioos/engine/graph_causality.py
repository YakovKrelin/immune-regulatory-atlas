from bioos.models import Assertion


def get_neighborhood(assertions: list[Assertion], start_node: str, max_hops: int = 2) -> dict:
    if not start_node:
        return {"nodes": [], "edges": []}
    visited_nodes = [start_node]
    visited_set = {start_node}
    frontier = {start_node}
    selected_edges: dict[str, Assertion] = {}
    for _ in range(max_hops):
        next_frontier = set()
        for a in assertions:
            if a.subject in frontier or a.object in frontier:
                selected_edges[a.id] = a
                for n in (a.subject, a.object):
                    if n not in visited_set:
                        visited_set.add(n)
                        visited_nodes.append(n)
                        next_frontier.add(n)
        frontier = next_frontier
        if not frontier:
            break
    return {"nodes": visited_nodes, "edges": [e.model_dump() for e in selected_edges.values()]}


def analyze_why_elevated(assertions: list[Assertion], target_entity: str) -> dict:
    direct = [a for a in assertions if a.object == target_entity and a.direction == "up"]
    upstream = [a for a in assertions if any(d.subject == a.object and a.direction == "up" for d in direct)]
    return {
        "direct_causes": [a.model_dump() for a in direct],
        "upstream_causes": [a.model_dump() for a in upstream],
    }


def simulate_perturbation(assertions: list[Assertion], target_entity: str, action: str = "suppress") -> dict:
    trade_offs = (
        ["CRP down", "STAT3 signaling down", "Th17 differentiation down", "Potential infection-defense trade-off"]
        if target_entity == "IL6" and action == "suppress"
        else ["Depends on cell type, dose, and exposure duration"]
    )
    return {
        "target_entity": target_entity,
        "action": action,
        "resulting_direction": "down" if action == "suppress" else "up",
        "trade_offs": trade_offs,
        "disclaimer": "Graph modeling reflects known assertions and context limits; not a clinical recommendation.",
    }