from career_graph_builder import build_career_graph
from role_builder import normalize_role
from skill_matching import ROLE_SKILLS

CAREER_GRAPH = build_career_graph(ROLE_SKILLS)

def get_next_roles(role):
    role = normalize_role(role) or role.strip()

    if role in CAREER_GRAPH:
        return CAREER_GRAPH[role]

    return []
print("Total career transitions:", len(CAREER_GRAPH))
#print(CAREER_GRAPH)
if __name__ == "__main__":

    for role in list(CAREER_GRAPH.keys())[:10]:
        print(role, "→", CAREER_GRAPH[role])
