from collections import defaultdict
from role_builder import canonicalize_role_name

def build_career_graph(role_skills):

    career_graph = defaultdict(list)

    roles = list(role_skills.keys())

    for role_a in roles:

        skills_a = set(role_skills[role_a])

        for role_b in roles:

            if role_a == role_b:
                continue

            skills_b = set(role_skills[role_b])
            canonical_a = canonicalize_role_name(role_a)
            canonical_b = canonicalize_role_name(role_b)

            if canonical_a == canonical_b:
                continue

            overlap = len(skills_a & skills_b) / max(len(skills_a), 1)

            if overlap >= 0.5 and len(skills_b) > len(skills_a):

                career_graph[role_a].append(role_b)

    return career_graph
