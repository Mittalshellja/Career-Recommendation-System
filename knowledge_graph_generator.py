# ==========================================================
# FINAL KG RECOMMENDER V2
# Clean Recommendations + Threshold + Graph Export
#
# FILE NAME:
# knowledge_graph_recommender_v2.py
# ==========================================================

import pandas as pd
import networkx as nx
from pathlib import Path

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:
    SentenceTransformer = None
    cosine_similarity = None

from skill_ontology import SKILL_ONTOLOGY

# ==========================================================
# PATHS
# ==========================================================
BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "archive" / "job_dataset.csv"
GRAPH_PATH = BASE_DIR / "career_graph.gexf"
SUMMARY_PATH = BASE_DIR / "career_graph_summary.csv"
MODEL_NAME = "all-MiniLM-L6-v2"

# ==========================================================
# LOAD MODEL
# ==========================================================
def load_similarity_model():
    if SentenceTransformer is None:
        print("Semantic model unavailable. Using offline overlap scoring.")
        return None

    try:
        return SentenceTransformer(MODEL_NAME, local_files_only=True)
    except Exception as error:
        print(f"Semantic model not loaded: {error}")
        print("Using offline overlap scoring instead.")
        return None

model = load_similarity_model()

# ==========================================================
# LOAD JOB DATASET
# Columns required:
# Role, Skills
# ==========================================================
df = pd.read_csv(DATASET_PATH)

# ==========================================================
# GRAPH
# ==========================================================
G = nx.DiGraph()

# ==========================================================
# HELPERS
# ==========================================================
def clean(x):
    return str(x).strip().lower()

def split_skills(text):

    if pd.isna(text):
        return []

    text = str(text).replace(";", ",")
    arr = text.split(",")

    return [clean(x) for x in arr if clean(x)]

# ==========================================================
# BUILD GRAPH
# ==========================================================
for _, row in df.iterrows():

    role = clean(row["Title"])
    skills = split_skills(row["Skills"])

    G.add_node(role, node_type="role")

    for skill in skills:

        G.add_node(skill, node_type="skill")
        G.add_edge(role, skill, relation="requires")

# ==========================================================
# ONTOLOGY EDGES
# ==========================================================
for child, parent in SKILL_ONTOLOGY.items():

    child = clean(child)
    parent = clean(parent)

    G.add_node(child, node_type="skill")
    G.add_node(parent, node_type="skill")

    G.add_edge(child, parent, relation="related_to")

# ==========================================================
# CO-OCCURRENCE EDGES
# ==========================================================
for _, row in df.iterrows():

    skills = split_skills(row["Skills"])

    for i in range(len(skills)):
        for j in range(i + 1, len(skills)):

            s1 = skills[i]
            s2 = skills[j]

            if s1 != s2:
                G.add_edge(s1, s2, relation="co_occurs")
                G.add_edge(s2, s1, relation="co_occurs")

# ==========================================================
# EXPORT GRAPH
# ==========================================================
nx.write_gexf(G, GRAPH_PATH)

summary = pd.DataFrame([{
    "graph_file": str(GRAPH_PATH),
    "nodes": G.number_of_nodes(),
    "edges": G.number_of_edges(),
    "roles": sum(1 for _, d in G.nodes(data=True) if d.get("node_type") == "role"),
    "skills": sum(1 for _, d in G.nodes(data=True) if d.get("node_type") == "skill"),
}])
summary.to_csv(SUMMARY_PATH, index=False)

# ==========================================================
# ROLE EMBEDDINGS
# ==========================================================
roles = [
    n for n, d in G.nodes(data=True)
    if d.get("node_type") == "role"
]

role_vectors = model.encode(roles) if model is not None else None

# ==========================================================
# GRAPH SCORE
# ==========================================================
def kg_score(user_skills, role):

    required = []

    for nbr in G.successors(role):
        if G[role][nbr]["relation"] == "requires":
            required.append(nbr)

    if not required:
        return 0, [], [], 0, 0

    score = 0
    matched = []
    missing = []
    matched_count = 0

    for req in required:

        # direct match
        if req in user_skills:
            score += 2
            matched.append(req)
            matched_count += 1

        else:
            found = False

            for us in user_skills:

                # ontology indirect
                if G.has_edge(us, req) or G.has_edge(req, us):
                    score += 1
                    matched.append(us + " -> " + req)
                    matched_count += 1
                    found = True
                    break

            if not found:
                missing.append(req)

    coverage = matched_count / len(required)

    final = score / (2 * len(required))

    return round(final,3), matched, missing, matched_count, round(coverage,3)

# ==========================================================
# SEMANTIC SCORE
# ==========================================================
def semantic_score(user_skills, role):

    required = [
        nbr for nbr in G.successors(role)
        if G[role][nbr]["relation"] == "requires"
    ]

    if model is None or cosine_similarity is None or role_vectors is None:
        user_tokens = set(user_skills)
        role_tokens = set(required)
        overlap = len(user_tokens & role_tokens)
        union = len(user_tokens | role_tokens)
        return round(overlap / max(union, 1), 3)

    text = " ".join(user_skills)
    u = model.encode([text])

    idx = roles.index(role)
    r = role_vectors[idx].reshape(1,-1)

    sim = cosine_similarity(u, r)[0][0]

    return round(float(sim),3)

# ==========================================================
# INTEREST SCORE
# ==========================================================
def interest_score(role, interest):

    if not interest:
        return 0

    role = role.lower()
    interest = interest.lower()

    if role == interest:
        return 1.0

    if interest in role or role in interest:
        return 0.7

    return 0

# ==========================================================
# EXPERIENCE SCORE
# ==========================================================
def exp_score(years):

    if years is None:
        return 0.5

    if years == 0:
        return 0.4

    if years <= 2:
        return 0.7

    if years <= 5:
        return 1.0

    return 0.9

# ==========================================================
# FINAL RECOMMENDER
# ==========================================================
def recommend_roles(
        user_skills,
        field_of_interest=None,
        experience_years=None,
        top_n=5
):

    user_skills = [clean(x) for x in user_skills]

    results = []

    for role in roles:

        kg, matched, missing, matched_count, coverage = kg_score(
            user_skills, role
        )

        sem = semantic_score(user_skills, role)
        intr = interest_score(role, field_of_interest)
        exp = exp_score(experience_years)

        # ------------------------------------------
        # THRESHOLD FILTER
        # ------------------------------------------
        exact_interest = (
            field_of_interest and
            clean(field_of_interest) == role
        )

        if (
            matched_count < 2 and
            coverage < 0.30 and
            not exact_interest
        ):
            continue

        # ------------------------------------------
        # FINAL SCORE
        # ------------------------------------------
        final = (
            0.40 * sem +
            0.30 * kg +
            0.20 * intr +
            0.10 * exp
        )

        # match label
        if final >= 0.70:
            strength = "Strong Match"
        elif final >= 0.50:
            strength = "Moderate Match"
        else:
            strength = "Weak Match"

        results.append({
            "role": role.title(),
            "score": round(final,3),
            "strength": strength,
            "matched": matched[:8],
            "missing": missing[:8],
            "coverage": coverage
        })

    # ------------------------------------------
    # SORT
    # ------------------------------------------
    results = sorted(
        results,
        key=lambda x: x["score"],
        reverse=True
    )

    return results[:top_n]

# ==========================================================
# TEST
# ==========================================================
if __name__ == "__main__":

    print("\n===== KNOWLEDGE GRAPH GENERATED =====\n")
    print(f"Dataset : {DATASET_PATH}")
    print(f"Graph   : {GRAPH_PATH}")
    print(f"Summary : {SUMMARY_PATH}")
    print(f"Nodes   : {G.number_of_nodes()}")
    print(f"Edges   : {G.number_of_edges()}")
    print(f"Roles   : {len(roles)}")

    skills = [
        "sql",
        "excel",
        "power bi",
        "tableau",
        "communication",
        "analysis"
    ]

    recs = recommend_roles(
        user_skills=skills,
        field_of_interest="business analyst",
        experience_years=2,
        top_n=5
    )

    print("\n===== TOP RECOMMENDATIONS =====\n")

    for i, r in enumerate(recs,1):

        print(f"#{i} {r['role']} - {r['strength']} ({r['score']})")
        print("Skill Coverage :", r["coverage"])
        print("Matching Skills:", r["matched"])
        print("Missing Skills :", r["missing"])
        print("-"*50)

    print(f"\nGraph Exported: {GRAPH_PATH}")
