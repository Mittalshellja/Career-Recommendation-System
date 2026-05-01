from pathlib import Path

from role_builder import canonicalize_role_name, build_role_skill_map
from skill_ontology import SKILL_ONTOLOGY
from skill_weights import get_role_skill_weight
from user_profile import calculate_experience_score, calculate_interest_score

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:
    SentenceTransformer = None
    cosine_similarity = None

BASE_DIR = Path(__file__).resolve().parent
ROLE_SKILLS = build_role_skill_map(BASE_DIR / "archive" / "job_dataset.json")
MODEL_NAME = "all-MiniLM-L6-v2"
_MODEL = None
_MODEL_LOAD_FAILED = False


def get_similarity_model():
    global _MODEL, _MODEL_LOAD_FAILED

    if _MODEL is not None:
        return _MODEL

    if _MODEL_LOAD_FAILED or SentenceTransformer is None:
        return None

    try:
        _MODEL = SentenceTransformer(MODEL_NAME, local_files_only=True)
    except Exception:
        _MODEL_LOAD_FAILED = True
        _MODEL = None

    return _MODEL


def compute_semantic_score(user_skills, role_skills):
    model = get_similarity_model()
    user_text = " ".join(user_skills)
    role_text = " ".join(role_skills)

    if model is not None and cosine_similarity is not None:
        user_embedding = model.encode([user_text])
        role_embedding = model.encode([role_text])
        return float(cosine_similarity(user_embedding, role_embedding)[0][0])

    user_tokens = set(user_skills)
    role_tokens = set(role_skills)
    overlap = len(user_tokens & role_tokens)
    union = len(user_tokens | role_tokens)
    return overlap / max(union, 1)


print("Total roles loaded:", len(ROLE_SKILLS))


def get_interest_experience_bonus(
    canonical_interest,
    canonical_role,
    experience_years,
    experience_score,
):
    if not canonical_interest or canonical_role != canonical_interest:
        return 0.0

    if experience_years is None:
        return 0.15

    if experience_years >= 8:
        return 0.28 if experience_score >= 0.4 else 0.22
    if experience_years >= 5:
        return 0.24 if experience_score >= 0.4 else 0.19
    if experience_years >= 2:
        return 0.20 if experience_score >= 0.4 else 0.17
    return 0.15


def get_interest_weight_multiplier(experience_years):
    if experience_years is None:
        return 1.0
    if experience_years >= 8:
        return 1.35
    if experience_years >= 5:
        return 1.25
    if experience_years >= 2:
        return 1.1
    return 1.0


def rank_roles(
    user_skills,
    field_of_interest=None,
    experience_years=None,
    experience_profile=None,
):
    has_interest = bool(field_of_interest)
    has_experience = experience_years is not None

    if has_interest and has_experience:
        alpha = 0.30
        beta = 0.20
        gamma = 0.40
        delta = 0.10
    elif has_interest and not has_experience:
        alpha = 0.35
        beta = 0.15
        gamma = 0.50
        delta = 0.00
    elif has_experience and not has_interest:
        alpha = 0.55
        beta = 0.25
        gamma = 0.00
        delta = 0.20
    else:
        alpha = 0.70
        beta = 0.30
        gamma = 0.00
        delta = 0.00

    role_scores = {}

    for role, skills in ROLE_SKILLS.items():
        semantic_score = compute_semantic_score(user_skills, skills)
        matching_skills = []
        missing_skills = []

        for skill in skills:
            if is_skill_satisfied(skill, user_skills):
                matching_skills.append(skill)
            else:
                missing_skills.append(skill)

        matching_skills = list(set(matching_skills))
        missing_skills = list(set(missing_skills) - set(matching_skills))
        weight_score = calculate_weight_score(role, matching_skills) / max(len(skills), 1)

        interest_score = (
            calculate_interest_score(role, field_of_interest, skills)
            if has_interest
            else 0.0
        )
        canonical_interest = (
            canonicalize_role_name(field_of_interest) if has_interest else None
        )
        canonical_role = canonicalize_role_name(role)
        if canonical_interest and canonical_role == canonical_interest:
            interest_score = 1.0

        experience_score = (
            calculate_experience_score(role, experience_years, experience_profile or {})
            if has_experience
            else 0.0
        )

        total_skills = len(matching_skills) + len(missing_skills)
        fit_percentage = round(len(matching_skills) / max(total_skills, 1) * 100, 1)

        effective_interest_score = interest_score
        if canonical_interest and canonical_role == canonical_interest:
            effective_interest_score = min(
                1.0,
                interest_score * get_interest_weight_multiplier(experience_years),
            )

        final_score = (
            alpha * float(semantic_score)
            + beta * weight_score
            + gamma * effective_interest_score
            + delta * experience_score
        )

        if canonical_interest and canonical_role == canonical_interest:
            final_score += get_interest_experience_bonus(
                canonical_interest,
                canonical_role,
                experience_years,
                experience_score,
            )
        elif canonical_interest and (
            canonical_interest in canonical_role or canonical_role in canonical_interest
        ):
            final_score += 0.08

        role_scores[role] = {
            "score": round(min(final_score, 1.0), 3),
            "semantic_score": round(float(semantic_score), 3),
            "weight_score": round(weight_score, 3),
            "interest_score": round(interest_score, 3),
            "effective_interest_score": round(effective_interest_score, 3),
            "experience_score": round(experience_score, 3),
            "missing_skills": missing_skills,
            "matching_skills": matching_skills,
            "role_skills": skills,
            "fit_percentage": fit_percentage,
            "interest_aligned": bool(
                canonical_interest and canonical_role == canonical_interest
            ),
            "senior_interest_match": bool(
                canonical_interest
                and canonical_role == canonical_interest
                and experience_years is not None
                and experience_years >= 5
                and experience_score >= 0.4
            ),
        }

    ranked_roles = sorted(
        role_scores.items(),
        key=lambda item: item[1]["score"],
        reverse=True,
    )
    return ranked_roles


def is_skill_satisfied(skill, user_skills):
    if skill in user_skills:
        return True

    parent = SKILL_ONTOLOGY.get(skill)
    if parent and parent in user_skills:
        return True

    return False


def calculate_weight_score(role, matching_skills):
    score = 0

    for skill in matching_skills:
        weight = get_role_skill_weight(role, skill)
        score += weight

    return score


if __name__ == "__main__":
    user_skills = ["python", "sql", "statistics"]
    ranked = rank_roles(user_skills)

    print("Recommended Roles:")
    for role, details in ranked:
        print(f"\nRole: {role}")
        print(f"Score: {round(details['score'], 3)}")
        print(f"Missing Skills: {details['missing_skills']}")
