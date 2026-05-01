# user_profile.py
# Handles user interest and experience scoring
# Fully data-driven with offline fallback

import json
import re

import numpy as np

from role_builder import canonicalize_role_name

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:
    SentenceTransformer = None
    cosine_similarity = None

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


def tokenize_role_text(text):
    cleaned = re.sub(r"[^\w\s]", " ", text.lower())
    return {token for token in cleaned.split() if token}


def calculate_interest_score(career, field_of_interest, role_skills):
    """
    Compute how well a career aligns with
    the user's field of interest.
    """

    if not field_of_interest:
        return 0.5

    canonical_interest = canonicalize_role_name(field_of_interest) or field_of_interest.lower()
    canonical_career = canonicalize_role_name(career) or career.lower()

    if canonical_interest == canonical_career:
        return 1.0

    model = get_similarity_model()
    career_context = f"{career} {' '.join(role_skills)}"

    if model is not None and cosine_similarity is not None:
        interest_vec = model.encode([field_of_interest])
        career_vec = model.encode([career_context])
        semantic_score = float(cosine_similarity(interest_vec, career_vec)[0][0])
    else:
        interest_tokens = tokenize_role_text(canonical_interest)
        career_tokens = tokenize_role_text(canonical_career)
        overlap = len(interest_tokens & career_tokens)
        union = len(interest_tokens | career_tokens)
        semantic_score = overlap / max(union, 1)

    interest_tokens = tokenize_role_text(canonical_interest)
    career_tokens = tokenize_role_text(canonical_career)
    token_overlap = len(interest_tokens & career_tokens) / max(len(interest_tokens), 1)

    boosted_score = max(semantic_score, token_overlap)
    if canonical_interest in canonical_career or canonical_career in canonical_interest:
        boosted_score = max(boosted_score, 0.9)

    return float(round(min(boosted_score, 1.0), 3))


def build_experience_profile(json_path):
    """
    Derive experience requirements per role
    from job dataset JSON.
    """

    with open(json_path, "r", encoding="utf-8") as file:
        jobs = json.load(file)

    exp_map = {
        "0-1": 0.5,
        "0-2": 1.0,
        "1-3": 2.0,
        "3-5": 4.0,
        "5+": 6.0,
        "5-8": 6.5,
        "8+": 9.0,
    }

    role_exp = {}

    for job in jobs:
        title = canonicalize_role_name(job.get("Title", ""))
        exp_raw = str(job.get("YearsOfExperience", "")).strip()
        exp_val = exp_map.get(exp_raw, None)

        if not title or exp_val is None:
            continue

        role_exp.setdefault(title, []).append(exp_val)

    result = {}
    for title, values in role_exp.items():
        result[title] = {
            "mean": round(np.mean(values), 1),
            "std": round(np.std(values) if len(values) > 1 else 1.0, 1),
        }

    print(f"Experience profile built for {len(result)} roles")
    return result


def calculate_experience_score(career, user_exp_years, experience_profile):
    """
    Score how well user experience matches
    career expectation derived from dataset.
    """

    if not user_exp_years and user_exp_years != 0:
        return 0.5

    career_norm = canonicalize_role_name(career)
    if career_norm not in experience_profile:
        return 0.5

    stats = experience_profile[career_norm]
    mean = stats["mean"]
    std = max(stats["std"], 0.5)
    distance = abs(user_exp_years - mean)

    if distance <= std * 0.5:
        return 1.0
    if distance <= std:
        return 0.7
    if distance <= std * 2:
        return 0.4
    return 0.1


def get_user_input():
    """
    Collect user interest and experience
    interactively.
    """

    print("\n=== User Interest Profile ===\n")

    field_of_interest = input(
        "Enter your field of interest\n"
        "(e.g. machine learning, web development, finance, data analytics): "
    ).strip()

    experience_input = input(
        "\nEnter your years of experience (enter 0 if fresher): "
    ).strip()

    try:
        experience_years = float(experience_input)
    except ValueError:
        experience_years = 0
        print("Invalid input, setting experience to 0")

    return field_of_interest, experience_years
