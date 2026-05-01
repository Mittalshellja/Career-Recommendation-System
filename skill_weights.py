import json
from collections import Counter, defaultdict
from pathlib import Path

from role_builder import normalize_role
from skill_filters import NON_SKILL_TERMS, clean_skills

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "archive" / "job_dataset.json"
DEFAULT_SKILL_WEIGHT = 0.1


def build_role_skill_weights(json_path):
    with open(json_path, "r", encoding="utf-8") as file:
        jobs = json.load(file)

    role_skill_counter = defaultdict(Counter)

    for job in jobs:
        role = normalize_role(job.get("Title", ""))
        if role is None:
            continue

        skills = [skill.lower().strip() for skill in job.get("Skills", [])]
        keywords = [keyword.lower().strip() for keyword in job.get("Keywords", [])]
        combined = [
            skill
            for skill in skills + keywords
            if skill not in NON_SKILL_TERMS and len(skill) > 2
        ]
        combined = clean_skills(
            combined,
            remove_soft=True,
            remove_low_signal=True,
        )

        for skill in combined:
            role_skill_counter[role][skill] += 1

    role_skill_weights = {}
    for role, skill_counts in role_skill_counter.items():
        max_count = max(skill_counts.values(), default=1)
        role_skill_weights[role] = {
            skill: round(count / max_count, 3)
            for skill, count in skill_counts.items()
        }

    return role_skill_weights


ROLE_SKILL_WEIGHTS = build_role_skill_weights(DATASET_PATH)


def get_role_skill_weight(role, skill):
    role_weights = ROLE_SKILL_WEIGHTS.get(role, {})
    return role_weights.get(skill, DEFAULT_SKILL_WEIGHT)
