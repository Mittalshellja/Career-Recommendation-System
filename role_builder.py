import json
import re
from collections import Counter, defaultdict

from skill_filters import NON_SKILL_TERMS, clean_skills

ROLE_ALIASES = {
    "data science": "data scientist",
    "data scientist": "data scientist",
    "data analyst / data scientist": "data scientist",
    "data science intern": "data scientist",
    "web development": "web developer",
    "web developer": "web developer",
    "web development intern": "web developer",
    "front end development": "front end developer",
    "frontend development": "front end developer",
    "back end development": "back end developer",
    "backend development": "back end developer",
    "machine learning": "machine learning engineer",
    "ml engineer": "machine learning engineer",
    "software development": "software developer",
    "software engineering": "software developer",
    "software engineer": "software developer",
    "frontend developer": "front end developer",
    "backend developer": "back end developer",
}


def canonicalize_role_name(title):
    if not isinstance(title, str):
        return None

    title = title.lower().strip()
    title = title.replace("&", " and ")
    title = re.sub(r"[^\w\s/+-]", " ", title)
    title = re.sub(r"\s+", " ", title).strip()

    remove_terms = [
        "fresher",
        "entry-level",
        "entry level",
        "intern",
        "junior",
        "senior",
        "lead",
        "principal",
        "experienced",
        "associate",
        "trainee",
    ]

    for term in remove_terms:
        title = title.replace(f" - {term}", "")
        title = title.replace(term, "")

    title = title.replace("-", " ").strip(" /")
    title = re.sub(r"\s+", " ", title).strip()
    if not title:
        return None

    return ROLE_ALIASES.get(title, title)


def normalize_role(title):
    canonical = canonicalize_role_name(title)
    if canonical is None:
        return None
    return canonical.title()


def build_role_skill_map(json_path, top_k=15):
    with open(json_path, "r", encoding="utf-8") as file:
        jobs = json.load(file)

    role_skill_counter = defaultdict(Counter)

    for job in jobs:
        title = job.get("Title", "")
        role = normalize_role(title)

        if role is None:
            continue

        skills = [skill.lower().strip() for skill in job.get("Skills", [])]
        keywords = [
            keyword.lower().strip()
            for keyword in job.get("Keywords", [])
            if not any(
                word in keyword.lower()
                for word in [
                    "entry-level",
                    "fresher",
                    "senior",
                    "mid",
                    "junior",
                    "full stack developer",
                    "frontend",
                    "backend",
                    "developer",
                    "engineer",
                ]
            )
        ]

        combined = skills + keywords
        combined = [
            skill
            for skill in combined
            if skill not in NON_SKILL_TERMS and len(skill) > 2
        ]
        combined = clean_skills(
            combined,
            remove_soft=True,
            remove_low_signal=True,
        )

        for skill in combined:
            role_skill_counter[role][skill] += 1

    role_skill_map = {}

    for role, skill_counts in role_skill_counter.items():
        top_skills = [skill for skill, _ in skill_counts.most_common(top_k)]
        role_skill_map[role] = list(set(top_skills))

    print("Total roles loaded:", len(role_skill_map))
    return role_skill_map
