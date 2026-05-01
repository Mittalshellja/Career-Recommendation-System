import re

NON_SKILL_TERMS = {
    "fresher",
    "entry-level",
    "entry level",
    "senior",
    "senior-level",
    "junior",
    "associate",
    "experienced",
    "trainee",
    "intern",
    "mentor",
    "mentoring",
    "lead",
    "team",
    "developers",
    "project",
    "experience",
    "years",
    "role",
    "responsibilities",
}

STOP_SKILLS = {
    "fresher",
    "entry-level",
    "senior-level",
    "experienced",
    "internship",
    "intern",
    "data scientist",
    "data analyst",
    "machine learning engineer",
    "team",
    "developer",
    "engineer",
    "lead",
    "junior",
    "senior",
    "data science intern",
    "software",
    "backend",
    "frontend",
    "apis",
    "api",
    "rest api",
    "rest apis",
    "database",
    "analytics",
    "data analytics",
    "testing",
    "manual testing",
    "testing and debugging",
    "debugging",
    "classification",
    "bi",
    "go",
    "ros",
    "ui",
    "spa",
    "sem",
    "ai",
    "gpt",
    "js",
    "functions",
    "city",
    "state",
    "country",
}

SOFT_SKILLS = {
    "communication",
    "team communication",
    "stakeholder communication",
    "stakeholder management",
    "collaboration",
    "collaboration and communication",
    "problem solving",
    "creative problem solving",
    "critical thinking",
    "presentation",
    "presenting",
    "mentoring",
    "leadership",
    "training",
    "teamwork",
}

LOW_SIGNAL_SKILLS = {
    "documentation",
    "monitoring",
    "tracking",
    "performance tracking",
}

LOCATION_TERMS = {
    "chennai",
    "bangalore",
    "bengaluru",
    "hyderabad",
    "pune",
    "mumbai",
    "delhi",
    "noida",
    "gurgaon",
    "india",
}

CANONICAL_SKILL_MAP = {
    "agile methodology": "agile project management",
    "agile methodologies": "agile project management",
    "agile leadership": "agile project management",
    "ai-powered analytics": "ai analytics",
    "ai powered analytics": "ai analytics",
    "ai enhanced analytics": "ai analytics",
    "ai-enhanced analytics": "ai analytics",
    "ai driven analytics": "ai analytics",
    "ai-driven analytics": "ai analytics",
    "visualization": "data visualization",
    "stakeholder engagement": "stakeholder management",
    "stakeholder engagement and communication": "stakeholder management",
    "stakeholder communication and engagement": "stakeholder management",
    "requirement management": "requirement gathering",
}

PROTECTED_SHORT_SKILLS = {
    "sql",
    "nlp",
    "crm",
    "erp",
    "jira",
    "html",
    "css",
    "r",
}


def normalize_skill_text(skill):
    skill = skill.lower().strip()
    skill = skill.replace("&", " and ")
    skill = skill.replace("-", " ")
    skill = re.sub(r"[^\w\s.+#/-]", " ", skill)
    skill = re.sub(r"\s+", " ", skill).strip()
    skill = re.sub(r"^(advanced|basic)\s+", "", skill)
    skill = re.sub(r"\s+basics?$", "", skill)
    return CANONICAL_SKILL_MAP.get(skill, skill)


def is_soft_skill(skill):
    return normalize_skill_text(skill) in SOFT_SKILLS


def is_noise_skill(skill):
    normalized = normalize_skill_text(skill)
    if not normalized or len(normalized) <= 2:
        return True
    if normalized in STOP_SKILLS or normalized in LOCATION_TERMS:
        return True
    if normalized in LOW_SIGNAL_SKILLS:
        return True
    if normalized.isdigit():
        return True
    return False


def remove_redundant_skills(skill_list):
    unique_skills = list(dict.fromkeys(skill_list))
    filtered = []

    for skill in unique_skills:
        skill_tokens = set(skill.split())
        if skill in PROTECTED_SHORT_SKILLS:
            filtered.append(skill)
            continue

        is_redundant = False
        for other in unique_skills:
            if other == skill:
                continue
            other_tokens = set(other.split())
            if len(skill_tokens) < len(other_tokens) and skill_tokens.issubset(other_tokens):
                is_redundant = True
                break

        if not is_redundant:
            filtered.append(skill)

    return filtered


def clean_skills(skill_list, remove_soft=False, remove_low_signal=False):
    cleaned = []

    for skill in skill_list:
        normalized = normalize_skill_text(skill)

        if is_noise_skill(normalized):
            continue
        if remove_soft and is_soft_skill(normalized):
            continue
        if remove_low_signal and normalized in LOW_SIGNAL_SKILLS:
            continue

        cleaned.append(normalized)

    return remove_redundant_skills(cleaned)
