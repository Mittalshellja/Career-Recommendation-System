SKILL_SYNONYMS = {
    "ml": "machine learning",
    "dl": "deep learning",
    "pandas": "data analysis",
    "numpy": "data analysis",
    "nat lang" : "nlp",
    "stats" : "statistics",
}

def normalize_skills(skill_list):
    if skill_list is None:
        return []
    
    normalized = []
    for skill in skill_list:
        skill = skill.lower()
        if skill in SKILL_SYNONYMS:
            normalized.append(SKILL_SYNONYMS[skill])
        else:
            normalized.append(skill)
    return list(set(normalized))


