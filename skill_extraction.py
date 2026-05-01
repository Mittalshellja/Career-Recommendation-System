# skill_extraction.py
import json
import re
from pathlib import Path

import spacy
from skill_filters import clean_skills

nlp = spacy.load("en_core_web_sm")
BASE_DIR = Path(__file__).resolve().parent

def build_skill_vocabulary(json_path):
    """
    Build master skill list from job dataset JSON.
    Skills and Keywords are already lists — 
    no splitting needed.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        jobs = json.load(f)

    all_skills = set()

    for job in jobs:
        # Direct list — just iterate
        for skill in job.get("Skills", []):
            all_skills.add(skill.lower().strip())

        for keyword in job.get("Keywords", []):
            all_skills.add(keyword.lower().strip())

    print(f"Skill vocabulary size: {len(all_skills)}")
    return all_skills


# Build vocabulary at import time
SKILL_VOCAB = build_skill_vocabulary(
    BASE_DIR / "archive" / "job_dataset.json"
)


def extract_skills_from_text(text):
    """
    Extract skills from resume text by matching
    against job dataset vocabulary.
    Handles both single word and multi word skills.
    """
    if not isinstance(text, str):
        return []

    text_lower = text.lower()
    normalized_text = re.sub(r"[^\w\s.+#/-]", " ", text_lower)
    normalized_text = re.sub(r"\s+", " ", normalized_text).strip()
    extracted = set()

    # Sort by length — match longer skills first
    # e.g. "machine learning" before "learning"
    sorted_vocab = sorted(
        SKILL_VOCAB,
        key=len,
        reverse=True
    )

    '''for skill in sorted_vocab:
    # exact match
        if skill in text_lower:
            extracted.add(skill)
    # handle "javascript basics" → match "javascript"
        for word in skill.split():
            if len(word) > 3 and word in text_lower:
                extracted.add(skill)

    return list(extracted)
    '''
    for skill in sorted_vocab:
    # Skip anything longer than 4 words
        if len(skill.split()) > 4:
            continue
    # Skip job titles
        if any(word in skill for word in 
             ["engineer", "developer", "analyst",
                "manager", "intern", "senior", 
                "junior", "lead", "scientist"]):
            continue
        escaped_skill = re.escape(skill)
        pattern = rf"(?<!\w){escaped_skill}(?!\w)"
        if re.search(pattern, normalized_text):
            extracted.add(skill)

    return sorted(clean_skills(extracted, remove_soft=True, remove_low_signal=True))


if __name__ == "__main__":
    sample = """
    Experienced HR professional with skills in 
    customer service, team management, budgeting,
    conflict resolution and Microsoft Office.
    """
    skills = extract_skills_from_text(sample)
    #print("Extracted Skills:", skills)
