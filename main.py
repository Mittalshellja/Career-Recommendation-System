from pathlib import Path

from LLM_Validation import llm_judge
from career_graph import get_next_roles
from explainability import generate_explanation
from role_builder import canonicalize_role_name
from skill_extraction import extract_skills_from_text
from skill_matching import rank_roles
from skill_normalization import normalize_skills
from skill_ontology import expand_skills
from user_profile import build_experience_profile, get_user_input

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "archive" / "job_dataset.json"
DEFAULT_RESUME_PATH = BASE_DIR / "resumes" / "resume1.txt"


def load_resume_text(resume_path=DEFAULT_RESUME_PATH):
    with open(resume_path, "r", encoding="utf-8") as file:
        return file.read()


def extract_user_skills(resume_text):
    user_skills = extract_skills_from_text(resume_text)
    user_skills = normalize_skills(user_skills)
    user_skills = expand_skills(user_skills)
    return sorted(set(user_skills))


def load_experience_profile():
    return build_experience_profile(DATASET_PATH)


def build_recommendation(role, details):
    score = details["score"]
    if details.get("senior_interest_match") and score >= 0.45:
        match_level = "Experienced Interest Match"
    elif score > 0.75:
        match_level = "Strong Match"
    elif score > 0.55:
        match_level = "Moderate Match"
    else:
        match_level = "Weak Match"

    return {
        "role": role,
        "details": details,
        "match_level": match_level,
        "explanation": generate_explanation(
            role,
            details["matching_skills"],
            details["missing_skills"],
        ),
        "next_roles": get_next_roles(role),
    }


def is_displayable_recommendation(role, details, field_of_interest=""):
    canonical_interest = canonicalize_role_name(field_of_interest) if field_of_interest else None
    canonical_role = canonicalize_role_name(role)
    matching_skill_count = len(details["matching_skills"])

    if canonical_interest and canonical_role == canonical_interest:
        return True

    return matching_skill_count >= 2


def select_recommendations(ranked_roles, field_of_interest="", min_score=0.0, top_k=5):
    eligible_roles = [
        (role, details)
        for role, details in ranked_roles
        if details["score"] >= min_score
        and is_displayable_recommendation(role, details, field_of_interest)
    ]

    selected = eligible_roles[:top_k]
    canonical_interest = canonicalize_role_name(field_of_interest) if field_of_interest else None

    if canonical_interest:
        interest_match = next(
            (
                (role, details)
                for role, details in eligible_roles
                if canonicalize_role_name(role) == canonical_interest
            ),
            None,
        )
        if interest_match and all(role != interest_match[0] for role, _ in selected):
            if len(selected) >= top_k:
                selected = selected[:-1]
            selected.append(interest_match)

    return [build_recommendation(role, details) for role, details in selected]


def generate_recommendations(
    resume_text,
    field_of_interest="",
    experience_years=0,
    experience_profile=None,
    min_score=0.25,
    top_k=5,
):
    user_skills = extract_user_skills(resume_text)

    if not user_skills:
        return {
            "user_skills": [],
            "recommendations": [],
            "ranked_roles": [],
            "error": "No relevant skills found in the resume.",
        }

    if experience_profile is None:
        experience_profile = load_experience_profile()

    ranked_roles = rank_roles(
        user_skills,
        field_of_interest=field_of_interest,
        experience_years=experience_years,
        experience_profile=experience_profile,
    )

    recommendations = select_recommendations(
        ranked_roles,
        field_of_interest=field_of_interest,
        min_score=min_score,
        top_k=top_k,
    )
    for i, rec in enumerate(recommendations):

        if i >= 3:
            rec["llm_verdict"] = None
            rec["llm_explanation"] = "Skipped for efficiency"
            continue

        try:
            verdict, explanation = llm_judge(
                resume_text,
                rec["role"]
            )
        except Exception as e:
            verdict = -1
            explanation = str(e)

        rec["llm_verdict"] = verdict
        rec["llm_explanation"] = explanation

    return {
        "user_skills": user_skills,
        "recommendations": recommendations,
        "ranked_roles": ranked_roles,
        "error": None,
    }


def main():
    field_of_interest, experience_years = get_user_input()

    try:
        experience_profile = load_experience_profile()
    except Exception as error:
        experience_profile = {}
        print(f"Experience profile error: {error}")

    try:
        resume_text = load_resume_text()
        print("\nResume loaded successfully")
    except FileNotFoundError:
        print(f"Resume file not found at {DEFAULT_RESUME_PATH}")
        print("Please place your resume text in resumes/resume1.txt")
        return

    result = generate_recommendations(
        resume_text,
        field_of_interest=field_of_interest,
        experience_years=experience_years,
        experience_profile=experience_profile,
    )

    if result["error"]:
        print(f"\n{result['error']}")
        print("Please make sure your resume contains relevant technical skills.")
        return

    print(f"Field of Interest : {field_of_interest}")
    print(f"Experience Years  : {experience_years}")
    print("\n=== Career Recommendations ===\n")

    if not result["recommendations"]:
        print("No strong matches found.")
        print("Try updating your resume with more relevant technical skills.")
        return

    print("\n=== DEBUG: Top Role Scores ===\n")
    for role, details in result["ranked_roles"][:10]:
        print(role, "->", details["score"])

    for recommendation in result["recommendations"]:
        role = recommendation["role"]
        details = recommendation["details"]

        print(f"Role             : {role}")
        print(
            f"Match Strength   : {recommendation['match_level']} ({details['score']})"
        )
        print(f"Skill Fit        : {details['fit_percentage']}%")
        print(f"Interest Score   : {details['interest_score']}")
        print(f"Experience Score : {details['experience_score']}")
        print(f"Matching Skills  : {details['matching_skills']}")
        print(f"Missing Skills   : {details['missing_skills']}")
        print(f"Explanation      : {recommendation['explanation']}")

        if recommendation["next_roles"]:
            print(f"Career Path      : {recommendation['next_roles']}")

        print("-" * 55)


if __name__ == "__main__":
    main()
