def generate_explanation(role, matching_skills, missing_skills):

    explanation = ""

    if matching_skills:
        explanation += (
            f"Your skills {', '.join(matching_skills[:4])} "
            f"strongly match the requirements of the {role} role. "
        )

    if missing_skills:
        explanation += (
            "To improve your chances you should develop skills in "
            f"{', '.join(missing_skills[:4])}."
        )

    return explanation