SKILL_ONTOLOGY = {

    # Python ecosystem
    "pandas": "python",
    "numpy": "python",
    "matplotlib": "python",
    "seaborn": "python",
    "scikit-learn": "machine learning",

    # ML frameworks
    "tensorflow": "deep learning",
    "keras": "deep learning",
    "pytorch": "deep learning",

    # data tools
    "tableau": "data visualization",
    "power bi": "data visualization",

    # databases
    "postgresql": "sql",
    "mysql": "sql",

    # big data
    "spark": "big data",
    "hadoop": "big data"
}
def expand_skills(user_skills):

    expanded = set(user_skills)

    for children,parent in SKILL_ONTOLOGY.items():

       # for skill in user_skills:

        if children in user_skills:
            expanded.add(parent)

    return list(expanded)