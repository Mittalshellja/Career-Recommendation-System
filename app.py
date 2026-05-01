from main import generate_recommendations, load_experience_profile
from resume_parser import extract_text_from_file
from pathlib import Path
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
LABELED_METRICS_PATH = BASE_DIR / "labeled_metrics.csv"

st.set_page_config(
    page_title="Career Recommender System",
    page_icon="Career",
    layout="wide",
)


@st.cache_resource
def get_cached_experience_profile():
    try:
        return load_experience_profile()
    except Exception:
        return {}


experience_profile = get_cached_experience_profile()


def show_validation_metrics():
    if not LABELED_METRICS_PATH.exists():
        st.info("Validation metrics are not available yet. Run validation.py to create them.")
        return

    metrics = pd.read_csv(LABELED_METRICS_PATH)
    if metrics.empty:
        st.info("Validation metrics file is empty. Run validation.py again.")
        return

    row = metrics.iloc[0]
    st.subheader("Validation Metrics")
    col_1, col_2, col_3, col_4, col_5 = st.columns(5)
    col_1.metric("Accuracy", f"{row.get('Accuracy', 0):.3f}")
    col_2.metric("Precision", f"{row.get('Precision', 0):.3f}")
    col_3.metric("Recall", f"{row.get('Recall', 0):.3f}")
    col_4.metric("F1 Score", f"{row.get('F1 Score', 0):.3f}")
    col_5.metric("Top-3 Accuracy", f"{row.get('Top3 Accuracy', 0):.3f}")

st.title("Personalized Career Recommender System")
st.markdown(
    "Upload your resume in PDF, DOCX, or TXT format and get career recommendations based on extracted skills."
)
st.divider()

field_of_interest = ""
experience_years = 0
top_k = 5

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Resume Input")
    uploaded_file = st.file_uploader(
        "Upload your resume",
        type=["pdf", "docx", "txt"],
        help="Supported formats: PDF, DOCX, TXT",
    )

    if uploaded_file:
        with st.spinner("Reading resume..."):
            resume_input, error = extract_text_from_file(uploaded_file)

        if error:
            st.error(f"Error reading file: {error}")
            resume_input = ""
        else:
            st.success(
                f"Resume loaded successfully. {len(resume_input.split())} words found."
            )
            with st.expander("Preview extracted text"):
                st.text(resume_input[:1000] if resume_input else "")
    else:
        resume_input = st.text_area(
            "Or paste your resume text here",
            height=300,
            placeholder=(
                "Paste your resume content here...\n"
                "Example:\n"
                "Experienced Data Scientist with 3 years of experience in Python, SQL, and Machine Learning."
            ),
        )

with col2:
    st.subheader("Your Preferences")
    field_of_interest = st.text_input(
        "Field of Interest",
        placeholder="e.g. Data Science, Web Development, Finance",
    )
    experience_years = st.slider(
        "Years of Experience",
        min_value=0,
        max_value=20,
        value=0,
        step=1,
    )
    st.markdown(
        f"**Experience Level:** {'Fresher' if experience_years <= 1 else 'Junior' if experience_years <= 3 else 'Mid Level' if experience_years <= 6 else 'Senior'}"
    )
    top_k = st.selectbox(
        "Number of Recommendations",
        options=[3, 5, 10],
        index=1,
    )

generate = st.button(
    "Generate Recommendations",
    type="primary",
    use_container_width=True,
)

if generate:
    if not resume_input:
        st.error("Please upload a resume file or paste resume text first.")
    else:
        with st.spinner("Analyzing resume and generating recommendations..."):
            result = generate_recommendations(
                resume_input,
                field_of_interest=field_of_interest,
                experience_years=float(experience_years),
                experience_profile=experience_profile,
                min_score=0.0,
                top_k=top_k,
            )

        if result["error"]:
            st.error(result["error"])
        else:
            st.subheader("Extracted Skills")
            st.write(", ".join(result["user_skills"]))
            st.divider()

            recommendations = result["recommendations"]
            if not recommendations:
                st.warning(
                    "No strong matches found. Try adding more relevant technical skills to your resume."
                )
            else:
                st.subheader(f"Top {len(recommendations)} Career Recommendations")

                for index, recommendation in enumerate(recommendations, start=1):
                    role = recommendation["role"]
                    details = recommendation["details"]
                    label = recommendation["match_level"]

                    with st.expander(
                        f"#{index} {role} - {label} ({details['score']:.3f})",
                        expanded=index == 1,
                    ):
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.metric("Match Score", f"{details['score']:.3f}")
                        with col_b:
                            st.metric("Skill Fit", f"{details['fit_percentage']}%")
                        with col_c:
                            st.metric(
                                "Interest Score",
                                f"{details['interest_score']:.3f}",
                            )

                        if details["matching_skills"]:
                            st.markdown("**Matching Skills**")
                            st.success(", ".join(details["matching_skills"]))

                        if details["missing_skills"]:
                            st.markdown("**Skills to Develop**")
                            st.error(", ".join(details["missing_skills"][:6]))

                        st.info(recommendation["explanation"])
                        if "llm_verdict" in recommendation:
                            if recommendation["llm_verdict"] == 1:
                                st.success("LLM Verdict: Suitable")
                            elif recommendation["llm_verdict"] == 0:
                                st.error("LLM Verdict: Not Suitable")
                            else:
                                st.warning("LLM Verdict: Uncertain")

                            st.caption(recommendation["llm_explanation"])
                        if recommendation["next_roles"]:
                            st.markdown("**Career Path**")
                            st.write(" -> ".join(recommendation["next_roles"][:3]))

                st.divider()
                show_validation_metrics()

st.divider()
st.markdown(
    "<center>Personalized Career Recommender System </center>",
    unsafe_allow_html=True,
)
