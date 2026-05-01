# self_consistency_validator.py

import json
import pandas as pd
from skill_normalization import normalize_skills
from skill_ontology     import expand_skills
from skill_matching     import rank_roles, ROLE_SKILLS
from user_profile       import build_experience_profile
from role_builder       import normalize_role

# ── CONFIG ───────────────────────────────────────────
JOB_JSON        = "archive/job_dataset.json"
RESULTS_CSV     = "self_consistency_results.csv"
TOP_K           = 5
SCORE_THRESHOLD = 0.30
# ─────────────────────────────────────────────────────


def load_jobs(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        jobs = json.load(f)
    print(f"Total jobs loaded : {len(jobs)}")
    return jobs


def run_validation():

    # 1. Load jobs
    jobs = load_jobs(JOB_JSON)

    # 2. Build experience profile
    try:
        experience_profile = build_experience_profile(
                                JOB_JSON)
    except Exception:
        experience_profile = {}

    # Map experience range to midpoint
    exp_map = {
        "0-1" : 0.5,
        "0-2" : 1.0,
        "1-3" : 2.0,
        "3-5" : 4.0,
        "5+"  : 6.0,
        "5-8" : 6.5,
        "8+"  : 9.0
    }

    results      = []
    top1_correct = 0
    topk_correct = 0
    skipped      = 0
    total        = len(jobs)

    print("\nRunning self consistency validation...\n")

    for job in jobs:

        # 3. Get role and skills from job entry
        title      = job.get("Title", "")
        exp_raw    = str(job.get(
                        "YearsOfExperience", ""))
        skills_raw = job.get("Skills", [])

        # Normalize role name same way system does
        expected_role = normalize_role(title)

        if not expected_role or not skills_raw:
            skipped += 1
            continue

        # Check if role exists in system
        if expected_role not in ROLE_SKILLS:
            skipped += 1
            continue

        # 4. Use job skills as user skills
        user_skills = [s.lower().strip() 
                       for s in skills_raw]
        user_skills = normalize_skills(user_skills)
        user_skills = expand_skills(user_skills)

        if not user_skills:
            skipped += 1
            continue

        # 5. Get experience years
        exp_years = exp_map.get(exp_raw, 1.0)

        # 6. Run through your pipeline
        ranked = rank_roles(
            user_skills,
            experience_years   = exp_years,
            experience_profile = experience_profile
        )

        recommendations = [
            role for role, details in ranked
            if details["score"] >= SCORE_THRESHOLD
        ][:TOP_K]

        if not recommendations:
            skipped += 1
            results.append({
                "job_id"       : job.get("JobID", ""),
                "expected_role": expected_role,
                "top1_rec"     : "None",
                "topk_recs"    : [],
                "top1_match"   : False,
                "topk_match"   : False,
                "skills_count" : len(user_skills)
            })
            continue

        # 7. Check if expected role in recommendations
        expected_norm = expected_role.lower().strip()

        top1_match = recommendations[0]\
                        .lower().strip() == expected_norm

        topk_match = any(
            expected_norm in r.lower().strip() or
            r.lower().strip() in expected_norm
            for r in recommendations
        )

        if top1_match: top1_correct += 1
        if topk_match: topk_correct += 1

        status = "✅" if topk_match else "❌"
        print(
            f"{status} {job.get('JobID',''):<12} | "
            f"Expected: {expected_role:<30} | "
            f"Got: {recommendations[0]:<30}"
        )

        results.append({
            "job_id"       : job.get("JobID", ""),
            "expected_role": expected_role,
            "top1_rec"     : recommendations[0],
            "topk_recs"    : str(recommendations),
            "top1_match"   : top1_match,
            "topk_match"   : topk_match,
            "skills_count" : len(user_skills)
        })

    # 8. Calculate accuracy
    evaluated     = len(results)
    top1_accuracy = round(
        top1_correct / max(evaluated, 1), 3)
    topk_accuracy = round(
        topk_correct / max(evaluated, 1), 3)

    # 9. Print report
    print("\n" + "="*60)
    print("    SELF CONSISTENCY VALIDATION REPORT")
    print("="*60)
    print(f"Total Jobs        : {total}")
    print(f"Evaluated         : {evaluated}")
    print(f"Skipped           : {skipped}")
    print(f"Top-1 Accuracy    : "
          f"{top1_accuracy:.1%} "
          f"({top1_correct}/{evaluated})")
    print(f"Top-{TOP_K} Accuracy   : "
          f"{topk_accuracy:.1%} "
          f"({topk_correct}/{evaluated})")

    # 10. Role wise breakdown
    results_df = pd.DataFrame(results)

    print("\n--- Role Wise Accuracy ---")
    print(f"{'Role':<30} {'Correct':<10} "
          f"{'Total':<10} {'Accuracy'}")
    print("-" * 60)

    for role in results_df["expected_role"].unique():
        role_df  = results_df[
                    results_df["expected_role"] == role]
        total_r  = len(role_df)
        correct  = role_df["topk_match"].sum()
        accuracy = correct / total_r \
                   if total_r > 0 else 0
        bar      = "█" * int(accuracy * 10)
        print(f"{role:<30} {correct:<10} "
              f"{total_r:<10} {accuracy:.0%}  {bar}")

    # 11. Save results
    results_df.to_csv(RESULTS_CSV, index=False)
    print(f"\nResults saved to {RESULTS_CSV}")

    return top1_accuracy, topk_accuracy


if __name__ == "__main__":
    run_validation()