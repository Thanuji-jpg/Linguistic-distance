"""
Full WHO AUDIT (10-item) LLM pilot with HAC disease profiles.

Based on auditcode_original.py structure. Every HAC comorbidity
profile is tested against one fixed AUDIT scenario (WHO Zone IV,
Referral to Specialist), so disease effects are compared with
drinking severity held constant.

Set OPENAI_API_KEY in the environment before running.
"""

#AUDIT C Testing

import hashlib
import json
import os
import platform
import re
import time
import unicodedata
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import openai
import pandas as pd
import rapidfuzz
import sentence_transformers
from openai import OpenAI
from rapidfuzz.distance import Levenshtein
from sentence_transformers import SentenceTransformer, util


# PASTE YOUR OPENAI API KEY BETWEEN THE QUOTES BELOW:
# Example: OPENAI_API_KEY_PASTE_HERE = "sk-proj-...."
OPENAI_API_KEY_PASTE_HERE = ""

_openai_api_key = (
    OPENAI_API_KEY_PASTE_HERE.strip()
    or os.environ.get("OPENAI_API_KEY")
)

if not _openai_api_key:
    raise RuntimeError(
        "No OpenAI API key found. Paste it into "
        "OPENAI_API_KEY_PASTE_HERE near the top of this file, "
        "or export OPENAI_API_KEY in your terminal."
    )

client = OpenAI(api_key=_openai_api_key)



# =============================================================================
# 2. EDITABLE RUN SETTINGS
# =============================================================================

model_name = "gpt-5.1"

# Number of independent responses requested for EACH
# (disease profile × AUDIT scenario) combination.
num_iterations = 2

# Only HAC disease profiles are supported in this adapted original script.
condition_mode = "hac_dataset"

# Change this label whenever the prompt wording changes.
prompt_version = "auditc_prompt_v5_full_audit_who_zones"

output_directory = (
    Path(__file__).resolve().parent / "output"
)


# =============================================================================
# 3. HAC DISEASE CONDITION SETTINGS
# =============================================================================

hac_dataset_path = (
    Path(__file__).resolve().parent
    / "Dataset for HAC 3 dice.xlsx"
)

# Excel row that contains DUPERSID, SEX, HYPERTENSION, etc.
hac_excel_header_row = 1

# How to choose patient profiles from the HAC file:
# "unique_profiles" - one condition per unique 0/1 pattern
# "sample"          - random subset of unique profiles
# "all_patients"    - every valid patient row, deduplicated
hac_profile_mode = "sample"

# Used only when hac_profile_mode == "sample"
hac_sample_size = 50

hac_random_seed = 42

COMORBIDITY_ABSENT_LABEL = "Absent"

COMORBIDITY_PRESENT_LABEL = "Present"

HAC_NON_COMORBIDITY_COLUMNS = {
    "DUPERSID",
    "SEX",
    "RACE",
    "FAMINC10",
    "CONDITIONCOUNT",
    "AGE",
    "HEALTHCOSTS",
    "WEIGHT",
}

HAC_COLUMN_LABELS = {
    "spondylosisbackproblems": (
        "Spondylosis / Back Problems"
    ),
    "jointdisorders": "Joint Disorders",
    "depressionbipolar": "Depression / Bipolar",
    "hypertension": "Hypertension",
    "anxietydisorder": "Anxiety Disorder",
    "diabetes": "Diabetes",
    "connectivetissuedisease": (
        "Connective Tissue Disease"
    ),
    "highcholesterol": "High Cholesterol",
    "copd": "COPD",
    "rheumatoidarthritis": "Rheumatoid Arthritis",
    "upperrespinfections": (
        "Upper Respiratory Infections"
    ),
    "adhd": "ADHD",
    "upperrespdisease": (
        "Upper Respiratory Disease"
    ),
    "headachesmigraine": "Headaches / Migraine",
    "nervoussystemdisorders": (
        "Nervous System Disorders"
    ),
    "asthma": "Asthma",
    "kidneydisease": "Kidney Disease",
    "osteoarthritis": "Osteoarthritis",
    "kidneyfailure": "Kidney Failure",
    "heartdisease": "Heart Disease",
}


comorbidity_dimensions = {}

COMORBIDITY_FIELD_NAMES = ()

COMORBIDITY_FIELD_LABELS = {}

hac_loaded_conditions = []


# =============================================================================
# 4. EDITABLE AUDIT RESPONSES (DEFAULTS; OVERRIDDEN BY SCENARIO PRESETS)
# =============================================================================

# Each value must exactly match one option in its scoring dictionary below.
# These defaults are replaced per run by AUDIT_SCENARIOS below.

auditc_item1_response = "2–3 times per week"

auditc_item2_response = "7–9 standard drinks"

auditc_item3_response = "Weekly"

auditc_item4_response = "Never"

auditc_item5_response = "Never"

auditc_item6_response = "Never"

auditc_item7_response = "Never"

auditc_item8_response = "Never"

auditc_item9_response = "No"

auditc_item10_response = "No"


# =============================================================================
# 5. OPTIONAL API SETTINGS
# =============================================================================

# None means that the parameter will not be sent to the API.

temperature = None

top_p = None

reasoning_effort = None

max_output_tokens = None

# Optional pause between requests.
seconds_between_requests = 0.0


# =============================================================================
# 6. LINGUISTIC-ANALYSIS SETTINGS
# =============================================================================

semantic_model_name = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

semantic_batch_size = 32

word_normalization_version = (
    "NFKC_lowercase_normalized_apostrophes_regex_tokens_v1"
)


# =============================================================================
# 7. FULL WHO AUDIT SCORING DEFINITIONS (ITEMS 1–10)
# =============================================================================

auditc_item1_scoring = {
    "Never": 0,
    "Monthly or less": 1,
    "2–4 times per month": 2,
    "2–3 times per week": 3,
    "4 or more times per week": 4,
}

auditc_item2_scoring = {
    "1–2 standard drinks": 0,
    "3–4 standard drinks": 1,
    "5–6 standard drinks": 2,
    "7–9 standard drinks": 3,
    "10 or more standard drinks": 4,
}

auditc_item3_scoring = {
    "Never": 0,
    "Less than monthly": 1,
    "Monthly": 2,
    "Weekly": 3,
    "Daily or almost daily": 4,
}

# Item 4: unable to stop drinking once started
auditc_item4_scoring = {
    "Never": 0,
    "Less than monthly": 1,
    "Monthly": 2,
    "Weekly": 3,
    "Daily or almost daily": 4,
}

# Item 5: failed to do what was normally expected due to drinking
auditc_item5_scoring = {
    "Never": 0,
    "Less than monthly": 1,
    "Monthly": 2,
    "Weekly": 3,
    "Daily or almost daily": 4,
}

# Item 6: needed a morning drink to get going
auditc_item6_scoring = {
    "Never": 0,
    "Less than monthly": 1,
    "Monthly": 2,
    "Weekly": 3,
    "Daily or almost daily": 4,
}

# Item 7: guilt or remorse after drinking
auditc_item7_scoring = {
    "Never": 0,
    "Less than monthly": 1,
    "Monthly": 2,
    "Weekly": 3,
    "Daily or almost daily": 4,
}

# Item 8: unable to remember the night before due to drinking
auditc_item8_scoring = {
    "Never": 0,
    "Less than monthly": 1,
    "Monthly": 2,
    "Weekly": 3,
    "Daily or almost daily": 4,
}

# Item 9: injured self/others due to drinking (0/2/4 scale)
auditc_item9_scoring = {
    "No": 0,
    "Yes, but not in the last year": 2,
    "Yes, during the last year": 4,
}

# Item 10: relative, friend, or doctor concerned (0/2/4 scale)
auditc_item10_scoring = {
    "No": 0,
    "Yes, but not in the last year": 2,
    "Yes, during the last year": 4,
}

AUDIT_ITEM_SCORING = {
    1: auditc_item1_scoring,
    2: auditc_item2_scoring,
    3: auditc_item3_scoring,
    4: auditc_item4_scoring,
    5: auditc_item5_scoring,
    6: auditc_item6_scoring,
    7: auditc_item7_scoring,
    8: auditc_item8_scoring,
    9: auditc_item9_scoring,
    10: auditc_item10_scoring,
}


# =============================================================================
# 8. WHO AUDIT ZONE SCENARIOS AND SCORE HELPERS
# =============================================================================

if not isinstance(num_iterations, int) or num_iterations < 1:
    raise ValueError(
        "num_iterations must be an integer of at least 1."
    )


def score_audit_responses(responses):
    """
    Validate response keys and return item scores plus total (0–40).
    """
    item_scores = {}

    for item_number in range(1, 11):
        response_key = f"item{item_number}_response"
        scoring = AUDIT_ITEM_SCORING[item_number]
        response_value = responses[response_key]

        if response_value not in scoring:
            raise ValueError(
                f"Invalid AUDIT Item {item_number} response "
                f"{response_value!r}. Select one of: "
                f"{list(scoring.keys())}"
            )

        item_scores[item_number] = scoring[response_value]

    total_score = sum(item_scores.values())

    return item_scores, total_score


# Single fixed scenario applied to every patient: WHO Zone IV
# (Referral to Specialist), so comorbidity effects are compared
# with drinking severity held constant across all patients.
AUDIT_SCENARIOS = {
    "zone_iv_specialist_referral": {
        "scenario_id": "zone_iv_specialist_referral",
        "who_zone": "IV",
        "who_zone_label": (
            "Referral to Specialist for Diagnostic "
            "Evaluation and Treatment"
        ),
        "item1_response": "2–3 times per week",
        "item2_response": "5–6 standard drinks",
        "item3_response": "Weekly",
        "item4_response": "Monthly",
        "item5_response": "Monthly",
        "item6_response": "Monthly",
        "item7_response": "Weekly",
        "item8_response": "Monthly",
        "item9_response": "Yes, but not in the last year",
        "item10_response": "Yes, during the last year",
    },
}


def build_audit_scenario(scenario_template):
    """
    Attach item scores, total score, and zone fields to a scenario.
    """
    responses = {
        f"item{item_number}_response": (
            scenario_template[
                f"item{item_number}_response"
            ]
        )
        for item_number in range(1, 11)
    }

    item_scores, total_score = score_audit_responses(
        responses,
    )

    scenario = {
        **scenario_template,
        **{
            f"item{item_number}_score": (
                item_scores[item_number]
            )
            for item_number in range(1, 11)
        },
        "auditc_total_score": total_score,
    }

    return scenario


AUDIT_SCENARIO_LIST = [
    build_audit_scenario(template)
    for template in AUDIT_SCENARIOS.values()
]


# =============================================================================
# 9. WHO AUDIT ZONE BENCHMARK TEXT
# =============================================================================

# WHO AUDIT risk zones sourced from:
# Babor, T. F., Higgins-Biddle, J. C., Saunders, J. B., & Monteiro, M. G.
# AUDIT: The Alcohol Use Disorders Identification Test. Guidelines for Use
# in Primary Care (2nd ed.). Geneva: World Health Organization; 2001.
# WHO/MSD/MSB/01.6a. Box 6 (zones / interventions) and Box 7 (feedback
# elements: present the score, identify risks/consequences, give medical
# advice, solicit patient commitment, identify a goal, give advice and
# encouragement).

WHO_AUDIT_ZONES = {
    "I": {
        "score_min": 0,
        "score_max": 7,
        "label": "Alcohol Education",
    },
    "II": {
        "score_min": 8,
        "score_max": 15,
        "label": "Simple Advice",
    },
    "III": {
        "score_min": 16,
        "score_max": 19,
        "label": (
            "Simple Advice plus Brief Counseling and "
            "Continued Monitoring"
        ),
    },
    "IV": {
        "score_min": 20,
        "score_max": 40,
        "label": (
            "Referral to Specialist for Diagnostic "
            "Evaluation and Treatment"
        ),
    },
}


def lookup_who_audit_zone(auditc_total_score):
    """
    Return WHO zone key and label for a full AUDIT score (0–40).
    """
    score = int(auditc_total_score)

    if score < 0 or score > 40:
        raise ValueError(
            "AUDIT total score must be between 0 and 40."
        )

    for zone_key, zone_info in WHO_AUDIT_ZONES.items():
        if (
            zone_info["score_min"]
            <= score
            <= zone_info["score_max"]
        ):
            return zone_key, zone_info["label"]

    raise ValueError(
        f"No WHO zone found for AUDIT score {score}."
    )


def build_benchmark_text(auditc_total_score):
    """
    Build zone-appropriate benchmark counseling text from WHO Box 6/7.
    """
    zone_key, zone_label = lookup_who_audit_zone(
        auditc_total_score,
    )

    score = int(auditc_total_score)

    if zone_key == "I":
        return (
            f"From what you've told me on the AUDIT, you've scored "
            f"{score} out of 40, which places you in Zone I "
            f"({zone_label}). At this level, your drinking is "
            f"generally lower risk, though alcohol can still affect "
            f"your health, sleep, mood, and medications. I recommend "
            f"learning about safer drinking limits and watching for "
            f"any changes in how much you drink. Would you be willing "
            f"to keep your drinking at or below low-risk levels, and "
            f"set a simple goal such as staying within recommended "
            f"limits this month? I encourage you — small choices now "
            f"help protect your health over time."
        )

    if zone_key == "II":
        return (
            f"From what you've told me on the AUDIT, you've scored "
            f"{score} out of 40, which places you in Zone II "
            f"({zone_label}). This level of drinking can raise your "
            f"risk of health problems, accidents, and strain on "
            f"conditions such as blood pressure, mood, or sleep. My "
            f"medical advice is to cut back to lower-risk limits. "
            f"Would you be willing to commit to reducing how often "
            f"or how much you drink? A clear goal could be fewer "
            f"drinks per occasion or more alcohol-free days each "
            f"week. I encourage you — simple advice and a concrete "
            f"plan can make a real difference."
        )

    if zone_key == "III":
        return (
            f"From what you've told me on the AUDIT, you've scored "
            f"{score} out of 40, which places you in Zone III "
            f"({zone_label}). Drinking at this level is likely "
            f"already causing or increasing harm — including health "
            f"risks, missed responsibilities, and possible "
            f"dependence warning signs. My medical advice is to "
            f"reduce substantially and follow brief counseling with "
            f"continued monitoring. Would you commit to cutting "
            f"down and checking in on your progress? A good goal "
            f"is a specific weekly limit and scheduled follow-up. "
            f"I encourage you — with advice, counseling, and "
            f"ongoing support, change is achievable."
        )

    return (
        f"From what you've told me on the AUDIT, you've scored "
        f"{score} out of 40, which places you in Zone IV "
        f"({zone_label}). This score suggests possible alcohol "
        f"dependence and serious health and safety consequences. "
        f"My medical advice is that you need specialist diagnostic "
        f"evaluation and treatment rather than brief advice alone. "
        f"Would you be willing to accept a referral and take the "
        f"next step toward assessment? The goal is connecting you "
        f"with specialty care promptly. I encourage you — seeking "
        f"help is a strong and important decision for your health."
    )


# Validate that each preset lands in the intended WHO zone.
for _scenario in AUDIT_SCENARIO_LIST:
    _zone_key, _zone_label = lookup_who_audit_zone(
        _scenario["auditc_total_score"],
    )

    if _zone_key != _scenario["who_zone"]:
        raise ValueError(
            f"Scenario {_scenario['scenario_id']} total "
            f"{_scenario['auditc_total_score']} maps to "
            f"Zone {_zone_key}, expected Zone "
            f"{_scenario['who_zone']}."
        )

    if _zone_label != _scenario["who_zone_label"]:
        raise ValueError(
            f"Scenario {_scenario['scenario_id']} zone "
            f"label mismatch."
        )

    _scenario["benchmark_text"] = build_benchmark_text(
        _scenario["auditc_total_score"],
    )

    print(
        f"AUDIT scenario {_scenario['scenario_id']}: "
        f"total={_scenario['auditc_total_score']}, "
        f"Zone {_scenario['who_zone']} "
        f"({_scenario['who_zone_label']})"
    )


# =============================================================================
# 9b. PER-DISEASE REFERENCE SENTENCES (author-written, not WHO-sourced)
# =============================================================================
# WHO does not publish per-disease alcohol-counseling scripts (verified
# against the AUDIT manual and its companion Brief Intervention manual).
# Only the WHO Zone I-IV text in build_benchmark_text() above is WHO-
# sourced. These 20 sentences are authored for this project so the
# reference/benchmark text varies by patient, not just by AUDIT score.

COMORBIDITY_REFERENCE_ADDITIONS = {
    "hypertension": (
        "Because you have hypertension, alcohol can raise "
        "your blood pressure and increase your risk of "
        "stroke and heart disease."
    ),
    "diabetes": (
        "Because you have diabetes, alcohol can cause "
        "dangerous blood sugar swings and increase "
        "complications over time."
    ),
    "depressionbipolar": (
        "Because you have depression or bipolar disorder, "
        "alcohol can worsen mood symptoms, increase suicide "
        "risk, and interfere with psychiatric medications."
    ),
    "anxietydisorder": (
        "Because you have an anxiety disorder, alcohol may "
        "feel calming at first but often worsens anxiety "
        "and sleep problems over time."
    ),
    "heartdisease": (
        "Because you have heart disease, alcohol can worsen "
        "heart function, raise blood pressure, and increase "
        "the risk of arrhythmias."
    ),
    "copd": (
        "Because you have COPD, alcohol can irritate your "
        "airways, worsen breathing, and interact with "
        "medications."
    ),
    "kidneydisease": (
        "Because you have kidney disease, alcohol adds "
        "stress to your kidneys and can worsen fluid and "
        "electrolyte problems."
    ),
    "kidneyfailure": (
        "Because you have kidney failure, alcohol can be "
        "especially harmful and should only be used after "
        "discussion with your kidney doctor."
    ),
    "highcholesterol": (
        "Because you have high cholesterol, regular heavy "
        "drinking can raise triglycerides and worsen "
        "cardiovascular risk."
    ),
    "asthma": (
        "Because you have asthma, alcohol can trigger "
        "symptoms, worsen reflux, and interact with "
        "inhalers or other medicines."
    ),
    "rheumatoidarthritis": (
        "Because you have rheumatoid arthritis, alcohol "
        "can interact with methotrexate and other "
        "arthritis medicines and affect your liver."
    ),
    "osteoarthritis": (
        "Because you have osteoarthritis, alcohol can "
        "increase bleeding risk if you take NSAIDs or "
        "other pain medicines."
    ),
    "connectivetissuedisease": (
        "Because you have connective tissue disease, "
        "alcohol can interact with immune-suppressing "
        "medicines and worsen inflammation."
    ),
    "nervoussystemdisorders": (
        "Because you have a nervous system disorder, "
        "alcohol can worsen symptoms and is unsafe with "
        "many neurologic medications."
    ),
    "headachesmigraine": (
        "Because you have headaches or migraine, alcohol "
        "is a common trigger and can make attacks more "
        "frequent or severe."
    ),
    "adhd": (
        "Because you have ADHD, alcohol can impair "
        "judgment and focus and is unsafe with stimulant "
        "or other ADHD medications."
    ),
    "upperrespinfections": (
        "Because you have had upper respiratory "
        "infections, alcohol can slow healing and "
        "irritate your throat and airways."
    ),
    "upperrespdisease": (
        "Because you have upper respiratory disease, "
        "alcohol can irritate your airways and worsen "
        "cough or breathing symptoms."
    ),
    "spondylosisbackproblems": (
        "Because you have back problems, alcohol can "
        "increase sedation and bleeding risk with pain "
        "medicines such as opioids or NSAIDs."
    ),
    "jointdisorders": (
        "Because you have joint disorders, alcohol can "
        "interact with pain or anti-inflammatory "
        "medicines and worsen sleep or mood."
    ),
}


def build_comorbidity_reference_text(condition, auditc_total_score):
    """
    Combine the WHO zone benchmark text with one sentence per Present
    disease, so the reference text -- and therefore word/semantic
    similarity scoring -- varies by patient, not only by AUDIT score.
    """
    reference_parts = [
        build_benchmark_text(auditc_total_score),
    ]

    for field_name in COMORBIDITY_FIELD_NAMES:
        if condition.get(field_name) != COMORBIDITY_PRESENT_LABEL:
            continue

        addition = COMORBIDITY_REFERENCE_ADDITIONS.get(field_name)

        if addition:
            reference_parts.append(addition)
            continue

        disease_label = format_comorbidity_label(field_name)

        reference_parts.append(
            "Because you have "
            f"{disease_label}, alcohol may pose additional "
            "health risks for you."
        )

    return " ".join(reference_parts)


# =============================================================================
# 10. CONDITION AND PROMPT FUNCTIONS
# =============================================================================

def format_comorbidity_label(field_name):
    """
    Convert a comorbidity field name into a prompt label.
    """
    if field_name in COMORBIDITY_FIELD_LABELS:
        return COMORBIDITY_FIELD_LABELS[field_name]

    return field_name.replace("_", " ").title()


def hac_binary_value_to_label(value):
    """
    Convert a HAC dataset 0/1 value into a prompt label.
    """
    if int(value) == 1:
        return COMORBIDITY_PRESENT_LABEL

    return COMORBIDITY_ABSENT_LABEL


def load_hac_comorbidity_settings():
    """
    Load all HAC disease columns and patient profiles from the Excel file.
    """
    global comorbidity_dimensions
    global COMORBIDITY_FIELD_NAMES
    global COMORBIDITY_FIELD_LABELS
    global hac_loaded_conditions

    if not hac_dataset_path.exists():
        raise FileNotFoundError(
            "HAC dataset not found at: "
            f"{hac_dataset_path}"
        )

    try:
        hac_dataframe = pd.read_excel(
            hac_dataset_path,
            sheet_name=0,
            header=hac_excel_header_row,
        )
    except ImportError as error:
        raise ImportError(
            "Reading the HAC Excel file requires openpyxl. "
            "Install it with: pip install openpyxl"
        ) from error

    hac_comorbidity_columns = [
        column_name
        for column_name in hac_dataframe.columns
        if column_name not in HAC_NON_COMORBIDITY_COLUMNS
    ]

    if not hac_comorbidity_columns:
        raise ValueError(
            "No comorbidity columns were found in the HAC dataset."
        )

    field_names = [
        column_name.lower()
        for column_name in hac_comorbidity_columns
    ]

    comorbidity_dimensions = {
        field_name: [
            COMORBIDITY_ABSENT_LABEL,
            COMORBIDITY_PRESENT_LABEL,
        ]
        for field_name in field_names
    }

    COMORBIDITY_FIELD_NAMES = tuple(field_names)

    COMORBIDITY_FIELD_LABELS = {
        field_name: HAC_COLUMN_LABELS.get(
            field_name,
            field_name.replace("_", " ").title(),
        )
        for field_name in field_names
    }

    comorbidity_values = hac_dataframe[
        hac_comorbidity_columns
    ].apply(
        pd.to_numeric,
        errors="coerce",
    )

    valid_row_mask = comorbidity_values.apply(
        lambda column: column.isin([0, 1])
    ).all(axis=1)

    invalid_row_count = int(
        (~valid_row_mask).sum()
    )

    if invalid_row_count > 0:
        print(
            "Warning: skipped "
            f"{invalid_row_count} HAC rows that were not "
            "pure 0/1 across all comorbidity columns."
        )

    comorbidity_values = (
        comorbidity_values.loc[valid_row_mask]
        .astype(int)
    )
    comorbidity_values.columns = field_names

    if hac_profile_mode == "unique_profiles":
        profile_dataframe = (
            comorbidity_values.drop_duplicates()
        )

    elif hac_profile_mode == "sample":
        profile_dataframe = (
            comorbidity_values.drop_duplicates()
        )

        sample_size = min(
            hac_sample_size,
            len(profile_dataframe),
        )

        profile_dataframe = profile_dataframe.sample(
            n=sample_size,
            random_state=hac_random_seed,
        )

    elif hac_profile_mode == "all_patients":
        profile_dataframe = (
            comorbidity_values.drop_duplicates()
        )

        print(
            "Note: all_patients mode uses each valid HAC row, "
            "but duplicate 0/1 profiles are merged into one "
            "condition before API calls."
        )

    else:
        raise ValueError(
            "hac_profile_mode must be "
            "'unique_profiles', 'sample', or 'all_patients'."
        )

    hac_loaded_conditions = [
        {
            field_name: hac_binary_value_to_label(
                row[field_name]
            )
            for field_name in field_names
        }
        for _, row in profile_dataframe.iterrows()
    ]

    print(
        f"Loaded {len(hac_loaded_conditions)} HAC profiles "
        f"across {len(field_names)} diseases from "
        f"{hac_dataset_path.name}."
    )


def format_condition_summary(condition):
    """
    Format one condition for console output.
    """
    return ", ".join(
        f"{field}={condition[field]}"
        for field in COMORBIDITY_FIELD_NAMES
    )


def validate_condition(condition):
    """
    Validate one HAC disease condition.
    """
    required_fields = set(COMORBIDITY_FIELD_NAMES)

    if not isinstance(condition, dict):
        raise ValueError(
            "Each disease condition must be a dictionary."
        )

    missing_fields = required_fields - set(condition)

    if missing_fields:
        raise ValueError(
            "Condition is missing required fields: "
            f"{sorted(missing_fields)}"
        )

    validated_condition = {}

    for field_name in COMORBIDITY_FIELD_NAMES:
        field_value = condition[field_name]
        allowed_values = comorbidity_dimensions[field_name]

        if (
            not isinstance(field_value, str)
            or field_value not in allowed_values
        ):
            raise ValueError(
                f"Invalid value for {field_name}: "
                f"{field_value}. Allowed: {allowed_values}"
            )

        validated_condition[field_name] = field_value

    return validated_condition


def build_conditions():
    """
    Create HAC disease conditions for the selected mode.
    """
    if condition_mode != "hac_dataset":
        raise ValueError(
            "condition_mode must be 'hac_dataset'."
        )

    load_hac_comorbidity_settings()

    selected_conditions = [
        condition.copy()
        for condition in hac_loaded_conditions
    ]

    if not selected_conditions:
        raise ValueError(
            "The selected condition list is empty."
        )

    validated_conditions = [
        validate_condition(condition)
        for condition in selected_conditions
    ]

    condition_keys = [
        tuple(
            condition[field_name]
            for field_name in COMORBIDITY_FIELD_NAMES
        )
        for condition in validated_conditions
    ]

    if len(condition_keys) != len(set(condition_keys)):
        raise ValueError(
            "The selected disease conditions contain "
            "duplicate combinations."
        )

    return [
        {
            "condition_id": condition_id,
            **condition,
        }
        for condition_id, condition in enumerate(
            validated_conditions,
            start=1,
        )
    ]


def build_prompt(condition, audit_scenario):
    """
    Construct the exact prompt for one HAC disease profile
    and one full WHO AUDIT scenario.
    """
    comorbidity_lines = "\n".join(
        f"{format_comorbidity_label(field_name)}: "
        f"{condition[field_name]}"
        for field_name in COMORBIDITY_FIELD_NAMES
    )

    return f"""

## TASK

Review the patient and determine the most clinically appropriate first thing to say to the patient.

The AUDIT is the Alcohol Use Disorders Identification Test (full 10-item WHO instrument). 

## PATIENT INFORMATION: COMORBIDITIES

{comorbidity_lines}

## PATIENT INFORMATION: AUDIT RESPONSES

Item 1 response: {audit_scenario['item1_response']}
Item 1 score: {audit_scenario['item1_score']}

Item 2 response: {audit_scenario['item2_response']}
Item 2 score: {audit_scenario['item2_score']}

Item 3 response: {audit_scenario['item3_response']}
Item 3 score: {audit_scenario['item3_score']}

Item 4 response: {audit_scenario['item4_response']}
Item 4 score: {audit_scenario['item4_score']}

Item 5 response: {audit_scenario['item5_response']}
Item 5 score: {audit_scenario['item5_score']}

Item 6 response: {audit_scenario['item6_response']}
Item 6 score: {audit_scenario['item6_score']}

Item 7 response: {audit_scenario['item7_response']}
Item 7 score: {audit_scenario['item7_score']}

Item 8 response: {audit_scenario['item8_response']}
Item 8 score: {audit_scenario['item8_score']}

Item 9 response: {audit_scenario['item9_response']}
Item 9 score: {audit_scenario['item9_score']}

Item 10 response: {audit_scenario['item10_response']}
Item 10 score: {audit_scenario['item10_score']}

Total AUDIT score: {audit_scenario['auditc_total_score']}
WHO risk zone: {audit_scenario['who_zone']} ({audit_scenario['who_zone_label']})

### AUDIT ITEMS

### Item 1

How often do you have a drink containing alcohol?

- Never = 0
- Monthly or less = 1
- 2–4 times per month = 2
- 2–3 times per week = 3
- 4 or more times per week = 4

### Item 2

How many standard drinks, each containing 10 grams of alcohol, do you drink
on a typical day when you are drinking?

- 1–2 standard drinks = 0
- 3–4 standard drinks = 1
- 5–6 standard drinks = 2
- 7–9 standard drinks = 3
- 10 or more standard drinks = 4

### Item 3

How often have you had 6 or more standard drinks, equivalent to 60 grams of
alcohol, on a single occasion in the last year?

- Never = 0
- Less than monthly = 1
- Monthly = 2
- Weekly = 3
- Daily or almost daily = 4

### Item 4

How often during the last year have you found that you were not able to stop
drinking once you had started?

- Never = 0
- Less than monthly = 1
- Monthly = 2
- Weekly = 3
- Daily or almost daily = 4

### Item 5

How often during the last year have you failed to do what was normally
expected of you because of drinking?

- Never = 0
- Less than monthly = 1
- Monthly = 2
- Weekly = 3
- Daily or almost daily = 4

### Item 6

How often during the last year have you needed a first drink in the morning
to get yourself going after a heavy drinking session?

- Never = 0
- Less than monthly = 1
- Monthly = 2
- Weekly = 3
- Daily or almost daily = 4

### Item 7

How often during the last year have you had a feeling of guilt or remorse
after drinking?

- Never = 0
- Less than monthly = 1
- Monthly = 2
- Weekly = 3
- Daily or almost daily = 4

### Item 8

How often during the last year have you been unable to remember what
happened the night before because of your drinking?

- Never = 0
- Less than monthly = 1
- Monthly = 2
- Weekly = 3
- Daily or almost daily = 4

### Item 9

Have you or someone else been injured because of your drinking?

- No = 0
- Yes, but not in the last year = 2
- Yes, during the last year = 4

### Item 10

Has a relative, friend, doctor, or other health care worker been concerned
about your drinking or suggested you cut down?

- No = 0
- Yes, but not in the last year = 2
- Yes, during the last year = 4

The total AUDIT score is the sum of the ten item scores and ranges
from 0 to 40.

## OUTPUT

Provide only the exact words that you would say to the patient.

"""


# =============================================================================
# 11. GENERAL HELPER FUNCTIONS
# =============================================================================

def get_field(value, field_name, default=None):
    """
    Safely retrieve a field from a dictionary or object.
    """
    if value is None:
        return default

    if isinstance(value, dict):
        return value.get(
            field_name,
            default,
        )

    return getattr(
        value,
        field_name,
        default,
    )


def timestamp_to_utc(value):
    """
    Convert a Unix timestamp or datetime to ISO-formatted UTC text.
    """
    if value is None:
        return None

    if isinstance(value, datetime):

        if value.tzinfo is None:
            value = value.replace(
                tzinfo=timezone.utc
            )

        return value.astimezone(
            timezone.utc
        ).isoformat()

    if isinstance(value, str):
        return value

    try:
        return datetime.fromtimestamp(
            float(value),
            tz=timezone.utc,
        ).isoformat()

    except (TypeError, ValueError, OSError):
        return None


def object_to_serializable(value):
    """
    Convert an SDK object into a JSON-serializable value.
    """
    if value is None:
        return None

    if hasattr(value, "model_dump"):

        try:
            return value.model_dump(
                mode="json"
            )

        except TypeError:
            return value.model_dump()

    if hasattr(value, "to_dict"):
        return value.to_dict()

    if isinstance(
        value,
        (
            dict,
            list,
            str,
            int,
            float,
            bool,
        )
    ):
        return value

    return str(value)


def object_to_json(value):
    """
    Convert a value into JSON text for storage in one DataFrame cell.
    """
    if value is None:
        return None

    return json.dumps(
        object_to_serializable(value),
        ensure_ascii=False,
    )


def extract_response_text(response):
    """
    Extract visible text from a Responses API response.
    """
    try:

        output_text = response.output_text

        if isinstance(output_text, str):
            return output_text

    except Exception:
        pass

    text_parts = []

    response_output = get_field(
        response,
        "output",
        [],
    ) or []

    for output_item in response_output:

        if get_field(
            output_item,
            "type",
        ) != "message":
            continue

        content_items = get_field(
            output_item,
            "content",
            [],
        ) or []

        for content_item in content_items:

            if get_field(
                content_item,
                "type",
            ) != "output_text":
                continue

            text_value = get_field(
                content_item,
                "text",
            )

            if isinstance(text_value, str):
                text_parts.append(text_value)

    return "".join(text_parts)


def count_words(text):
    """
    Count whitespace-delimited words.
    """
    if not isinstance(text, str) or not text.strip():
        return 0

    return len(text.split())


def count_sentences(text):
    """
    Estimate the number of sentences in visible response text.
    """
    if not isinstance(text, str) or not text.strip():
        return 0

    sentence_parts = re.split(
        r'(?<=[.!?])["”’\']*\s+',
        text.strip(),
    )

    return len(
        [
            part
            for part in sentence_parts
            if part.strip()
        ]
    )


def save_json(file_path, value):
    """
    Save a Python object as formatted UTF-8 JSON.
    """
    with file_path.open(
        "w",
        encoding="utf-8",
    ) as output_file:

        json.dump(
            value,
            output_file,
            ensure_ascii=False,
            indent=2,
        )


# =============================================================================
# 12. LINGUISTIC-DISTANCE HELPER FUNCTIONS
# =============================================================================

def tokenize_for_word_distance(text):
    """
    Normalize text and return word tokens for Levenshtein analysis.
    """
    if not isinstance(text, str) or not text.strip():
        return []

    normalized_text = unicodedata.normalize(
        "NFKC",
        text,
    ).lower()

    normalized_text = (
        normalized_text
        .replace("’", "'")
        .replace("‘", "'")
    )

    return re.findall(
        r"[a-z0-9]+(?:'[a-z0-9]+)?",
        normalized_text,
    )


def calculate_word_distance_metrics(
    reference_text,
    comparison_text,
):
    """
    Calculate raw and normalized word-level Levenshtein measures.
    """
    reference_tokens = tokenize_for_word_distance(
        reference_text
    )

    comparison_tokens = tokenize_for_word_distance(
        comparison_text
    )

    maximum_token_length = max(
        len(reference_tokens),
        len(comparison_tokens),
    )

    if maximum_token_length == 0:
        return np.nan, np.nan, np.nan

    word_edit_distance = Levenshtein.distance(
        reference_tokens,
        comparison_tokens,
    )

    normalized_word_distance = (
        word_edit_distance
        / maximum_token_length
    )

    word_structural_similarity = (
        1.0
        - normalized_word_distance
    )

    return (
        word_edit_distance,
        normalized_word_distance,
        word_structural_similarity,
    )


# =============================================================================
# 13. BUILD CONDITIONS AND CONDITION-SPECIFIC PROMPTS
# =============================================================================

conditions = build_conditions()

condition_prompt_lookup = {}

condition_prompt_metadata = []

for condition in conditions:

    for audit_scenario in AUDIT_SCENARIO_LIST:

        condition_prompt = build_prompt(
            condition,
            audit_scenario,
        )

        condition_prompt_sha256 = hashlib.sha256(
            condition_prompt.encode("utf-8")
        ).hexdigest()

        prompt_key = (
            condition["condition_id"],
            audit_scenario["scenario_id"],
        )

        condition_prompt_lookup[prompt_key] = {
            "prompt": condition_prompt,
            "prompt_sha256": condition_prompt_sha256,
            "scenario_id": audit_scenario["scenario_id"],
        }

        condition_prompt_metadata.append({
            **condition,
            "scenario_id": audit_scenario["scenario_id"],
            "who_zone": audit_scenario["who_zone"],
            "who_zone_label": (
                audit_scenario["who_zone_label"]
            ),
            "auditc_total_score": (
                audit_scenario["auditc_total_score"]
            ),
            "prompt_sha256": condition_prompt_sha256,
            "prompt": condition_prompt,
        })

number_of_conditions = len(conditions)

number_of_audit_scenarios = len(AUDIT_SCENARIO_LIST)

total_planned_requests = (
    number_of_conditions
    * number_of_audit_scenarios
    * num_iterations
)

print("=" * 80)
print("PLANNED PILOT RUN")
print("=" * 80)

print(
    f"Condition mode: "
    f"{condition_mode}"
)

print(
    f"Number of disease conditions: "
    f"{number_of_conditions}"
)

print(
    f"Number of AUDIT scenarios: "
    f"{number_of_audit_scenarios}"
)

print(
    f"Iterations per condition × scenario: "
    f"{num_iterations}"
)

print(
    f"Total planned API requests: "
    f"{total_planned_requests}"
)

print("\nAUDIT scenarios:")

for audit_scenario in AUDIT_SCENARIO_LIST:

    print(
        f"  {audit_scenario['scenario_id']}: "
        f"score={audit_scenario['auditc_total_score']}, "
        f"Zone {audit_scenario['who_zone']} "
        f"({audit_scenario['who_zone_label']})"
    )

print("\nConditions:")

for condition in conditions:

    print(
        f"  {condition['condition_id']}: "
        f"{format_condition_summary(condition)}"
    )


# =============================================================================
# 14. DEFINE RUN IDENTIFIERS AND OUTPUT FILES
# =============================================================================

output_directory.mkdir(
    parents=True,
    exist_ok=True,
)

run_id = str(
    uuid.uuid4()
)

run_start_utc = datetime.now(
    timezone.utc
)

run_timestamp = run_start_utc.strftime(
    "%Y%m%dT%H%M%SZ"
)

file_prefix = (
    f"auditc_{run_timestamp}_{run_id[:8]}"
)

checkpoint_csv_path = output_directory / (
    f"{file_prefix}_checkpoint.csv"
)

raw_csv_path = output_directory / (
    f"{file_prefix}_raw_responses.csv"
)

enriched_csv_path = output_directory / (
    f"{file_prefix}_responses_with_linguistic_distances.csv"
)

raw_jsonl_path = output_directory / (
    f"{file_prefix}_raw_api_responses.jsonl"
)

run_metadata_path = output_directory / (
    f"{file_prefix}_run_metadata.json"
)


# =============================================================================
# 15. DEFINE RAW DATAFRAME COLUMN ORDER
# =============================================================================

raw_primary_columns = [
    "condition_id",
    "condition_iteration",
    "scenario_id",
    "who_zone",
    "who_zone_label",
    *COMORBIDITY_FIELD_NAMES,

    "auditc_item1_response",
    "auditc_item1_score",
    "auditc_item2_response",
    "auditc_item2_score",
    "auditc_item3_response",
    "auditc_item3_score",
    "auditc_item4_response",
    "auditc_item4_score",
    "auditc_item5_response",
    "auditc_item5_score",
    "auditc_item6_response",
    "auditc_item6_score",
    "auditc_item7_response",
    "auditc_item7_score",
    "auditc_item8_response",
    "auditc_item8_score",
    "auditc_item9_response",
    "auditc_item9_score",
    "auditc_item10_response",
    "auditc_item10_score",
    "auditc_total_score",

    "benchmark_text",

    "model_response",
    "analysis_eligible",
]

raw_secondary_columns = [
    "overall_request_number",
    "condition_mode",

    "response_word_count",
    "response_character_count",
    "response_sentence_count",

    "requested_model",
    "returned_model",
    "prompt_version",
    "condition_prompt_sha256",

    "requested_temperature",
    "returned_temperature",
    "requested_top_p",
    "returned_top_p",
    "requested_reasoning_effort",
    "returned_reasoning_effort",
    "requested_max_output_tokens",
    "returned_max_output_tokens",
    "returned_service_tier",

    "input_tokens",
    "cached_input_tokens",
    "output_tokens",
    "reasoning_tokens",
    "total_tokens",

    "request_start_utc",
    "request_end_utc",
    "elapsed_seconds",
    "api_created_at_utc",
    "api_completed_at_utc",

    "request_completed_without_exception",
    "response_status",

    "request_used_previous_response_id",
    "request_used_conversation",
    "returned_previous_response_id",
    "returned_conversation_id",

    "response_id",
    "request_id",
    "run_id",

    "incomplete_details_json",
    "response_error_json",

    "exception_type",
    "exception_message",
    "exception_status_code",

    "python_version",
    "openai_package_version",
    "pandas_package_version",
]

raw_column_order = (
    raw_primary_columns
    + raw_secondary_columns
)


# =============================================================================
# 16. RECORD AND DATAFRAME FUNCTIONS
# =============================================================================

def create_base_record(
    condition,
    condition_iteration,
    overall_request_number,
    condition_prompt_sha256,
    audit_scenario,
):
    """
    Create one response record before the API call.
    """
    return {
        "condition_id": condition["condition_id"],

        "condition_iteration": condition_iteration,

        "scenario_id": audit_scenario["scenario_id"],

        "who_zone": audit_scenario["who_zone"],

        "who_zone_label": (
            audit_scenario["who_zone_label"]
        ),

        **{
            field_name: condition[field_name]
            for field_name in COMORBIDITY_FIELD_NAMES
        },

        "auditc_item1_response": (
            audit_scenario["item1_response"]
        ),

        "auditc_item1_score": (
            audit_scenario["item1_score"]
        ),

        "auditc_item2_response": (
            audit_scenario["item2_response"]
        ),

        "auditc_item2_score": (
            audit_scenario["item2_score"]
        ),

        "auditc_item3_response": (
            audit_scenario["item3_response"]
        ),

        "auditc_item3_score": (
            audit_scenario["item3_score"]
        ),

        "auditc_item4_response": (
            audit_scenario["item4_response"]
        ),

        "auditc_item4_score": (
            audit_scenario["item4_score"]
        ),

        "auditc_item5_response": (
            audit_scenario["item5_response"]
        ),

        "auditc_item5_score": (
            audit_scenario["item5_score"]
        ),

        "auditc_item6_response": (
            audit_scenario["item6_response"]
        ),

        "auditc_item6_score": (
            audit_scenario["item6_score"]
        ),

        "auditc_item7_response": (
            audit_scenario["item7_response"]
        ),

        "auditc_item7_score": (
            audit_scenario["item7_score"]
        ),

        "auditc_item8_response": (
            audit_scenario["item8_response"]
        ),

        "auditc_item8_score": (
            audit_scenario["item8_score"]
        ),

        "auditc_item9_response": (
            audit_scenario["item9_response"]
        ),

        "auditc_item9_score": (
            audit_scenario["item9_score"]
        ),

        "auditc_item10_response": (
            audit_scenario["item10_response"]
        ),

        "auditc_item10_score": (
            audit_scenario["item10_score"]
        ),

        "auditc_total_score": (
            audit_scenario["auditc_total_score"]
        ),

        "benchmark_text": (
            build_comorbidity_reference_text(
                condition,
                audit_scenario["auditc_total_score"],
            )
        ),

        "model_response": None,

        "analysis_eligible": False,

        "overall_request_number": (
            overall_request_number
        ),

        "condition_mode": condition_mode,

        "response_word_count": None,

        "response_character_count": None,

        "response_sentence_count": None,

        "requested_model": model_name,

        "returned_model": None,

        "prompt_version": prompt_version,

        "condition_prompt_sha256": (
            condition_prompt_sha256
        ),

        "requested_temperature": temperature,

        "returned_temperature": None,

        "requested_top_p": top_p,

        "returned_top_p": None,

        "requested_reasoning_effort": (
            reasoning_effort
        ),

        "returned_reasoning_effort": None,

        "requested_max_output_tokens": (
            max_output_tokens
        ),

        "returned_max_output_tokens": None,

        "returned_service_tier": None,

        "input_tokens": None,

        "cached_input_tokens": None,

        "output_tokens": None,

        "reasoning_tokens": None,

        "total_tokens": None,

        "request_start_utc": None,

        "request_end_utc": None,

        "elapsed_seconds": None,

        "api_created_at_utc": None,

        "api_completed_at_utc": None,

        "request_completed_without_exception": False,

        "response_status": None,

        # These remain False because the API request supplies neither field.
        "request_used_previous_response_id": False,

        "request_used_conversation": False,

        "returned_previous_response_id": None,

        "returned_conversation_id": None,

        "response_id": None,

        "request_id": None,

        "run_id": run_id,

        "incomplete_details_json": None,

        "response_error_json": None,

        "exception_type": None,

        "exception_message": None,

        "exception_status_code": None,

        "python_version": (
            platform.python_version()
        ),

        "openai_package_version": (
            openai.__version__
        ),

        "pandas_package_version": (
            pd.__version__
        ),
    }


def build_raw_dataframe(record_list):
    """
    Build the combined raw DataFrame with the intended column order.
    """
    dataframe = pd.DataFrame(
        record_list
    )

    for column in raw_column_order:

        if column not in dataframe.columns:
            dataframe[column] = pd.NA

    return dataframe[
        raw_column_order
    ]


# =============================================================================
# 17. SAVE INITIAL RUN METADATA
# =============================================================================

run_metadata = {
    "run_id": run_id,

    "run_start_utc": (
        run_start_utc.isoformat()
    ),

    "condition_mode": condition_mode,

    "number_of_conditions": (
        number_of_conditions
    ),

    "number_of_audit_scenarios": (
        number_of_audit_scenarios
    ),

    "iterations_per_condition_scenario": (
        num_iterations
    ),

    "total_planned_requests": (
        total_planned_requests
    ),

    "conditions": conditions,

    "condition_prompts": (
        condition_prompt_metadata
    ),

    "audit_scenarios": [
        {
            "scenario_id": scenario["scenario_id"],
            "who_zone": scenario["who_zone"],
            "who_zone_label": (
                scenario["who_zone_label"]
            ),
            "item1_response": (
                scenario["item1_response"]
            ),
            "item1_score": scenario["item1_score"],
            "item2_response": (
                scenario["item2_response"]
            ),
            "item2_score": scenario["item2_score"],
            "item3_response": (
                scenario["item3_response"]
            ),
            "item3_score": scenario["item3_score"],
            "item4_response": (
                scenario["item4_response"]
            ),
            "item4_score": scenario["item4_score"],
            "item5_response": (
                scenario["item5_response"]
            ),
            "item5_score": scenario["item5_score"],
            "item6_response": (
                scenario["item6_response"]
            ),
            "item6_score": scenario["item6_score"],
            "item7_response": (
                scenario["item7_response"]
            ),
            "item7_score": scenario["item7_score"],
            "item8_response": (
                scenario["item8_response"]
            ),
            "item8_score": scenario["item8_score"],
            "item9_response": (
                scenario["item9_response"]
            ),
            "item9_score": scenario["item9_score"],
            "item10_response": (
                scenario["item10_response"]
            ),
            "item10_score": (
                scenario["item10_score"]
            ),
            "total_score": (
                scenario["auditc_total_score"]
            ),
            "benchmark_text": (
                scenario["benchmark_text"]
            ),
        }
        for scenario in AUDIT_SCENARIO_LIST
    ],

    "who_audit_zones": WHO_AUDIT_ZONES,

    "request_configuration": {
        "model": model_name,

        "temperature": temperature,

        "top_p": top_p,

        "reasoning_effort": reasoning_effort,

        "max_output_tokens": max_output_tokens,

        "seconds_between_requests": (
            seconds_between_requests
        ),

        "uses_previous_response_id": False,

        "uses_conversation": False,
    },

    "prompt_version": prompt_version,

    "software_versions": {
        "python": platform.python_version(),

        "openai": openai.__version__,

        "pandas": pd.__version__,

        "numpy": np.__version__,

        "rapidfuzz": rapidfuzz.__version__,

        "sentence_transformers": (
            sentence_transformers.__version__
        ),
    },
}

save_json(
    run_metadata_path,
    run_metadata,
)


# =============================================================================
# 18. COLLECT INDEPENDENT OPENAI RESPONSES
# =============================================================================

records = []

overall_request_number = 0

with raw_jsonl_path.open(
    "w",
    encoding="utf-8",
) as raw_jsonl_file:

    for condition in conditions:

        condition_id = condition[
            "condition_id"
        ]

        for audit_scenario in AUDIT_SCENARIO_LIST:

            prompt_key = (
                condition_id,
                audit_scenario["scenario_id"],
            )

            condition_prompt = (
                condition_prompt_lookup[
                    prompt_key
                ]["prompt"]
            )

            condition_prompt_sha256 = (
                condition_prompt_lookup[
                    prompt_key
                ]["prompt_sha256"]
            )

            for condition_iteration in range(
                1,
                num_iterations + 1,
            ):

                overall_request_number += 1

                record = create_base_record(
                    condition=condition,

                    condition_iteration=(
                        condition_iteration
                    ),

                    overall_request_number=(
                        overall_request_number
                    ),

                    condition_prompt_sha256=(
                        condition_prompt_sha256
                    ),

                    audit_scenario=audit_scenario,
                )

                request_start_datetime = datetime.now(
                    timezone.utc
                )

                request_start_counter = (
                    time.perf_counter()
                )

                record["request_start_utc"] = (
                    request_start_datetime.isoformat()
                )

                try:

                    request_arguments = {
                        "model": model_name,

                        "input": condition_prompt,
                    }

                    if temperature is not None:

                        request_arguments[
                            "temperature"
                        ] = temperature

                    if top_p is not None:

                        request_arguments[
                            "top_p"
                        ] = top_p

                    if max_output_tokens is not None:

                        request_arguments[
                            "max_output_tokens"
                        ] = max_output_tokens

                    if reasoning_effort is not None:

                        request_arguments[
                            "reasoning"
                        ] = {
                            "effort": reasoning_effort
                        }

                    # No previous_response_id is supplied.
                    # No conversation identifier is supplied.
                    # No prior messages are supplied.
                    # Each call is an independent response request.

                    response = client.responses.create(
                        **request_arguments
                    )

                    request_end_datetime = datetime.now(
                        timezone.utc
                    )

                    elapsed_seconds = (
                        time.perf_counter()
                        - request_start_counter
                    )

                    response_text = (
                        extract_response_text(
                            response
                        )
                        or ""
                    )

                    response_status = get_field(
                        response,
                        "status",
                    )

                    usage = get_field(
                        response,
                        "usage",
                    )

                    input_token_details = get_field(
                        usage,
                        "input_tokens_details",
                    )

                    output_token_details = get_field(
                        usage,
                        "output_tokens_details",
                    )

                    returned_reasoning = get_field(
                        response,
                        "reasoning",
                    )

                    returned_conversation = get_field(
                        response,
                        "conversation",
                    )

                    if isinstance(
                        returned_conversation,
                        str,
                    ):

                        returned_conversation_id = (
                            returned_conversation
                        )

                    else:

                        returned_conversation_id = get_field(
                            returned_conversation,
                            "id",
                        )

                    record.update({
                        "model_response": (
                            response_text
                        ),

                        "analysis_eligible": (
                            response_status == "completed"
                            and bool(
                                response_text.strip()
                            )
                        ),

                        "response_word_count": (
                            count_words(
                                response_text
                            )
                        ),

                        "response_character_count": (
                            len(response_text)
                        ),

                        "response_sentence_count": (
                            count_sentences(
                                response_text
                            )
                        ),

                        "returned_model": get_field(
                            response,
                            "model",
                        ),

                        "returned_temperature": get_field(
                            response,
                            "temperature",
                        ),

                        "returned_top_p": get_field(
                            response,
                            "top_p",
                        ),

                        "returned_reasoning_effort": get_field(
                            returned_reasoning,
                            "effort",
                        ),

                        "returned_max_output_tokens": get_field(
                            response,
                            "max_output_tokens",
                        ),

                        "returned_service_tier": get_field(
                            response,
                            "service_tier",
                        ),

                        "input_tokens": get_field(
                            usage,
                            "input_tokens",
                        ),

                        "cached_input_tokens": get_field(
                            input_token_details,
                            "cached_tokens",
                        ),

                        "output_tokens": get_field(
                            usage,
                            "output_tokens",
                        ),

                        "reasoning_tokens": get_field(
                            output_token_details,
                            "reasoning_tokens",
                        ),

                        "total_tokens": get_field(
                            usage,
                            "total_tokens",
                        ),

                        "request_end_utc": (
                            request_end_datetime.isoformat()
                        ),

                        "elapsed_seconds": (
                            elapsed_seconds
                        ),

                        "api_created_at_utc": (
                            timestamp_to_utc(
                                get_field(
                                    response,
                                    "created_at",
                                )
                            )
                        ),

                        "api_completed_at_utc": (
                            timestamp_to_utc(
                                get_field(
                                    response,
                                    "completed_at",
                                )
                            )
                        ),

                        "request_completed_without_exception": True,

                        "response_status": (
                            response_status
                        ),

                        "returned_previous_response_id": get_field(
                            response,
                            "previous_response_id",
                        ),

                        "returned_conversation_id": (
                            returned_conversation_id
                        ),

                        "response_id": get_field(
                            response,
                            "id",
                        ),

                        "request_id": get_field(
                            response,
                            "_request_id",
                        ),

                        "incomplete_details_json": object_to_json(
                            get_field(
                                response,
                                "incomplete_details",
                            )
                        ),

                        "response_error_json": object_to_json(
                            get_field(
                                response,
                                "error",
                            )
                        ),
                    })

                    raw_response_entry = {
                        "run_id": run_id,

                        "condition_id": (
                            condition_id
                        ),

                        "scenario_id": (
                            audit_scenario[
                                "scenario_id"
                            ]
                        ),

                        "who_zone": (
                            audit_scenario[
                                "who_zone"
                            ]
                        ),

                        "condition_iteration": (
                            condition_iteration
                        ),

                        "overall_request_number": (
                            overall_request_number
                        ),

                        **{
                            field_name: condition[field_name]
                            for field_name in COMORBIDITY_FIELD_NAMES
                        },

                        "condition_prompt_sha256": (
                            condition_prompt_sha256
                        ),

                        "response": (
                            object_to_serializable(
                                response
                            )
                        ),
                    }

                    raw_jsonl_file.write(
                        json.dumps(
                            raw_response_entry,
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

                    raw_jsonl_file.flush()

                    print(
                        f"Request {overall_request_number} of "
                        f"{total_planned_requests} completed | "
                        f"condition {condition_id}, "
                        f"scenario {audit_scenario['scenario_id']}, "
                        f"iteration {condition_iteration}"
                    )

                    print(
                        f"Response: "
                        f"{response_text}\n"
                    )

                except Exception as exception:

                    request_end_datetime = datetime.now(
                        timezone.utc
                    )

                    elapsed_seconds = (
                        time.perf_counter()
                        - request_start_counter
                    )

                    record.update({
                        "request_end_utc": (
                            request_end_datetime.isoformat()
                        ),

                        "elapsed_seconds": (
                            elapsed_seconds
                        ),

                        "request_completed_without_exception": False,

                        "request_id": get_field(
                            exception,
                            "request_id",
                        ),

                        "exception_type": (
                            type(exception).__name__
                        ),

                        "exception_message": (
                            str(exception)
                        ),

                        "exception_status_code": get_field(
                            exception,
                            "status_code",
                        ),
                    })

                    raw_error_entry = {
                        "run_id": run_id,

                        "condition_id": condition_id,

                        "scenario_id": (
                            audit_scenario[
                                "scenario_id"
                            ]
                        ),

                        "who_zone": (
                            audit_scenario[
                                "who_zone"
                            ]
                        ),

                        "condition_iteration": (
                            condition_iteration
                        ),

                        "overall_request_number": (
                            overall_request_number
                        ),

                        **{
                            field_name: condition[field_name]
                            for field_name in COMORBIDITY_FIELD_NAMES
                        },

                        "condition_prompt_sha256": (
                            condition_prompt_sha256
                        ),

                        "exception_type": (
                            type(exception).__name__
                        ),

                        "exception_message": (
                            str(exception)
                        ),

                        "exception_status_code": get_field(
                            exception,
                            "status_code",
                        ),

                        "request_id": get_field(
                            exception,
                            "request_id",
                        ),
                    }

                    raw_jsonl_file.write(
                        json.dumps(
                            raw_error_entry,
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

                    raw_jsonl_file.flush()

                    print(
                        f"Request {overall_request_number} of "
                        f"{total_planned_requests} failed | "
                        f"condition {condition_id}, "
                        f"scenario {audit_scenario['scenario_id']}, "
                        f"iteration {condition_iteration}"
                    )

                    print(
                        f"{type(exception).__name__}: "
                        f"{exception}\n"
                    )

                records.append(record)

                checkpoint_df = build_raw_dataframe(
                    records
                )

                checkpoint_df.to_csv(
                    checkpoint_csv_path,
                    index=False,
                    encoding="utf-8-sig",
                )

                if (
                    seconds_between_requests > 0
                    and overall_request_number
                    < total_planned_requests
                ):

                    time.sleep(
                        seconds_between_requests
                    )


# =============================================================================
# 19. SAVE THE COMBINED RAW RESPONSE DATASET
# =============================================================================

results_df = build_raw_dataframe(
    records
)

results_df.to_csv(
    raw_csv_path,
    index=False,
    encoding="utf-8-sig",
)

if checkpoint_csv_path.exists():
    checkpoint_csv_path.unlink()



# =============================================================================
# 20. INITIALIZE LINGUISTIC-DISTANCE COLUMNS
# =============================================================================

results_df[
    "word_edit_distance"
] = pd.Series(
    pd.NA,
    index=results_df.index,
    dtype="Int64",
)

results_df[
    "normalized_word_distance"
] = np.nan

results_df[
    "word_structural_similarity"
] = np.nan

results_df[
    "semantic_cosine_similarity"
] = np.nan

results_df[
    "semantic_cosine_distance"
] = np.nan

# benchmark_text is already present per row from create_base_record.


# =============================================================================
# 21. IDENTIFY RESPONSES ELIGIBLE FOR LINGUISTIC ANALYSIS
# =============================================================================

valid_response_mask = (
    results_df[
        "analysis_eligible"
    ]
    .fillna(False)
    .astype(bool)
    &
    results_df[
        "model_response"
    ].apply(
        lambda value: (
            isinstance(value, str)
            and bool(value.strip())
        )
    )
)

valid_response_indices = results_df.index[
    valid_response_mask
]


# =============================================================================
# 22. CALCULATE WORD-LEVEL LEVENSHTEIN METRICS
# =============================================================================

for row_index in valid_response_indices:

    response_text = results_df.at[
        row_index,
        "model_response",
    ]

    row_benchmark_text = results_df.at[
        row_index,
        "benchmark_text",
    ]

    (
        word_edit_distance,
        normalized_word_distance,
        word_structural_similarity,
    ) = calculate_word_distance_metrics(
        reference_text=row_benchmark_text,
        comparison_text=response_text,
    )

    results_df.at[
        row_index,
        "word_edit_distance",
    ] = word_edit_distance

    results_df.at[
        row_index,
        "normalized_word_distance",
    ] = normalized_word_distance

    results_df.at[
        row_index,
        "word_structural_similarity",
    ] = word_structural_similarity


# =============================================================================
# 23. CALCULATE SENTENCE-BERT SEMANTIC METRICS
# =============================================================================

semantic_analysis_error = None

if len(valid_response_indices) > 0:

    try:

        print(
            "\nLoading the SentenceTransformer model..."
        )

        semantic_model = SentenceTransformer(
            semantic_model_name
        )

        valid_response_texts = (
            results_df.loc[
                valid_response_indices,
                "model_response",
            ]
            .astype(str)
            .tolist()
        )

        row_benchmark_texts = (
            results_df.loc[
                valid_response_indices,
                "benchmark_text",
            ]
            .astype(str)
            .tolist()
        )

        benchmark_embeddings = semantic_model.encode(
            row_benchmark_texts,

            batch_size=semantic_batch_size,

            convert_to_tensor=True,

            normalize_embeddings=True,

            show_progress_bar=False,
        )

        response_embeddings = semantic_model.encode(
            valid_response_texts,

            batch_size=semantic_batch_size,

            convert_to_tensor=True,

            normalize_embeddings=True,

            show_progress_bar=True,
        )

        # Paired cosine similarity: each response vs its zone benchmark.
        semantic_cosine_similarities = (
            (benchmark_embeddings * response_embeddings)
            .sum(dim=1)
            .detach()
            .cpu()
            .numpy()
        )

        semantic_cosine_similarities = np.clip(
            semantic_cosine_similarities,
            -1.0,
            1.0,
        )

        semantic_cosine_distances = (
            1.0
            - semantic_cosine_similarities
        )

        results_df.loc[
            valid_response_indices,
            "semantic_cosine_similarity",
        ] = semantic_cosine_similarities

        results_df.loc[
            valid_response_indices,
            "semantic_cosine_distance",
        ] = semantic_cosine_distances

    except Exception as exception:

        semantic_analysis_error = (
            f"{type(exception).__name__}: "
            f"{exception}"
        )

        print(
            "\nSentence-BERT analysis failed. "
            "The raw responses and Levenshtein results "
            "will still be saved."
        )

        print(
            semantic_analysis_error
        )

else:

    print(
        "\nNo eligible responses were available "
        "for linguistic analysis."
    )


# =============================================================================
# 24. ADD LINGUISTIC-ANALYSIS METADATA
# =============================================================================

results_df[
    "semantic_model_name"
] = semantic_model_name

results_df[
    "semantic_batch_size"
] = semantic_batch_size

results_df[
    "word_normalization_version"
] = word_normalization_version

results_df[
    "rapidfuzz_package_version"
] = rapidfuzz.__version__

results_df[
    "sentence_transformers_package_version"
] = sentence_transformers.__version__


# =============================================================================
# 25. REORDER AND SAVE THE ENRICHED DATASET
# =============================================================================

enriched_left_columns = [
    "condition_id",
    "condition_iteration",
    "scenario_id",
    "who_zone",
    "who_zone_label",
    *COMORBIDITY_FIELD_NAMES,

    "auditc_item1_response",
    "auditc_item1_score",
    "auditc_item2_response",
    "auditc_item2_score",
    "auditc_item3_response",
    "auditc_item3_score",
    "auditc_item4_response",
    "auditc_item4_score",
    "auditc_item5_response",
    "auditc_item5_score",
    "auditc_item6_response",
    "auditc_item6_score",
    "auditc_item7_response",
    "auditc_item7_score",
    "auditc_item8_response",
    "auditc_item8_score",
    "auditc_item9_response",
    "auditc_item9_score",
    "auditc_item10_response",
    "auditc_item10_score",
    "auditc_total_score",

    "model_response",

    "word_edit_distance",
    "normalized_word_distance",
    "word_structural_similarity",

    "semantic_cosine_similarity",
    "semantic_cosine_distance",

    "benchmark_text",
    "analysis_eligible",
]

remaining_columns = [
    column
    for column in results_df.columns
    if column not in enriched_left_columns
]

results_df = results_df[
    enriched_left_columns
    + remaining_columns
]

results_df.to_csv(
    enriched_csv_path,
    index=False,
    encoding="utf-8-sig",
)


# =============================================================================
# 26. UPDATE RUN METADATA
# =============================================================================

run_end_utc = datetime.now(
    timezone.utc
)

run_metadata[
    "run_end_utc"
] = run_end_utc.isoformat()

run_metadata[
    "response_collection"
] = {
    "number_requested": (
        total_planned_requests
    ),

    "number_completed_without_exception": int(
        results_df[
            "request_completed_without_exception"
        ]
        .fillna(False)
        .sum()
    ),

    "number_eligible_for_analysis": int(
        valid_response_mask.sum()
    ),

    "raw_csv_path": str(
        raw_csv_path
    ),

    "raw_jsonl_path": str(
        raw_jsonl_path
    ),

    "enriched_csv_path": str(
        enriched_csv_path
    ),
}

run_metadata[
    "linguistic_analysis"
] = {
    "benchmark_texts_by_scenario": {
        scenario["scenario_id"]: scenario[
            "benchmark_text"
        ]
        for scenario in AUDIT_SCENARIO_LIST
    },

    "semantic_model_name": (
        semantic_model_name
    ),

    "semantic_batch_size": (
        semantic_batch_size
    ),

    "word_normalization_version": (
        word_normalization_version
    ),

    "rapidfuzz_package_version": (
        rapidfuzz.__version__
    ),

    "sentence_transformers_package_version": (
        sentence_transformers.__version__
    ),

    "number_of_analyzed_responses": int(
        valid_response_mask.sum()
    ),

    "semantic_analysis_error": (
        semantic_analysis_error
    ),

    "analysis_completed_utc": (
        run_end_utc.isoformat()
    ),
}

save_json(
    run_metadata_path,
    run_metadata,
)


# =============================================================================
# 27. DISPLAY THE MAIN RESULTS
# =============================================================================

print(
    "\n"
    + "=" * 80
)

print(
    "PILOT RESPONSE COLLECTION AND "
    "LINGUISTIC ANALYSIS COMPLETE"
)

print("=" * 80)

print(
    f"\nNumber of disease conditions: "
    f"{number_of_conditions}"
)

print(
    f"Number of AUDIT scenarios: "
    f"{number_of_audit_scenarios}"
)

print(
    f"Iterations per condition × scenario: "
    f"{num_iterations}"
)

print(
    f"Total requested responses: "
    f"{total_planned_requests}"
)

print(
    "Responses eligible for linguistic analysis: "
    f"{int(valid_response_mask.sum())}"
)

print(
    f"\nRaw response dataset:\n"
    f"{raw_csv_path}"
)

print(
    f"\nEnriched response dataset:\n"
    f"{enriched_csv_path}"
)

print(
    f"\nRaw API response archive:\n"
    f"{raw_jsonl_path}"
)

print(
    f"\nRun metadata and exact prompts:\n"
    f"{run_metadata_path}"
)

print(
    "\nMain result columns:\n"
)

print(
    results_df[
        [
            "condition_id",
            "condition_iteration",
            "scenario_id",
            "who_zone",
            "auditc_total_score",
            "model_response",
            "word_edit_distance",
            "normalized_word_distance",
            "word_structural_similarity",
            "semantic_cosine_similarity",
            "semantic_cosine_distance",
        ]
    ].to_string(
        index=False
    )
)
        
        
        
        
        
        