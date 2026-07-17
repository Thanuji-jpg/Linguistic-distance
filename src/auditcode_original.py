"""
Original AUDIT-C LLM pilot script, adapted to use HAC disease profiles
instead of demographic (age/race/sex) conditions.

Keeps the original simple prompt style and single benchmark_text.
For the full SBIRT + clinical coverage pipeline, use auditcode.py.

Before running: set your API key.
"""

#AUDIT C Testing

import hashlib
import json
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


client = OpenAI(api_key='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')



# =============================================================================
# 2. EDITABLE RUN SETTINGS
# =============================================================================

model_name = "gpt-5.1"

# Number of independent responses requested for EACH disease profile.
num_iterations = 2

# Only HAC disease profiles are supported in this adapted original script.
condition_mode = "hac_dataset"

# Change this label whenever the prompt wording changes.
prompt_version = "auditc_prompt_v2_hac_diseases"

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
# 4. EDITABLE AUDIT-C RESPONSES
# =============================================================================

# Each value must exactly match one option in its scoring dictionary below.

auditc_item1_response = "2–3 times per week"

auditc_item2_response = "7–9 standard drinks"

auditc_item3_response = "Weekly"


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
# 7. AUDIT-C SCORING DEFINITIONS
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


# =============================================================================
# 8. VALIDATE AUDIT-C SETTINGS AND CALCULATE SCORES
# =============================================================================

if not isinstance(num_iterations, int) or num_iterations < 1:
    raise ValueError(
        "num_iterations must be an integer of at least 1."
    )

if auditc_item1_response not in auditc_item1_scoring:
    raise ValueError(
        "Invalid AUDIT-C Item 1 response. Select one of: "
        f"{list(auditc_item1_scoring.keys())}"
    )

if auditc_item2_response not in auditc_item2_scoring:
    raise ValueError(
        "Invalid AUDIT-C Item 2 response. Select one of: "
        f"{list(auditc_item2_scoring.keys())}"
    )

if auditc_item3_response not in auditc_item3_scoring:
    raise ValueError(
        "Invalid AUDIT-C Item 3 response. Select one of: "
        f"{list(auditc_item3_scoring.keys())}"
    )

auditc_item1_score = auditc_item1_scoring[
    auditc_item1_response
]

auditc_item2_score = auditc_item2_scoring[
    auditc_item2_response
]

auditc_item3_score = auditc_item3_scoring[
    auditc_item3_response
]

auditc_total_score = (
    auditc_item1_score
    + auditc_item2_score
    + auditc_item3_score
)


# =============================================================================
# 9. DEFINE THE BENCHMARK TEXT
# =============================================================================

benchmark_text = (
    f"From what you’ve told me, on the quiz here, you’ve scored "
    f"{auditc_total_score}, which means that your drinking may cause you "
    f"health problems in the future or may be harming your health."
)


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


def build_prompt(condition):
    """
    Construct the exact prompt for one HAC disease condition.
    """
    comorbidity_lines = "\n".join(
        f"{format_comorbidity_label(field_name)}: "
        f"{condition[field_name]}"
        for field_name in COMORBIDITY_FIELD_NAMES
    )

    return f"""

## TASK

Review the patient and determine the most clinically appropriate first thing to say to the patient.

The AUDIT-C is the three-item consumption component of the Alcohol Use Disorders Identification Test. 

## PATIENT INFORMATION: COMORBIDITIES

{comorbidity_lines}

## PATIENT INFORMATION: AUDIT-C RESPONSES

Item 1 response: {auditc_item1_response}
Item 1 score: {auditc_item1_score}

Item 2 response: {auditc_item2_response}
Item 2 score: {auditc_item2_score}

Item 3 response: {auditc_item3_response}
Item 3 score: {auditc_item3_score}

Total AUDIT-C score: {auditc_total_score}

### AUDIT-C ITEMS

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

The total AUDIT-C score is the sum of the three item scores and ranges
from 0 to 12.

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

    condition_prompt = build_prompt(
        condition,
    )

    condition_prompt_sha256 = hashlib.sha256(
        condition_prompt.encode("utf-8")
    ).hexdigest()

    condition_prompt_lookup[
        condition["condition_id"]
    ] = {
        "prompt": condition_prompt,
        "prompt_sha256": condition_prompt_sha256,
    }

    condition_prompt_metadata.append({
        **condition,
        "prompt_sha256": condition_prompt_sha256,
        "prompt": condition_prompt,
    })

number_of_conditions = len(conditions)

total_planned_requests = (
    number_of_conditions
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
    f"Iterations per condition: "
    f"{num_iterations}"
)

print(
    f"Total planned API requests: "
    f"{total_planned_requests}"
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
    *COMORBIDITY_FIELD_NAMES,

    "auditc_item1_response",
    "auditc_item1_score",
    "auditc_item2_response",
    "auditc_item2_score",
    "auditc_item3_response",
    "auditc_item3_score",
    "auditc_total_score",

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
):
    """
    Create one response record before the API call.
    """
    return {
        "condition_id": condition["condition_id"],

        "condition_iteration": condition_iteration,

        **{
            field_name: condition[field_name]
            for field_name in COMORBIDITY_FIELD_NAMES
        },

        "auditc_item1_response": (
            auditc_item1_response
        ),

        "auditc_item1_score": (
            auditc_item1_score
        ),

        "auditc_item2_response": (
            auditc_item2_response
        ),

        "auditc_item2_score": (
            auditc_item2_score
        ),

        "auditc_item3_response": (
            auditc_item3_response
        ),

        "auditc_item3_score": (
            auditc_item3_score
        ),

        "auditc_total_score": (
            auditc_total_score
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

    "iterations_per_condition": (
        num_iterations
    ),

    "total_planned_requests": (
        total_planned_requests
    ),

    "conditions": conditions,

    "condition_prompts": (
        condition_prompt_metadata
    ),

    "auditc": {
        "item1_response": auditc_item1_response,
        "item1_score": auditc_item1_score,

        "item2_response": auditc_item2_response,
        "item2_score": auditc_item2_score,

        "item3_response": auditc_item3_response,
        "item3_score": auditc_item3_score,

        "total_score": auditc_total_score,
    },

    "benchmark_text": benchmark_text,

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

        condition_prompt = (
            condition_prompt_lookup[
                condition_id
            ]["prompt"]
        )

        condition_prompt_sha256 = (
            condition_prompt_lookup[
                condition_id
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

results_df[
    "benchmark_text"
] = benchmark_text


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

    (
        word_edit_distance,
        normalized_word_distance,
        word_structural_similarity,
    ) = calculate_word_distance_metrics(
        reference_text=benchmark_text,
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

        benchmark_embedding = semantic_model.encode(
            [benchmark_text],

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

        semantic_cosine_similarities = (
            util.cos_sim(
                benchmark_embedding,
                response_embeddings,
            )[0]
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
    *COMORBIDITY_FIELD_NAMES,

    "auditc_item1_response",
    "auditc_item1_score",
    "auditc_item2_response",
    "auditc_item2_score",
    "auditc_item3_response",
    "auditc_item3_score",
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
    "benchmark_text": (
        benchmark_text
    ),

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
    f"Iterations per condition: "
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
            *COMORBIDITY_FIELD_NAMES[:5],
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
        
        
        
        
        
        