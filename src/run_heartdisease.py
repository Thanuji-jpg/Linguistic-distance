"""
Heart-disease real-patient run (WHO Zone IV only).

Loads every unique HAC 0/1 disease profile, keeps those with
heartdisease = Present (all other comorbidities unchanged),
applies the fixed Zone IV AUDIT scenario from auditcode_who_audit.py,
and calls the API once per matched patient.

Does not modify auditcode_who_audit.py.
"""

from __future__ import annotations

import hashlib
import sys
import time
import types
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util


SCRIPT_DIR = Path(__file__).resolve().parent
WHO_AUDIT_PATH = SCRIPT_DIR / "auditcode_who_audit.py"
OUTPUT_DIR = SCRIPT_DIR / "output"

COST_PER_PATIENT_USD = 0.002
SECONDS_PER_PATIENT = 1.6
DISEASE_GROUP = "heartdisease"
NUM_ITERATIONS = 1


def load_who_audit_helpers():
    """
    Load definitions from auditcode_who_audit.py without running its
    main pilot loop (which starts at conditions = build_conditions()).
    """
    if not WHO_AUDIT_PATH.exists():
        raise FileNotFoundError(
            f"Expected {WHO_AUDIT_PATH} next to this script."
        )

    source = WHO_AUDIT_PATH.read_text(encoding="utf-8")
    marker = "\nconditions = build_conditions()\n"

    if marker not in source:
        raise RuntimeError(
            "Could not find 'conditions = build_conditions()' "
            "cutoff in auditcode_who_audit.py."
        )

    partial_source = source.split(marker, 1)[0]
    module = types.ModuleType("auditcode_who_audit")
    module.__dict__["__file__"] = str(WHO_AUDIT_PATH)
    module.__dict__["__name__"] = "auditcode_who_audit"
    exec(
        compile(
            partial_source,
            str(WHO_AUDIT_PATH),
            "exec",
        ),
        module.__dict__,
    )
    return module


def build_zone_iv_scenario(who):
    """
    Reuse the fixed Zone IV entry from AUDIT_SCENARIOS exactly.
    """
    if "zone_iv_specialist_referral" not in who.AUDIT_SCENARIOS:
        raise KeyError(
            "AUDIT_SCENARIOS is missing "
            "'zone_iv_specialist_referral'."
        )

    scenario = who.build_audit_scenario(
        who.AUDIT_SCENARIOS["zone_iv_specialist_referral"]
    )

    if scenario["who_zone"] != "IV":
        raise ValueError(
            "Expected WHO Zone IV scenario, got "
            f"Zone {scenario['who_zone']}."
        )

    if int(scenario["auditc_total_score"]) != 25:
        raise ValueError(
            "Expected Zone IV total score 25, got "
            f"{scenario['auditc_total_score']}."
        )

    return scenario


def load_heartdisease_patients(who):
    """
    Load all unique HAC profiles, then keep heartdisease = Present.
    """
    who.hac_profile_mode = "unique_profiles"
    who.load_hac_comorbidity_settings()

    if "heartdisease" not in who.COMORBIDITY_FIELD_NAMES:
        raise ValueError(
            "HAC comorbidity fields do not include heartdisease."
        )

    matched = [
        condition.copy()
        for condition in who.hac_loaded_conditions
        if condition["heartdisease"]
        == who.COMORBIDITY_PRESENT_LABEL
    ]

    if not matched:
        raise ValueError(
            "No unique HAC profiles have heartdisease = Present."
        )

    return [
        {
            "condition_id": condition_id,
            **condition,
        }
        for condition_id, condition in enumerate(
            matched,
            start=1,
        )
    ]


def confirm_paid_api_calls(patient_count, model_name):
    """
    Print count / cost / time estimates, then require typed 'yes'.
    """
    estimated_cost = patient_count * COST_PER_PATIENT_USD
    estimated_seconds = patient_count * SECONDS_PER_PATIENT
    estimated_minutes = estimated_seconds / 60.0

    print("\n" + "=" * 80)
    print("API CALL CONFIRMATION REQUIRED")
    print("=" * 80)

    print(
        f"\nReal unique HAC profiles with heartdisease = Present: "
        f"{patient_count}"
    )
    print(
        "AUDIT scenario: fixed WHO Zone IV "
        "(score 25, Referral to Specialist) for every patient."
    )
    print(f"API calls planned: {patient_count} (1 per patient)")
    print(f"Model: {model_name}")
    print(
        f"Estimated cost: ~${estimated_cost:.2f} "
        f"(${COST_PER_PATIENT_USD:.3f} per patient)"
    )
    print(
        f"Estimated time: ~{estimated_minutes:.1f} minutes "
        f"({SECONDS_PER_PATIENT:.1f}s per patient)"
    )

    print(
        "\nType 'yes' to proceed, or anything else to cancel."
    )

    response = input(
        "\nProceed with paid API calls? "
    ).strip().lower()

    if response != "yes":
        print("\nCancelled. No API calls were made.")
        sys.exit(0)

    print("\nConfirmed. Starting API requests...\n")


def create_record(who, condition, audit_scenario, overall_request_number, prompt_sha256):
    """
    One output row matching the who_audit pipeline columns, plus disease_group.
    """
    return {
        "condition_id": condition["condition_id"],
        "condition_iteration": NUM_ITERATIONS,
        "disease_group": DISEASE_GROUP,
        "scenario_id": audit_scenario["scenario_id"],
        "who_zone": audit_scenario["who_zone"],
        "who_zone_label": audit_scenario["who_zone_label"],
        **{
            field_name: condition[field_name]
            for field_name in who.COMORBIDITY_FIELD_NAMES
        },
        "auditc_item1_response": audit_scenario["item1_response"],
        "auditc_item1_score": audit_scenario["item1_score"],
        "auditc_item2_response": audit_scenario["item2_response"],
        "auditc_item2_score": audit_scenario["item2_score"],
        "auditc_item3_response": audit_scenario["item3_response"],
        "auditc_item3_score": audit_scenario["item3_score"],
        "auditc_item4_response": audit_scenario["item4_response"],
        "auditc_item4_score": audit_scenario["item4_score"],
        "auditc_item5_response": audit_scenario["item5_response"],
        "auditc_item5_score": audit_scenario["item5_score"],
        "auditc_item6_response": audit_scenario["item6_response"],
        "auditc_item6_score": audit_scenario["item6_score"],
        "auditc_item7_response": audit_scenario["item7_response"],
        "auditc_item7_score": audit_scenario["item7_score"],
        "auditc_item8_response": audit_scenario["item8_response"],
        "auditc_item8_score": audit_scenario["item8_score"],
        "auditc_item9_response": audit_scenario["item9_response"],
        "auditc_item9_score": audit_scenario["item9_score"],
        "auditc_item10_response": audit_scenario["item10_response"],
        "auditc_item10_score": audit_scenario["item10_score"],
        "auditc_total_score": audit_scenario["auditc_total_score"],
        "benchmark_text": who.build_comorbidity_reference_text(
            condition,
            audit_scenario["auditc_total_score"],
        ),
        "model_response": None,
        "analysis_eligible": False,
        "overall_request_number": overall_request_number,
        "condition_mode": "hac_dataset_heartdisease_present",
        "response_word_count": None,
        "response_character_count": None,
        "response_sentence_count": None,
        "requested_model": who.model_name,
        "returned_model": None,
        "prompt_version": who.prompt_version,
        "condition_prompt_sha256": prompt_sha256,
        "requested_temperature": who.temperature,
        "returned_temperature": None,
        "requested_top_p": who.top_p,
        "returned_top_p": None,
        "requested_reasoning_effort": who.reasoning_effort,
        "returned_reasoning_effort": None,
        "requested_max_output_tokens": who.max_output_tokens,
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
        "request_used_previous_response_id": False,
        "request_used_conversation": False,
        "returned_previous_response_id": None,
        "returned_conversation_id": None,
        "response_id": None,
        "request_id": None,
        "incomplete_details_json": None,
        "response_error_json": None,
        "exception_type": None,
        "exception_message": None,
        "exception_status_code": None,
        "word_edit_distance": None,
        "normalized_word_distance": None,
        "word_structural_similarity": None,
        "semantic_cosine_similarity": None,
        "semantic_cosine_distance": None,
    }


def call_model(who, prompt):
    request_arguments = {
        "model": who.model_name,
        "input": prompt,
    }

    if who.temperature is not None:
        request_arguments["temperature"] = who.temperature

    if who.top_p is not None:
        request_arguments["top_p"] = who.top_p

    if who.max_output_tokens is not None:
        request_arguments["max_output_tokens"] = (
            who.max_output_tokens
        )

    if who.reasoning_effort is not None:
        request_arguments["reasoning"] = {
            "effort": who.reasoning_effort
        }

    return who.client.responses.create(**request_arguments)


def fill_success_record(who, record, response, elapsed_seconds, end_utc):
    response_text = who.extract_response_text(response) or ""
    response_status = who.get_field(response, "status")
    usage = who.get_field(response, "usage")
    input_token_details = who.get_field(
        usage,
        "input_tokens_details",
    )
    output_token_details = who.get_field(
        usage,
        "output_tokens_details",
    )
    returned_reasoning = who.get_field(response, "reasoning")
    returned_conversation = who.get_field(
        response,
        "conversation",
    )

    if isinstance(returned_conversation, str):
        returned_conversation_id = returned_conversation
    else:
        returned_conversation_id = who.get_field(
            returned_conversation,
            "id",
        )

    record.update({
        "model_response": response_text,
        "analysis_eligible": (
            response_status == "completed"
            and bool(response_text.strip())
        ),
        "response_word_count": who.count_words(response_text),
        "response_character_count": len(response_text),
        "response_sentence_count": who.count_sentences(
            response_text
        ),
        "returned_model": who.get_field(response, "model"),
        "returned_temperature": who.get_field(
            response,
            "temperature",
        ),
        "returned_top_p": who.get_field(response, "top_p"),
        "returned_reasoning_effort": who.get_field(
            returned_reasoning,
            "effort",
        ),
        "returned_max_output_tokens": who.get_field(
            response,
            "max_output_tokens",
        ),
        "returned_service_tier": who.get_field(
            response,
            "service_tier",
        ),
        "input_tokens": who.get_field(usage, "input_tokens"),
        "cached_input_tokens": who.get_field(
            input_token_details,
            "cached_tokens",
        ),
        "output_tokens": who.get_field(usage, "output_tokens"),
        "reasoning_tokens": who.get_field(
            output_token_details,
            "reasoning_tokens",
        ),
        "total_tokens": who.get_field(usage, "total_tokens"),
        "request_end_utc": end_utc,
        "elapsed_seconds": elapsed_seconds,
        "api_created_at_utc": who.timestamp_to_utc(
            who.get_field(response, "created_at")
        ),
        "api_completed_at_utc": who.timestamp_to_utc(
            who.get_field(response, "completed_at")
        ),
        "request_completed_without_exception": True,
        "response_status": response_status,
        "returned_previous_response_id": who.get_field(
            response,
            "previous_response_id",
        ),
        "returned_conversation_id": returned_conversation_id,
        "response_id": who.get_field(response, "id"),
        "request_id": who.get_field(response, "_request_id"),
        "incomplete_details_json": who.object_to_json(
            who.get_field(response, "incomplete_details")
        ),
        "response_error_json": who.object_to_json(
            who.get_field(response, "error")
        ),
    })


def fill_error_record(who, record, exception, elapsed_seconds, end_utc):
    record.update({
        "request_end_utc": end_utc,
        "elapsed_seconds": elapsed_seconds,
        "request_completed_without_exception": False,
        "request_id": who.get_field(exception, "request_id"),
        "exception_type": type(exception).__name__,
        "exception_message": str(exception),
        "exception_status_code": who.get_field(
            exception,
            "status_code",
        ),
    })


def add_linguistic_metrics(who, records):
    """
    Compare each model_response to its comorbidity-aware benchmark.
    """
    eligible_indices = [
        index
        for index, record in enumerate(records)
        if record.get("analysis_eligible")
        and isinstance(record.get("model_response"), str)
        and record["model_response"].strip()
    ]

    for index in eligible_indices:
        record = records[index]
        (
            word_edit_distance,
            normalized_word_distance,
            word_structural_similarity,
        ) = who.calculate_word_distance_metrics(
            reference_text=record["benchmark_text"],
            comparison_text=record["model_response"],
        )
        record["word_edit_distance"] = word_edit_distance
        record["normalized_word_distance"] = (
            normalized_word_distance
        )
        record["word_structural_similarity"] = (
            word_structural_similarity
        )

    if not eligible_indices:
        return

    print("\nLoading SentenceTransformer for semantic metrics...")
    semantic_model = SentenceTransformer(who.semantic_model_name)

    response_texts = [
        records[index]["model_response"]
        for index in eligible_indices
    ]
    benchmark_texts = [
        records[index]["benchmark_text"]
        for index in eligible_indices
    ]

    benchmark_embeddings = semantic_model.encode(
        benchmark_texts,
        batch_size=who.semantic_batch_size,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    response_embeddings = semantic_model.encode(
        response_texts,
        batch_size=who.semantic_batch_size,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    similarities = (
        (benchmark_embeddings * response_embeddings)
        .sum(dim=1)
        .detach()
        .cpu()
        .numpy()
    )
    similarities = np.clip(similarities, -1.0, 1.0)

    for offset, index in enumerate(eligible_indices):
        similarity = float(similarities[offset])
        records[index]["semantic_cosine_similarity"] = similarity
        records[index]["semantic_cosine_distance"] = (
            1.0 - similarity
        )


def main():
    who = load_who_audit_helpers()
    audit_scenario = build_zone_iv_scenario(who)
    patients = load_heartdisease_patients(who)

    print("=" * 80)
    print("HEART DISEASE REAL-PATIENT RUN (ZONE IV FIXED)")
    print("=" * 80)
    print(
        f"Unique HAC profiles loaded: "
        f"{len(who.hac_loaded_conditions)}"
    )
    print(
        f"Profiles with heartdisease = Present: "
        f"{len(patients)}"
    )
    print(
        f"AUDIT: {audit_scenario['scenario_id']} | "
        f"score={audit_scenario['auditc_total_score']} | "
        f"Zone {audit_scenario['who_zone']} "
        f"({audit_scenario['who_zone_label']})"
    )

    confirm_paid_api_calls(
        patient_count=len(patients),
        model_name=who.model_name,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    run_timestamp = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )
    output_csv_path = (
        OUTPUT_DIR
        / f"heartdisease_real_patients_{run_timestamp}_responses.csv"
    )

    records = []
    total = len(patients)

    for overall_request_number, condition in enumerate(
        patients,
        start=1,
    ):
        prompt = who.build_prompt(condition, audit_scenario)
        prompt_sha256 = hashlib.sha256(
            prompt.encode("utf-8")
        ).hexdigest()

        record = create_record(
            who=who,
            condition=condition,
            audit_scenario=audit_scenario,
            overall_request_number=overall_request_number,
            prompt_sha256=prompt_sha256,
        )

        start_datetime = datetime.now(timezone.utc)
        start_counter = time.perf_counter()
        record["request_start_utc"] = start_datetime.isoformat()

        try:
            response = call_model(who, prompt)
            end_datetime = datetime.now(timezone.utc)
            elapsed_seconds = time.perf_counter() - start_counter
            fill_success_record(
                who=who,
                record=record,
                response=response,
                elapsed_seconds=elapsed_seconds,
                end_utc=end_datetime.isoformat(),
            )
            print(
                f"Request {overall_request_number} of {total} "
                f"completed | condition {condition['condition_id']}"
            )
            print(f"Response: {record['model_response']}\n")

        except Exception as exception:
            end_datetime = datetime.now(timezone.utc)
            elapsed_seconds = time.perf_counter() - start_counter
            fill_error_record(
                who=who,
                record=record,
                exception=exception,
                elapsed_seconds=elapsed_seconds,
                end_utc=end_datetime.isoformat(),
            )
            print(
                f"Request {overall_request_number} of {total} "
                f"failed | condition {condition['condition_id']}"
            )
            print(
                f"{type(exception).__name__}: {exception}\n"
            )

        records.append(record)

        if (
            who.seconds_between_requests > 0
            and overall_request_number < total
        ):
            time.sleep(who.seconds_between_requests)

    add_linguistic_metrics(who, records)

    results_df = pd.DataFrame(records)

    # Put disease_group near the front, matching pipeline-style ordering.
    preferred_front = [
        "condition_id",
        "condition_iteration",
        "disease_group",
        "scenario_id",
        "who_zone",
        "who_zone_label",
        *list(who.COMORBIDITY_FIELD_NAMES),
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
        "word_edit_distance",
        "normalized_word_distance",
        "word_structural_similarity",
        "semantic_cosine_similarity",
        "semantic_cosine_distance",
        "analysis_eligible",
    ]
    ordered_columns = [
        column
        for column in preferred_front
        if column in results_df.columns
    ] + [
        column
        for column in results_df.columns
        if column not in preferred_front
    ]
    results_df = results_df[ordered_columns]

    results_df.to_csv(
        output_csv_path,
        index=False,
        encoding="utf-8-sig",
    )

    eligible = int(
        results_df["analysis_eligible"].fillna(False).sum()
    )
    completed = int(
        results_df["request_completed_without_exception"]
        .fillna(False)
        .sum()
    )

    print("=" * 80)
    print("HEART DISEASE RUN COMPLETE")
    print("=" * 80)
    print(f"Patients: {len(patients)}")
    print(f"Completed without exception: {completed}")
    print(f"Eligible for analysis: {eligible}")
    print(f"Saved: {output_csv_path}")


if __name__ == "__main__":
    main()
