from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate dataset integrity and route rows into pass/hold/reject groups."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input dataset path (.xls/.xlsx/.csv).",
    )
    parser.add_argument(
        "--rules",
        default="docs/validation_rules.yaml",
        help="Validation rule file path (yaml).",
    )
    parser.add_argument(
        "--out-dir",
        default="outputs/validation",
        help="Output directory for validation reports.",
    )
    return parser.parse_args()


def load_rules(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_dataset(path: str | Path, excel_header_row: int) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input file not found: {p}")

    ext = p.suffix.lower()
    if ext in {".xls", ".xlsx"}:
        return pd.read_excel(p, header=excel_header_row)
    if ext == ".csv":
        return pd.read_csv(p)
    raise ValueError("Unsupported file extension. Use .xls, .xlsx, or .csv")


def to_numeric_safe(df: pd.DataFrame, cols: list[str]) -> None:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")


def reason_join(values: list[str]) -> str:
    return ";".join(values) if values else ""


def validate_rows(df: pd.DataFrame, rules: dict[str, Any]) -> pd.DataFrame:
    score_cfg = rules["scoring"]
    comp_cfg = rules["composition"]
    phys_cfg = rules["physical_bounds"]
    miss_cfg = rules["missing"]
    targets = rules["targets"]["names"]

    wt_cols = [c for c in comp_cfg["wt_columns"] if c in df.columns]
    min_zero_cols = [c for c in phys_cfg["min_zero_columns"] if c in df.columns]
    binary_cols = [c for c in phys_cfg["binary_columns"] if c in df.columns]
    critical_cols = [c for c in miss_cfg["critical_feature_columns"] if c in df.columns]

    to_numeric_safe(df, wt_cols + min_zero_cols + binary_cols + targets + critical_cols)

    hard_reason_codes: list[list[str]] = []
    soft_reason_codes: list[list[str]] = []
    hard_reason_detail: list[list[str]] = []
    soft_reason_detail: list[list[str]] = []
    scores: list[int] = []
    statuses: list[str] = []

    cr_min = comp_cfg["austenitic_family_bounds"]["cr_min"]
    cr_max = comp_cfg["austenitic_family_bounds"]["cr_max"]
    ni_min = comp_cfg["austenitic_family_bounds"]["ni_min"]
    ni_max = comp_cfg["austenitic_family_bounds"]["ni_max"]
    wt_upper = comp_cfg["wt_sum_upper"] + comp_cfg["wt_sum_tolerance"]
    max_feat_missing_ratio = miss_cfg["max_missing_ratio_features"]

    for _, row in df.iterrows():
        hard_codes: list[str] = []
        soft_codes: list[str] = []
        hard_details: list[str] = []
        soft_details: list[str] = []

        # HARD: all targets missing (not usable for supervised training labels)
        if all(pd.isna(row.get(t)) for t in targets):
            hard_codes.append("H_TARGET_ALL_MISSING")
            hard_details.append("All target columns are missing.")

        # HARD: physical impossible negatives
        for c in min_zero_cols:
            v = row.get(c)
            if pd.notna(v) and v < 0:
                hard_codes.append("H_NEGATIVE_VALUE")
                hard_details.append(f"{c}={v} is negative.")
                break

        # HARD: binary columns must be 0/1 when present
        for c in binary_cols:
            v = row.get(c)
            if pd.notna(v) and v not in (0, 1):
                hard_codes.append("H_BINARY_INVALID")
                hard_details.append(f"{c}={v} is not in {{0,1}}.")
                break

        # HARD: composition sum unrealistically above 100
        wt_sum = row[wt_cols].sum(skipna=True) if wt_cols else 0.0
        if wt_cols and pd.notna(wt_sum) and wt_sum > wt_upper:
            hard_codes.append("H_WT_SUM_GT_100")
            hard_details.append(f"Sum(wt%)={wt_sum:.3f} > {wt_upper:.3f}.")

        # SOFT: Cr/Ni family bounds for austenitic stainless family
        cr = row.get("Cr")
        ni = row.get("Ni")
        if pd.notna(cr) and (cr < cr_min or cr > cr_max):
            soft_codes.append("S_CR_OUT_OF_AUSTENITIC_RANGE")
            soft_details.append(f"Cr={cr:.3f} outside [{cr_min}, {cr_max}].")
        if pd.notna(ni) and (ni < ni_min or ni > ni_max):
            soft_codes.append("S_NI_OUT_OF_AUSTENITIC_RANGE")
            soft_details.append(f"Ni={ni:.3f} outside [{ni_min}, {ni_max}].")

        # SOFT: core feature missing ratio
        if critical_cols:
            missing_ratio = float(row[critical_cols].isna().sum()) / float(len(critical_cols))
            if missing_ratio > max_feat_missing_ratio:
                soft_codes.append("S_CORE_FEATURE_MISSING_HIGH")
                soft_details.append(
                    f"Critical feature missing ratio={missing_ratio:.2f} > {max_feat_missing_ratio:.2f}."
                )

        # SOFT: Fe balance negative (based on listed elements only)
        fe_balance = 100.0 - wt_sum if wt_cols else None
        if fe_balance is not None and fe_balance < -0.5:
            soft_codes.append("S_FE_BALANCE_NEGATIVE")
            soft_details.append(f"Fe_balance={fe_balance:.3f} < -0.5.")

        score = int(score_cfg["initial_score"])
        if hard_codes:
            score -= int(score_cfg["hard_fail_penalty"])
        score -= int(score_cfg["soft_fail_penalty"]) * len(soft_codes)
        score = max(score, 0)

        if hard_codes or score < int(score_cfg["hold_min_score"]):
            status = "reject"
        elif score < int(score_cfg["pass_min_score"]) or soft_codes:
            status = "hold"
        else:
            status = "pass"

        hard_reason_codes.append(hard_codes)
        soft_reason_codes.append(soft_codes)
        hard_reason_detail.append(hard_details)
        soft_reason_detail.append(soft_details)
        scores.append(score)
        statuses.append(status)

    out = df.copy()
    out["quality_score"] = scores
    out["quality_status"] = statuses
    out["hard_reason_codes"] = [reason_join(v) for v in hard_reason_codes]
    out["soft_reason_codes"] = [reason_join(v) for v in soft_reason_codes]
    out["hard_reason_detail"] = [reason_join(v) for v in hard_reason_detail]
    out["soft_reason_detail"] = [reason_join(v) for v in soft_reason_detail]
    return out


def validate_schema(df: pd.DataFrame, rules: dict[str, Any]) -> dict[str, Any]:
    required = rules["schema"]["required_columns"]
    missing_required = [c for c in required if c not in df.columns]
    return {
        "required_count": len(required),
        "missing_required_count": len(missing_required),
        "missing_required_columns": missing_required,
        "schema_ok": len(missing_required) == 0,
    }


def write_outputs(df_valid: pd.DataFrame, schema_report: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    report_csv = out_dir / "validation_report.csv"
    pass_csv = out_dir / "train_set.csv"
    hold_csv = out_dir / "holdout_review.csv"
    reject_csv = out_dir / "reject_set.csv"
    summary_json = out_dir / "validation_summary.json"

    df_valid.to_csv(report_csv, index=False)
    df_valid[df_valid["quality_status"] == "pass"].to_csv(pass_csv, index=False)
    df_valid[df_valid["quality_status"] == "hold"].to_csv(hold_csv, index=False)
    df_valid[df_valid["quality_status"] == "reject"].to_csv(reject_csv, index=False)

    summary = {
        "rows_total": int(len(df_valid)),
        "rows_pass": int((df_valid["quality_status"] == "pass").sum()),
        "rows_hold": int((df_valid["quality_status"] == "hold").sum()),
        "rows_reject": int((df_valid["quality_status"] == "reject").sum()),
        "schema": schema_report,
        "output_files": {
            "validation_report": str(report_csv),
            "train_set": str(pass_csv),
            "holdout_review": str(hold_csv),
            "reject_set": str(reject_csv),
        },
    }
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    args = parse_args()
    rules = load_rules(args.rules)
    excel_header_row = int(rules["schema"].get("excel_header_row", 0))
    df = load_dataset(args.input, excel_header_row=excel_header_row)

    schema_report = validate_schema(df, rules)
    df_valid = validate_rows(df, rules)
    write_outputs(df_valid, schema_report, out_dir=Path(args.out_dir))


if __name__ == "__main__":
    main()
