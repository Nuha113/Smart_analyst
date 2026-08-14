"""Inference-only transaction classifier runtime.

The module loads a self-contained trained bundle relative to this file. It does
not read training data and does not expose a fit method.
"""

from __future__ import annotations

import ast
import re
import unicodedata
from collections.abc import Collection, Mapping
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_FILENAME = "transaction_classifier_bundle.joblib"
DEFAULT_MODEL_PATH = PACKAGE_DIR / "models" / DEFAULT_MODEL_FILENAME

DIACRITICS_RE = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")
INVISIBLE_RE = re.compile(r"[\u200B\u200C\u200D\u200E\u200F\u2060\uFEFF]")
EMPTY_TEXT_VALUES = {"", "nan", "none", "null", "<na>", "nat"}


def safe_text(value) -> str:
    """Convert a scalar or collection to clean text without leaking NaN tokens."""
    if isinstance(value, Mapping):
        parts = [safe_text(item) for item in value.values()]
        return " ".join(item for item in parts if item)
    if isinstance(value, Collection) and not isinstance(value, (str, bytes, bytearray)):
        parts = [safe_text(item) for item in value]
        return " ".join(item for item in parts if item)
    if value is None:
        return ""
    try:
        if bool(pd.isna(value)):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return "" if text.casefold() in EMPTY_TEXT_VALUES else text


def clean_text(value) -> str:
    text = safe_text(value)
    text = unicodedata.normalize("NFKC", text)
    text = DIACRITICS_RE.sub("", text).replace("ـ", "")
    text = INVISIBLE_RE.sub("", text)
    text = re.sub(r"[أإآٱ]", "ا", text).replace("ى", "ي")
    return re.sub(r"\s+", " ", text).strip()


def clean_keywords(value) -> str:
    """Normalize scalar, collection, or list-like keyword values safely."""
    if isinstance(value, Mapping):
        value = list(value.values())
    if isinstance(value, Collection) and not isinstance(value, (str, bytes, bytearray)):
        parts = [clean_text(item) for item in value]
        return " ".join(item for item in parts if item)
    text = safe_text(value)
    if not text:
        return ""
    stripped = text.strip()
    if len(stripped) >= 2 and stripped[0] in "[({" and stripped[-1] in "])}":
        try:
            parsed = ast.literal_eval(stripped)
            if isinstance(parsed, Collection) and not isinstance(parsed, (str, bytes, bytearray)):
                return clean_keywords(parsed)
        except (ValueError, SyntaxError):
            pass
    text = re.sub(r"[,،;؛|\r\n]+", " ", text)
    return clean_text(text)


def prepare_input_text(problem, keywords) -> str:
    cleaned_problem = clean_text(safe_text(problem))
    cleaned_keywords = clean_keywords(keywords)
    return clean_text(f"{cleaned_problem} {cleaned_keywords}")


class TransactionClassifier:
    """Reusable inference and employee-review interface."""

    def __init__(self, model_path=None):
        self.model_path = Path(model_path).expanduser().resolve() if model_path else DEFAULT_MODEL_PATH
        self.main_model = None
        self.subcategory_models = {}
        self.label_metadata_by_id = {}
        self.labels_by_main_category = {}
        self.official_main_categories = []
        self.confidence_threshold = 0.80
        self.MAIN_CATEGORY_MAPPING = {}
        self.model_version = ""
        self.model_configuration = {}
        self.bundle_versions = {}
        self._load_default_model()

    def _load_default_model(self):
        if not self.model_path.exists():
            raise FileNotFoundError(f"Transaction classifier model was not found at: {self.model_path}")
        self.load_model(self.model_path)

    def load_model(self, path=None):
        target = Path(path).expanduser().resolve() if path else self.model_path
        if not target.exists():
            raise FileNotFoundError(f"Transaction classifier model was not found at: {target}")
        bundle = joblib.load(target)
        required = {"main_model", "subcategory_models", "label_metadata_by_id", "labels_by_main_category",
                    "official_main_categories", "confidence_threshold", "MAIN_CATEGORY_MAPPING"}
        missing = required - set(bundle)
        if missing:
            raise ValueError(f"Model bundle is missing keys: {sorted(missing)}")
        self.model_path = target
        self.main_model = bundle["main_model"]
        self.subcategory_models = bundle["subcategory_models"]
        self.label_metadata_by_id = bundle["label_metadata_by_id"]
        self.labels_by_main_category = bundle["labels_by_main_category"]
        self.official_main_categories = bundle["official_main_categories"]
        self.confidence_threshold = float(bundle["confidence_threshold"])
        self.MAIN_CATEGORY_MAPPING = bundle["MAIN_CATEGORY_MAPPING"]
        self.model_version = bundle.get("model_version", "")
        self.model_configuration = bundle.get("model_configuration", {})
        self.bundle_versions = {k: bundle.get(k, "") for k in ("scikit_learn_version", "pandas_version", "numpy_version")}
        return self

    def _metadata(self, label_id):
        return dict(self.label_metadata_by_id.get(str(label_id), {}))

    def _standardize_main_category(self, value):
        normalized = clean_text(safe_text(value))
        return self.MAIN_CATEGORY_MAPPING.get(normalized, normalized)

    def standardize_main_category(self, value):
        """Return the official branch name used by the trained models."""
        return self._standardize_main_category(value)

    def _predict_main(self, input_text):
        probabilities = self.main_model.predict_proba([input_text])[0]
        index = int(np.argmax(probabilities))
        return str(self.main_model.classes_[index]), float(probabilities[index])

    def _top_three(self, input_text, used_main):
        model = self.subcategory_models[used_main]
        probabilities = model.predict_proba([input_text])[0]
        classes = np.asarray(model.classes_, dtype=object)
        order = np.argsort(-probabilities)[:3]
        rows = []
        for rank, index in enumerate(order, start=1):
            label = str(classes[index]); metadata = self._metadata(label)
            rows.append({"rank": rank, "label_id": label, "sub_category": metadata.get("sub_category", ""),
                         "action": metadata.get("action", ""), "confidence": float(probabilities[index])})
        return rows

    def _apply_assistance_rules(self, current_label, input_text, used_main):
        if used_main != "مساعدات":
            return current_label, False, ""
        rules = [
            ("AID_18", ["فاتورة مياه", "فواتير المياه", "شركة المياه"]),
            ("AID_19", ["فاتورة كهرباء", "فواتير الكهرباء", "شركة الكهرباء"]),
            ("AID_20", ["فاتورة اتصالات", "فواتير الاتصالات", "خدمة الاتصال", "شركة الاتصالات"]),
            # AID_04 harmful traffic branch remains disabled.
            ("AID_05", ["رسوم اقامة", "تجديد اقامة", "تجديد الاقامة", "رسوم تجديد الاقامة"]),
            ("AID_06", ["سداد غرامة", "سداد غرامات", "غرامة مالية", "الغرامات"]),
            ("AID_12", [" دية "]),
            ("AID_13", ["رسوم دراسية", "قسط دراسي", "اقساط دراسية", "قسط مدرسة", "مدرسة اهلية"]),
            ("AID_17", ["اعفاء قرض", "اعفاء من قرض", "اعادة جدولة", "جدولة القرض", "تخفيض اقساط", "تاجيل الاقساط"]),
            ("AID_08", ["بنك التنمية الاجتماعية", "صندوق التنمية", "الصندوق العقاري", "صندوق حكومي"]),
            ("AID_07", ["مديونية بنك", "مديونية البنوك", "قرض شخصي", "سداد قرض", "قرض بنكي", "دين للبنك"]),
            ("AID_10", ["ايقاف الخدمات", "رفع ايقاف الخدمات"]),
        ]
        padded = f" {input_text} "
        for target, phrases in rules:
            if any(clean_text(phrase) in padded for phrase in phrases):
                changed = target != current_label
                return target, changed, f"تم التعرف على قاعدة المساعدات {target}" if changed else ""
        return current_label, False, ""

    def _apply_nationality_rule(self, current_label, top_labels, nationality):
        pair = {"HEA_06", "HEA_11"}
        if current_label not in pair or not (set(top_labels) & pair):
            return current_label, False, ""
        nationality = clean_text(safe_text(nationality))
        target = current_label; reason = ""
        if nationality in {"سعودي", "سعودية"}:
            target, reason = "HEA_06", "الجنسية سعودية ضمن سياق أهلية العلاج"
        elif nationality in {"غير سعودي", "غير سعودية", "مقيم", "مقيمة", "وافد", "وافدة"}:
            target, reason = "HEA_11", "الجنسية غير سعودية ضمن سياق أهلية العلاج"
        changed = target != current_label
        return target, changed, reason if changed else ""

    def _apply_safety_gates(self, input_text, prediction_available, mapping_available, mismatch, confidence):
        reasons = []
        if not input_text: reasons.append("REVIEW_EMPTY_INPUT_TEXT")
        if not prediction_available or not mapping_available: reasons.append("REVIEW_PREDICTION_OR_MAPPING_UNAVAILABLE")
        if mismatch: reasons.append("REVIEW_MAIN_CATEGORY_MISMATCH")
        if confidence is None or float(confidence) < self.confidence_threshold: reasons.append("REVIEW_LOW_SUBCATEGORY_CONFIDENCE")
        return bool(reasons), " | ".join(reasons)

    def predict_one(self, row):
        data = row.to_dict() if isinstance(row, pd.Series) else dict(row)
        problem = safe_text(data.get("problem", "")); keywords = data.get("keywords", "")
        provided_raw = safe_text(data.get("main_category", "")); nationality = safe_text(data.get("nationality", "غير معروف")) or "غير معروف"
        input_text = prepare_input_text(problem, keywords)

        model_main, main_confidence = self._predict_main(input_text)
        provided_main = self._standardize_main_category(provided_raw)
        provided_valid = provided_main in self.official_main_categories
        used_main = provided_main if provided_valid else model_main
        mismatch = bool(provided_valid and provided_main != model_main)

        top3 = self._top_three(input_text, used_main)
        predicted_label = top3[0]["label_id"]
        confidence = top3[0]["confidence"]
        margin = confidence - top3[1]["confidence"] if len(top3) > 1 else 0.0

        assistance_label, assistance_applied, assistance_reason = self._apply_assistance_rules(predicted_label, input_text, used_main)
        nationality_label, nationality_applied, nationality_reason = self._apply_nationality_rule(
            assistance_label, [item["label_id"] for item in top3], nationality)
        proposed_label = nationality_label
        actual_rule_change = proposed_label != predicted_label
        rule_reasons = []
        if assistance_applied: rule_reasons.append(f"ASSISTANCE_RULE: {assistance_reason}")
        if nationality_applied: rule_reasons.append(f"NATIONALITY_RULE: {nationality_reason}")

        predicted_meta = self._metadata(predicted_label); proposed_meta = self._metadata(proposed_label)
        gate_triggered, gate_reason = self._apply_safety_gates(
            input_text, bool(predicted_label), bool(proposed_meta), mismatch, confidence)
        if gate_triggered:
            decision_source, review_status = "EMPLOYEE_REVIEW", "PENDING"
        elif actual_rule_change:
            decision_source, review_status = "RULE_CORRECTED_AUTO", "NOT_REQUIRED"
        else:
            decision_source, review_status = "MODEL_AUTO", "NOT_REQUIRED"

        # Every row remains classified. For pending review, these fields are the
        # system proposal shown to the employee; review_status tracks approval.
        final_label, final_meta = proposed_label, proposed_meta

        result = {
            "provided_main_category": provided_raw, "model_main_category": model_main, "model_main_confidence": main_confidence,
            "used_main_category": used_main, "main_category_mismatch": mismatch,
            "predicted_label_id": predicted_label, "predicted_sub_category": predicted_meta.get("sub_category", ""),
            "predicted_action": predicted_meta.get("action", ""),
            "subcategory_confidence": confidence, "margin": margin, "top_3_labels": [x["label_id"] for x in top3],
            "assistance_rule_applied": assistance_applied,
            "assistance_rule_original_label": predicted_label if assistance_applied else "",
            "assistance_rule_corrected_label": assistance_label if assistance_applied else "",
            "assistance_rule_reason": assistance_reason if assistance_applied else "",
            "nationality_rule_applied": nationality_applied,
            "nationality_rule_original_label": assistance_label if nationality_applied else "",
            "nationality_rule_corrected_label": nationality_label if nationality_applied else "",
            "nationality_rule_reason": nationality_reason if nationality_applied else "",
            "rule_applied": actual_rule_change,
            "rule_original_label": predicted_label if actual_rule_change else "",
            "rule_corrected_label": proposed_label if actual_rule_change else "",
            "rule_reason": " | ".join(rule_reasons) if actual_rule_change else "",
            "gate_triggered": gate_triggered, "gate_reason": gate_reason,
            "proposed_label_id": proposed_label, "proposed_sub_category": proposed_meta.get("sub_category", ""),
            "proposed_action": proposed_meta.get("action", ""),
            "review_required": gate_triggered,
            "employee_selected_label_id": "", "employee_selection_source": "", "employee_review_completed": False,
            "final_label_id": final_label, "final_sub_category": final_meta.get("sub_category", ""),
            "final_action": final_meta.get("action", ""), "decision_source": decision_source, "review_status": review_status,
        }
        for rank in range(1, 4):
            item = top3[rank - 1] if len(top3) >= rank else {"label_id": "", "sub_category": "", "confidence": np.nan}
            result[f"top_{rank}_label_id"] = item["label_id"]
            result[f"top_{rank}_sub_category"] = item["sub_category"]
            result[f"top_{rank}_action"] = item.get("action", "")
            result[f"top_{rank}_confidence"] = item["confidence"]
        return result

    def predict_dataframe(self, input_df):
        if not isinstance(input_df, pd.DataFrame): raise TypeError("input_df must be a pandas DataFrame.")
        if "problem" not in input_df.columns: raise ValueError("Missing required input column: problem")
        work = input_df.copy()
        for column, default in (("keywords", ""), ("main_category", ""), ("nationality", "غير معروف")):
            if column not in work.columns: work[column] = default
        id_source = next((c for c in ("classification_row_id", "transaction_id", "Transaction ID", "case_id") if c in work.columns), None)
        if id_source: work["classification_row_id"] = work[id_source].map(safe_text)
        else: work["classification_row_id"] = [f"ROW_{i:06d}" for i in range(1, len(work) + 1)]
        if work["classification_row_id"].eq("").any() or work["classification_row_id"].duplicated().any():
            raise ValueError("classification_row_id values must be non-empty and unique.")
        predictions = pd.DataFrame([self.predict_one(row) for _, row in work.iterrows()], index=work.index)
        overlap = [column for column in predictions.columns if column in work.columns]
        if overlap: work = work.drop(columns=overlap)
        return pd.concat([work, predictions], axis=1)

    def get_review_queue(self, classified_df):
        pending = classified_df["review_status"].eq("PENDING")
        if {"review_required", "employee_review_completed"}.issubset(classified_df.columns):
            pending |= classified_df["review_required"].fillna(False).astype(bool) & ~classified_df["employee_review_completed"].fillna(False).astype(bool)
        return classified_df.loc[pending].copy()

    def get_employee_review_options(self, classified_df, row_id):
        match = classified_df[classified_df["classification_row_id"].map(safe_text).eq(safe_text(row_id))]
        if len(match) != 1: raise ValueError("Row identifier not found or not unique.")
        row = match.iloc[0]; top_labels = [str(row[f"top_{i}_label_id"]) for i in range(1, 4)]
        top_options = []
        for i, label in enumerate(top_labels, start=1):
            metadata = self._metadata(label)
            top_options.append({"rank": i, "label_id": label, "sub_category": metadata.get("sub_category", ""),
                                "action": metadata.get("action", ""), "confidence": row[f"top_{i}_confidence"]})
        other_options = [dict(self._metadata(label), label_id=label)
                         for label in self.labels_by_main_category[row["used_main_category"]] if label not in top_labels]
        return {"top_3_options": top_options, "other_options": other_options}

    def approve_system_prediction(self, classified_df, row_id):
        """Complete a pending review by accepting the visible system proposal."""
        result = classified_df.copy()
        mask = result["classification_row_id"].map(safe_text).eq(safe_text(row_id))
        if mask.sum() != 1:
            raise ValueError("Row identifier not found or not unique.")
        index = result.index[mask][0]
        if result.loc[index, "review_status"] != "PENDING":
            raise ValueError("The selected row does not have a pending employee review.")
        result.loc[index, "employee_selected_label_id"] = result.loc[index, "proposed_label_id"]
        result.loc[index, "employee_selection_source"] = "SYSTEM_PREDICTION_APPROVED"
        result.loc[index, "employee_review_completed"] = True
        result.loc[index, "final_label_id"] = result.loc[index, "proposed_label_id"]
        result.loc[index, "final_sub_category"] = result.loc[index, "proposed_sub_category"]
        result.loc[index, "final_action"] = result.loc[index, "proposed_action"]
        result.loc[index, "decision_source"] = "EMPLOYEE_REVIEW"
        result.loc[index, "review_status"] = "COMPLETED"
        return result

    def apply_employee_selection(self, classified_df, row_id, employee_selected_label_id):
        result = classified_df.copy(); mask = result["classification_row_id"].map(safe_text).eq(safe_text(row_id))
        if mask.sum() != 1: raise ValueError("Row identifier not found or not unique.")
        index = result.index[mask][0]; row = result.loc[index]; selected = str(employee_selected_label_id)
        if row["review_status"] not in {"PENDING", "COMPLETED"} or row["decision_source"] != "EMPLOYEE_REVIEW":
            raise ValueError("The selected row is not an employee-review case.")
        metadata = self._metadata(selected)
        if not metadata: raise ValueError("Selected label is not in label metadata.")
        if metadata["main_category"] != row["used_main_category"]:
            raise ValueError("Selected label must belong to the same Main Category.")
        top_labels = [str(row[f"top_{i}_label_id"]) for i in range(1, 4)]
        result.loc[index, "employee_selected_label_id"] = selected
        result.loc[index, "employee_selection_source"] = "TOP_3" if selected in top_labels else "OTHER_BRANCH_LABEL"
        result.loc[index, "employee_review_completed"] = True
        result.loc[index, "final_label_id"] = selected
        result.loc[index, "final_sub_category"] = metadata["sub_category"]
        result.loc[index, "final_action"] = metadata["action"]
        result.loc[index, "decision_source"] = "EMPLOYEE_REVIEW"
        result.loc[index, "review_status"] = "COMPLETED"
        return result

    def apply_employee_decisions(self, classified_df, employee_decisions_df):
        if "classification_row_id" not in employee_decisions_df.columns:
            raise ValueError("Employee decisions must include classification_row_id.")
        has_selection = "employee_selected_label_id" in employee_decisions_df.columns
        has_approval = "approve_system_prediction" in employee_decisions_df.columns
        if not has_selection and not has_approval:
            raise ValueError("Employee decisions must include a selection or approval column.")
        result = classified_df.copy()
        for _, decision in employee_decisions_df.iterrows():
            approve = False
            if has_approval:
                value = safe_text(decision["approve_system_prediction"]).casefold()
                approve = value in {"true", "1", "yes", "y", "نعم"}
            if approve:
                result = self.approve_system_prediction(result, decision["classification_row_id"])
            elif has_selection and safe_text(decision["employee_selected_label_id"]):
                result = self.apply_employee_selection(result, decision["classification_row_id"], decision["employee_selected_label_id"])
            else:
                raise ValueError("Each employee decision must approve the proposal or select a label.")
        return result

    def export_results(self, classified_df, path, require_completed_reviews=True):
        pending = int(classified_df["review_status"].eq("PENDING").sum())
        if pending and require_completed_reviews:
            raise ValueError("There are pending employee reviews.")
        classified_df.to_excel(path, index=False)
        return {"path": str(path), "status": "FINAL" if pending == 0 else "INTERIM", "pending_reviews": pending}
