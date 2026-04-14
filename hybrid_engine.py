import json
import math
import re
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from paddleocr import PaddleOCR
from pdf2image import convert_from_path
from transformers import AutoModelForImageTextToText, AutoProcessor


# ============================================================
# LOAD IMAGES
# ============================================================
def load_images(file_path: str) -> List[Image.Image]:
    """
    Accept either a PDF or an image and always return a list of PIL images.

    Why:
    - PDF -> many pages -> many images
    - Image -> one image -> still wrapped inside a list
    This keeps the downstream pipeline consistent.
    """
    if file_path.lower().endswith(".pdf"):
        return convert_from_path(file_path)
    return [Image.open(file_path).convert("RGB")]


# ============================================================
# TEXT NORMALIZATION
# ============================================================
def normalize_text(text: Any) -> str:
    """
    Normalize text for general matching.

    Steps:
    - convert to string
    - lowercase
    - strip spaces
    - collapse multiple spaces into one
    """
    text = "" if text is None else str(text)
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_code_like_text(text: Any) -> str:
    """
    Stronger normalization for short codes / ids / compact fields.

    Useful for:
    - diagnosis codes
    - tax id
    - invoice number
    - CPT codes

    Keeps only lowercase letters and numbers.
    """
    text = normalize_text(text)
    text = re.sub(r"[^a-z0-9]", "", text)
    return text


# ============================================================
# BBOX HELPERS
# ============================================================
def bbox_center(bbox: List[List[float]]) -> Tuple[float, float]:
    """
    Compute center point of a quadrilateral bounding box.
    """
    xs = [pt[0] for pt in bbox]
    ys = [pt[1] for pt in bbox]
    return sum(xs) / len(xs), sum(ys) / len(ys)


def euclidean_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """
    Standard Euclidean distance between two 2D points.
    """
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


# ============================================================
# FIELD LABEL ALIASES
# ============================================================
FIELD_LABEL_ALIASES = {
    "claimant_name": ["claimant name", "name"],
    "claimant_number": ["claimant number", "claim no", "claim #"],
    "tax_id": ["tax id", "taxid", "federal tax id", "tin", "tax identification"],
    "practice_address": ["practice address", "provider address", "service address"],
    "billing_address": ["billing address", "remit address", "remittance address", "bill to address"],
    "diagnosis_codes": ["diagnosis", "diagnosis code", "dx", "dx code", "icd", "icd code"],
    "date_of_service": ["date of service", "dos", "service date"],
    "cpt_codes": ["cpt", "procedure code", "cpt code", "hcpcs"],
    "charges": ["charges", "charge", "amount", "amt"],
    "units": ["units", "unit"],
    "invoice_date": ["invoice date", "bill date", "statement date"],
    "invoice_number": ["invoice number", "invoice #", "inv #", "invoice no"],
    "taxonomy": ["taxonomy", "taxonomy code"],
    "total_amount": ["total", "total amount", "balance due", "amount due"],
}


# ============================================================
# PADDLE OCR ENGINE
# ============================================================
class PaddleOCREngine:
    """
    Wrapper over PaddleOCR.

    Returns page-wise OCR lines in a normalized format:
    {
        "page_1": [
            {
                "text": "...",
                "confidence": 0.98,
                "bbox": [[x1,y1],[x2,y2],[x3,y3],[x4,y4]],
                "center": (cx, cy)
            },
            ...
        ]
    }
    """

    def __init__(self):
        self.ocr = PaddleOCR(
            use_angle_cls=True,
            lang="en",
            use_gpu=torch.cuda.is_available()
        )

    def extract_from_images(self, images: List[Image.Image]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Run OCR page by page and return structured OCR lines.
        Also extracts word-level bboxes for finer granularity.
        """
        import numpy as np

        page_results: Dict[str, List[Dict[str, Any]]] = {}

        for page_idx, image in enumerate(images, start=1):
            image_np = np.array(image)
            result = self.ocr.ocr(image_np, cls=True)

            structured_lines: List[Dict[str, Any]] = []

            if result and result[0]:
                for line in result[0]:
                    bbox = line[0]
                    text = line[1][0]
                    conf = float(line[1][1])

                    words: List[Dict[str, Any]] = []
                    if len(line) > 2 and line[2]:
                        for word in line[2]:
                            words.append({
                                "text": word[0],
                                "confidence": float(word[1]),
                                "bbox": word[0] if isinstance(word[0], list) else word[0],
                            })

                    structured_lines.append({
                        "text": text,
                        "confidence": round(conf, 4),
                        "bbox": bbox,
                        "center": bbox_center(bbox),
                        "words": words,
                    })

            page_results[f"page_{page_idx}"] = structured_lines

        return page_results


# ============================================================
# QWEN EXTRACTOR
# ============================================================
class QwenExtractor:
    """
    Runs Qwen extraction and computes two raw model signals:

    1. llm_confidence
       - derived from token probabilities
       - based on the best matching token window for a field value

    2. logprob
       - average logprob of the same best matching token window

    These are kept separate from OCR confidence.
    """

    def __init__(
        self,
        model_path: str = "/home/rohit.sahu/Qwen_model/qwen_models/Qwen2.5-VL-3B-Instruct",
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading Qwen model on {self.device}...")

        self.processor = AutoProcessor.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True,
        )

        model_kwargs = {
            "local_files_only": True,
            "trust_remote_code": True,
        }

        if torch.cuda.is_available():
            model_kwargs["torch_dtype"] = torch.float16
            model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["torch_dtype"] = torch.float32

        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            **model_kwargs,
        )

        print("✅ Qwen model loaded successfully")

        self.prompt = """
You are a highly accurate OCR data extraction system.
RULES:
1. Do not mistake ZIP codes for a Tax ID.
2. The claimant number is slightly different from the address.
3. charges, units, cpt_codes, and diagnosis_codes must be FLAT LISTS OF PLAIN STRINGS.
   - charges  : list of dollar amounts only, e.g. ["650.00", "120.00"]
   - units    : list of unit counts only,   e.g. ["1", "2"]
   - cpt_codes: list of procedure codes,    e.g. ["99213", "E1399"]
   - Do NOT return dicts or nested objects inside these lists.
4. Return ONLY valid JSON matching this structure:

{
  "claimant_name": "",
  "claimant_number": "",
  "tax_id": "",
  "practice_address": "",
  "billing_address": "",
  "diagnosis_codes": [],
  "date_of_service": "",
  "cpt_codes": [],
  "charges": [],
  "units": [],
  "invoice_date": "",
  "invoice_number": "",
  "taxonomy": "",
  "total_amount": ""
}
        """.strip()

    def extract_with_logprobs(self, image: Image.Image) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Run Qwen on one image and capture token-level probabilities/logprobs.

        Returns:
        - result_text: generated JSON text
        - token_data: list of token records like
            {
                "token": "...",
                "prob": 0.95,
                "logprob": -0.051
            }
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": self.prompt},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
        )

        inputs = {
            k: v.to(self.device) if hasattr(v, "to") else v
            for k, v in inputs.items()
        }

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=800,
                temperature=0.0,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
            )

        generated_ids = output.sequences[0]
        input_len = inputs["input_ids"].shape[-1]
        generated_only_ids = generated_ids[input_len:]

        result_text = self.processor.decode(
            generated_only_ids,
            skip_special_tokens=True,
        )

        scores = output.scores
        decoded_tokens = [self.processor.decode([token_id]) for token_id in generated_only_ids]

        token_data: List[Dict[str, Any]] = []

        for i, score in enumerate(scores):
            log_probs = F.log_softmax(score, dim=-1)
            token_id = generated_only_ids[i]
            logprob = log_probs[0, token_id].item()

            token_data.append({
                "token": decoded_tokens[i],
                "prob": math.exp(logprob),
                "logprob": logprob,
            })

        return result_text, token_data

    def compute_field_metrics(self, value: Any, token_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Compute raw Qwen metrics for a field value.

        Returns:
        {
            "llm_confidence": ...,
            "logprob": ...
        }

        For list fields:
        - score each item separately
        - average across items
        """
        if value is None:
            return {"llm_confidence": 0.0, "logprob": 0.0}

        if isinstance(value, list):
            if not value:
                return {"llm_confidence": 0.0, "logprob": 0.0}

            item_metrics = [
                self.compute_field_metrics(str(v), token_data)
                for v in value if str(v).strip()
            ]

            if not item_metrics:
                return {"llm_confidence": 0.0, "logprob": 0.0}

            avg_conf = sum(m["llm_confidence"] for m in item_metrics) / len(item_metrics)
            avg_logprob = sum(m["logprob"] for m in item_metrics) / len(item_metrics)

            return {
                "llm_confidence": round(avg_conf, 4),
                "logprob": round(avg_logprob, 4),
            }

        value_str = str(value).strip()
        if not value_str:
            return {"llm_confidence": 0.0, "logprob": 0.0}

        n = len(token_data)
        best_confidence = 0.0
        best_logprob = 0.0

        for start in range(n):
            reconstructed = ""

            for end in range(start, min(start + 40, n)):
                reconstructed += token_data[end]["token"]
                cleaned = reconstructed.strip()

                if value_str in cleaned or cleaned in value_str:
                    window_items = [
                        token_data[k]
                        for k in range(start, end + 1)
                        if token_data[k]["token"].strip()
                    ]

                    if not window_items:
                        continue

                    window_probs = [item["prob"] for item in window_items]
                    window_logprobs = [item["logprob"] for item in window_items]

                    # Geometric mean for confidence
                    log_sum = sum(math.log(p + 1e-12) for p in window_probs)
                    geo_mean = math.exp(log_sum / len(window_probs))

                    # Mean logprob of the same best window
                    avg_logprob = sum(window_logprobs) / len(window_logprobs)

                    # Slightly prefer fuller matches
                    length_bonus = len(cleaned) / len(value_str)
                    score = geo_mean * min(length_bonus, 1.0)

                    if score > best_confidence:
                        best_confidence = score
                        best_logprob = avg_logprob

        if len(value_str) <= 2:
            best_confidence = min(best_confidence, 0.75)

        return {
            "llm_confidence": round(best_confidence, 4),
            "logprob": round(best_logprob, 4),
        }

    def extract_data(self, file_path: str) -> Dict[str, Any]:
        """
        End-to-end Qwen extraction.

        Output is page-wise structured JSON with:
        - value
        - llm_confidence
        - logprob
        - review_required
        """
        images = load_images(file_path)
        final_output: Dict[str, Any] = {}

        for i, image in enumerate(images, start=1):
            print(f"Processing page {i}")

            result_text, token_data = self.extract_with_logprobs(image)
            cleaned = result_text.replace("```json", "").replace("```", "").strip()

            try:
                data = json.loads(cleaned)
            except Exception as e:
                print(f"⚠️ JSON parsing failed on page {i}: {e}")
                final_output[f"page_{i}"] = {"raw_output": cleaned}
                continue

            structured_data: Dict[str, Any] = {}

            for key, value in data.items():
                if isinstance(value, list):
                    items = []
                    for item in value:
                        # Qwen sometimes returns dicts for charges/units/cpt_codes
                        # instead of plain strings. Flatten them defensively.
                        if isinstance(item, dict):
                            item = (
                                item.get("amount")
                                or item.get("charge")
                                or item.get("value")
                                or item.get("total")
                                or item.get("code")
                                or item.get("unit")
                                or str(item)
                            )
                        metrics = self.compute_field_metrics(item, token_data)
                        items.append({
                            "value": item,
                            "llm_confidence": metrics["llm_confidence"],
                            "logprob": metrics["logprob"],
                            "review_required": metrics["llm_confidence"] < 0.80
                        })
                    structured_data[key] = items
                else:
                    metrics = self.compute_field_metrics(value, token_data)
                    structured_data[key] = {
                        "value": value,
                        "llm_confidence": metrics["llm_confidence"],
                        "logprob": metrics["logprob"],
                        "review_required": metrics["llm_confidence"] < 0.80
                    }

            final_output[f"page_{i}"] = structured_data

        return final_output


# ============================================================
# OCR FIELD MATCHER
# ============================================================
class OCRFieldMatcher:
    """
    Qwen decides the field name.
    OCR only provides supporting evidence for that field.

    For address fields:
    - use anchor-based multi-line window matching
    """

    def __init__(self):
        pass

    def _find_field_anchors(
        self,
        field_name: str,
        ocr_lines: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Find OCR lines that look like labels for the given field.
        """
        aliases = FIELD_LABEL_ALIASES.get(field_name, [])
        anchors = []

        for line in ocr_lines:
            text_norm = normalize_text(line["text"])

            for alias in aliases:
                alias_norm = normalize_text(alias)
                if alias_norm and alias_norm in text_norm:
                    anchors.append(line)
                    break

        return anchors

    def _text_match_score(self, field_name: str, qwen_value: Any, ocr_text: str) -> float:
        """
        Text-only similarity score between Qwen value and OCR text.
        """
        q_raw = normalize_text(qwen_value)
        o_raw = normalize_text(ocr_text)

        q_code = normalize_code_like_text(qwen_value)
        o_code = normalize_code_like_text(ocr_text)

        if not q_raw or not o_raw:
            return 0.0

        if q_raw == o_raw:
            return 1.0

        if q_code and o_code and q_code == o_code:
            return 0.98

        if q_raw in o_raw or o_raw in q_raw:
            return 0.92

        if q_code and o_code and (q_code in o_code or o_code in q_code):
            return 0.90

        ratio_raw = SequenceMatcher(None, q_raw, o_raw).ratio()
        ratio_code = SequenceMatcher(None, q_code, o_code).ratio() if q_code and o_code else 0.0

        return max(ratio_raw, ratio_code)

    def _anchor_proximity_score(
        self,
        candidate_line: Dict[str, Any],
        anchors: List[Dict[str, Any]]
    ) -> float:
        """
        If anchors exist, prefer OCR text physically closer to the field label.
        """
        if not anchors:
            return 0.0

        candidate_center = candidate_line["center"]
        distances = [
            euclidean_distance(candidate_center, anchor["center"])
            for anchor in anchors
        ]
        min_dist = min(distances)
        return math.exp(-min_dist / 250.0)

    def _merge_bboxes(self, bboxes: List[List[List[float]]]) -> Optional[List[List[float]]]:
        """
        Merge multiple OCR boxes into one rectangle-like polygon.
        """
        if not bboxes:
            return None

        all_x = []
        all_y = []

        for bbox in bboxes:
            for pt in bbox:
                all_x.append(pt[0])
                all_y.append(pt[1])

        x_min, x_max = min(all_x), max(all_x)
        y_min, y_max = min(all_y), max(all_y)

        return [
            [x_min, y_min],
            [x_max, y_min],
            [x_max, y_max],
            [x_min, y_max],
        ]

    def _extract_word_bboxes_from_line(self, line: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract word-level bboxes from an OCR line if available.
        PaddleOCR provides word-level info in the result structure.
        """
        import numpy as np

        if "words" in line:
            return line.get("words", [])

        return None

    def _calculate_proportional_bbox(
        self,
        full_bbox: List[List[float]],
        full_text: str,
        match_text: str,
    ) -> Optional[List[List[float]]]:
        """
        Calculate bbox for ONLY the portion of the OCR line that contains the
        Qwen-extracted value. Assumes left-to-right layout.

        Three progressive strategies so that proportional shrinking works even
        when Qwen and OCR text differ slightly:

        1. Exact normalized find  (lowercase + collapse spaces)
        2. Alphanumeric-only find (strips punctuation / dashes / spaces)
        3. Token-anchor find      (locate first + last significant tokens)

        Falls back to the full bbox only if all three strategies fail.
        """
        full_norm = normalize_text(full_text)
        match_norm = normalize_text(match_text)

        if not full_norm or not match_norm:
            return full_bbox

        total_chars = len(full_norm)
        if total_chars == 0:
            return full_bbox

        start_pos: int = -1
        end_pos: int = -1

        # ── Strategy 1: exact normalized substring ────────────────────────────
        idx = full_norm.find(match_norm)
        if idx != -1:
            start_pos = idx
            end_pos = idx + len(match_norm)

        # ── Strategy 2: alphanumeric-only find ───────────────────────────────
        if start_pos == -1:
            full_code = normalize_code_like_text(full_text)
            match_code = normalize_code_like_text(match_text)

            if full_code and match_code:
                code_idx = full_code.find(match_code)
                if code_idx != -1:
                    # Build a mapping: position in code-string → position in full_norm
                    code_to_norm: List[int] = [
                        i for i, ch in enumerate(full_norm) if ch.isalnum()
                    ]
                    if code_idx < len(code_to_norm):
                        start_pos = code_to_norm[code_idx]
                        code_end = code_idx + len(match_code) - 1
                        end_norm_idx = (
                            code_to_norm[code_end] + 1
                            if code_end < len(code_to_norm)
                            else total_chars
                        )
                        end_pos = end_norm_idx

        # ── Strategy 3: anchor on first and last significant tokens ──────────
        if start_pos == -1:
            match_tokens = [t for t in match_norm.split() if len(t) > 1]
            if match_tokens:
                first_tok = match_tokens[0]
                last_tok = match_tokens[-1]
                fp = full_norm.find(first_tok)
                if fp != -1:
                    lp = full_norm.rfind(last_tok, fp)
                    start_pos = fp
                    end_pos = (
                        (lp + len(last_tok))
                        if lp != -1
                        else min(fp + len(match_norm), total_chars)
                    )

        # ── Fallback: return the full line bbox unchanged ─────────────────────
        if start_pos == -1 or end_pos == -1:
            return full_bbox

        x_coords = [pt[0] for pt in full_bbox]
        y_coords = [pt[1] for pt in full_bbox]
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)

        start_ratio = start_pos / total_chars
        end_ratio = end_pos / total_chars

        new_x_min = x_min + (x_max - x_min) * start_ratio
        new_x_max = x_min + (x_max - x_min) * end_ratio

        return [
            [new_x_min, y_min],
            [new_x_max, y_min],
            [new_x_max, y_max],
            [new_x_min, y_max],
        ]

    def _build_candidate_windows_from_lines(
        self,
        lines: List[Dict[str, Any]],
        max_window_size: int = 4
    ) -> List[Dict[str, Any]]:
        """
        Build merged windows from consecutive OCR lines.

        This is especially useful for multi-line addresses.
        """
        if not lines:
            return []

        lines = sorted(lines, key=lambda x: (x["center"][1], x["center"][0]))
        windows = []

        for start in range(len(lines)):
            merged_lines = []

            for end in range(start, min(start + max_window_size, len(lines))):
                merged_lines.append(lines[end])
                merged_text = " ".join([ln["text"] for ln in merged_lines]).strip()
                merged_bbox = self._merge_bboxes([ln["bbox"] for ln in merged_lines])
                merged_conf = sum(ln["confidence"] for ln in merged_lines) / len(merged_lines)

                windows.append({
                    "text": merged_text,
                    "confidence": round(merged_conf, 4),
                    "bbox": merged_bbox,
                    "center": bbox_center(merged_bbox) if merged_bbox else (0, 0),
                    "lines": merged_lines
                })

        return windows

    def _collect_lines_near_anchor(
        self,
        anchor: Dict[str, Any],
        ocr_lines: List[Dict[str, Any]],
        max_vertical_gap: int = 220,
        max_horizontal_gap: int = 500
    ) -> List[Dict[str, Any]]:
        """
        Collect OCR lines near and below a field label anchor.
        Good for fields like addresses.
        """
        ax, ay = anchor["center"]
        nearby = []

        for line in ocr_lines:
            if line is anchor:
                continue

            lx, ly = line["center"]

            vertical_ok = (ly >= ay - 20) and (ly <= ay + max_vertical_gap)
            horizontal_ok = abs(lx - ax) <= max_horizontal_gap

            if vertical_ok and horizontal_ok:
                nearby.append(line)

        return sorted(nearby, key=lambda x: (x["center"][1], x["center"][0]))

    def _match_address_block(
        self,
        field_name: str,
        qwen_value: Any,
        ocr_lines: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Address-specific matcher:
        1. find anchor label
        2. gather nearby lines
        3. build candidate windows
        4. choose best window
        """
        q_norm = normalize_text(qwen_value)
        q_code = normalize_code_like_text(qwen_value)

        if not q_norm:
            return {
                "ocr_match_text": None,
                "ocr_confidence": None,
                "ocr_bbox": None,
                "ocr_bbox_list": None,
                "ocr_match_score": 0.0
            }

        anchors = self._find_field_anchors(field_name, ocr_lines)
        candidate_windows = []

        if anchors:
            for anchor in anchors:
                # Use a tighter horizontal gap (320px) to avoid pulling in
                # adjacent columns (e.g. SHIP TO when looking for BILL TO).
                nearby_lines = self._collect_lines_near_anchor(
                    anchor, ocr_lines,
                    max_vertical_gap=220,
                    max_horizontal_gap=320,
                )
                candidate_windows.extend(
                    self._build_candidate_windows_from_lines(nearby_lines, max_window_size=4)
                )
        else:
            # No anchor found — fall back but limit to top-half of page to
            # reduce false positives from footer / table rows.
            sorted_lines = sorted(ocr_lines, key=lambda x: x["center"][1])
            mid_y = sorted_lines[len(sorted_lines) // 2]["center"][1] if sorted_lines else 9999
            top_half = [l for l in sorted_lines if l["center"][1] <= mid_y]
            candidate_windows = self._build_candidate_windows_from_lines(top_half, max_window_size=4)

        best_candidate = None
        best_score = 0.0

        for candidate in candidate_windows:
            c_norm = normalize_text(candidate["text"])
            c_code = normalize_code_like_text(candidate["text"])

            if not c_norm:
                continue

            ratio_raw = SequenceMatcher(None, q_norm, c_norm).ratio()
            ratio_code = SequenceMatcher(None, q_code, c_code).ratio() if q_code and c_code else 0.0

            substring_bonus = 0.0
            if q_norm in c_norm or c_norm in q_norm:
                substring_bonus = 0.25

            q_tokens = set(q_norm.split())
            c_tokens = set(c_norm.split())
            overlap = len(q_tokens.intersection(c_tokens))
            token_overlap_bonus = min(overlap * 0.04, 0.20)

            total_score = max(ratio_raw, ratio_code) + substring_bonus + token_overlap_bonus
            total_score = min(total_score, 1.0)

            if total_score > best_score:
                best_score = total_score
                best_candidate = candidate

        if best_candidate is None or best_score < 0.50:
            return {
                "ocr_match_text": None,
                "ocr_confidence": None,
                "ocr_bbox": None,
                "ocr_bbox_list": None,
                "ocr_match_score": round(best_score, 4)
            }

        # Tighten the overall merged bbox to just the Qwen value portion.
        refined_bbox = self._calculate_proportional_bbox(
            best_candidate["bbox"],
            best_candidate["text"],
            qwen_value,
        )

        # Build a per-line bbox list so the UI can draw one tight box per
        # address line instead of one giant merged rectangle.
        #
        # IMPORTANT: only include lines whose tokens meaningfully overlap with
        # the Qwen address value.  Without this filter, stray lines in the
        # matched window (e.g. "DATE", "10/08/2025", "$650.00") get their
        # own highlight boxes even though they are not part of the address.
        qwen_tokens = {
            t for t in normalize_text(qwen_value).split() if len(t) > 2
        }

        per_line_bboxes: List[List[List[float]]] = []
        for component_line in best_candidate.get("lines", []):
            line_bbox = component_line.get("bbox")
            line_text = component_line.get("text", "")
            if not line_bbox:
                continue

            line_tokens = {
                t for t in normalize_text(line_text).split() if len(t) > 2
            }
            line_code  = normalize_code_like_text(line_text)
            value_code = normalize_code_like_text(qwen_value)

            # Accept the line if at least one meaningful token is shared, OR
            # if the alphanumeric form of the line appears in the value.
            has_token_overlap = bool(qwen_tokens & line_tokens)
            has_code_match    = bool(line_code) and (line_code in value_code)

            if not has_token_overlap and not has_code_match:
                continue  # skip stray lines (dates, amounts, labels)

            tightened = self._calculate_proportional_bbox(
                line_bbox, line_text, qwen_value
            )
            per_line_bboxes.append(tightened)

        return {
            "ocr_match_text": best_candidate["text"],
            "ocr_confidence": best_candidate["confidence"],
            "ocr_bbox": refined_bbox,
            "ocr_bbox_list": per_line_bboxes if per_line_bboxes else None,
            "ocr_match_score": round(best_score, 4)
        }

    def _match_word_level(
        self,
        qwen_value: str,
        ocr_line: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """
        Try to find exact word-level match within OCR line.
        Returns word bbox if found.
        """
        words = ocr_line.get("words", [])
        if not words:
            return None

        q_norm = normalize_text(qwen_value)
        
        for word in words:
            w_norm = normalize_text(word.get("text", ""))
            if q_norm == w_norm or q_norm in w_norm or w_norm in q_norm:
                return {
                    "bbox": word.get("bbox"),
                    "text": word.get("text"),
                    "confidence": word.get("confidence"),
                }
        
        return None

    def match_value_to_ocr(
        self,
        field_name: str,
        qwen_value: Any,
        ocr_lines: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Main field-to-OCR matcher.

        Address fields use special multi-line matching.
        Other fields use line-level matching.
        
        Now tries word-level matching first for better bbox precision.
        """
        if qwen_value is None:
            return {
                "ocr_match_text": None,
                "ocr_confidence": None,
                "ocr_bbox": None,
                "ocr_match_score": 0.0
            }

        qwen_value_str = str(qwen_value).strip()
        if not qwen_value_str:
            return {
                "ocr_match_text": None,
                "ocr_confidence": None,
                "ocr_bbox": None,
                "ocr_match_score": 0.0
            }

        if field_name in {"billing_address", "practice_address"}:
            return self._match_address_block(field_name, qwen_value_str, ocr_lines)

        # First try word-level matching for better bbox
        for line in ocr_lines:
            word_match = self._match_word_level(qwen_value_str, line)
            if word_match and word_match.get("bbox") is not None:
                text_score = self._text_match_score(field_name, qwen_value_str, word_match["text"])
                if text_score >= 0.50:
                    bbox = word_match["bbox"]
                    if isinstance(bbox, list) and len(bbox) > 0:
                        if isinstance(bbox[0], list):
                            return {
                                "ocr_match_text": word_match["text"],
                                "ocr_confidence": word_match.get("confidence"),
                                "ocr_bbox": bbox,
                                "ocr_match_score": round(text_score, 4)
                            }
                        elif isinstance(bbox[0], (int, float)):
                            polygon_bbox = [
                                [bbox[0], bbox[1]],
                                [bbox[0] + bbox[2], bbox[1]],
                                [bbox[0] + bbox[2], bbox[1] + bbox[3]],
                                [bbox[0], bbox[1] + bbox[3]],
                            ]
                            return {
                                "ocr_match_text": word_match["text"],
                                "ocr_confidence": word_match.get("confidence"),
                                "ocr_bbox": polygon_bbox,
                                "ocr_match_score": round(text_score, 4)
                            }

        anchors = self._find_field_anchors(field_name, ocr_lines)

        best_line = None
        best_score = 0.0
        best_bbox = None

        for line in ocr_lines:
            text_score = self._text_match_score(field_name, qwen_value_str, line["text"])
            proximity_score = self._anchor_proximity_score(line, anchors)

            total_score = (0.80 * text_score) + (0.20 * proximity_score)

            if total_score > best_score:
                best_score = total_score
                best_line = line 

        if best_line is None or best_score < 0.55:
            return {
                "ocr_match_text": None,
                "ocr_confidence": None,
                "ocr_bbox": None,
                "ocr_match_score": round(best_score, 4)
            }

        # Calculate proportional bbox for the matched portion
        refined_bbox = self._calculate_proportional_bbox(
            best_line["bbox"],
            best_line["text"],
            qwen_value_str
        )

        return {
            "ocr_match_text": best_line["text"],
            "ocr_confidence": round(float(best_line["confidence"]), 4),
            "ocr_bbox": refined_bbox,
            "ocr_match_score": round(best_score, 4)
        }


# ============================================================
# HYBRID EXTRACTOR
# ============================================================
class HybridExtractor:
    """
    Full hybrid pipeline:

    1. Qwen extracts structured values
    2. Qwen metrics added:
       - llm_confidence
       - logprob
    3. PaddleOCR extracts text + confidence + bbox
    4. OCR matcher aligns OCR evidence to each Qwen field

    Final output keeps both confidence sources separately.
    """

    def __init__(
        self,
        qwen_model_path: str = "/home/rohit.sahu/Qwen_model/qwen_models/Qwen2.5-VL-3B-Instruct",
    ):
        self.qwen = QwenExtractor(model_path=qwen_model_path)
        self.ocr = PaddleOCREngine()
        self.matcher = OCRFieldMatcher()

    def extract_data(self, file_path: str) -> Dict[str, Any]:
        """
        Run the full hybrid extraction pipeline.
        """
        images = load_images(file_path)

        qwen_result = self.qwen.extract_data(file_path)
        ocr_result = self.ocr.extract_from_images(images)

        final_result: Dict[str, Any] = {}

        for page_name, page_data in qwen_result.items():
            if "raw_output" in page_data:
                final_result[page_name] = page_data
                continue

            page_ocr_lines = ocr_result.get(page_name, [])
            enriched_page: Dict[str, Any] = {}

            for field_name, field_data in page_data.items():
                if isinstance(field_data, list):
                    enriched_items = []

                    for item in field_data:
                        qwen_value = item.get("value")
                        ocr_info = self.matcher.match_value_to_ocr(
                            field_name=field_name,
                            qwen_value=qwen_value,
                            ocr_lines=page_ocr_lines
                        )

                        enriched_item = {
                            "value": qwen_value,
                            "llm_confidence": item.get("llm_confidence"),
                            "logprob": item.get("logprob"),
                            "review_required": item.get("review_required"),
                            "ocr_match_text": ocr_info["ocr_match_text"],
                            "ocr_confidence": ocr_info["ocr_confidence"],
                            "ocr_bbox": ocr_info["ocr_bbox"],
                            "ocr_bbox_list": ocr_info.get("ocr_bbox_list"),
                            "ocr_match_score": ocr_info["ocr_match_score"],
                        }
                        enriched_items.append(enriched_item)

                    enriched_page[field_name] = enriched_items

                else:
                    qwen_value = field_data.get("value")
                    ocr_info = self.matcher.match_value_to_ocr(
                        field_name=field_name,
                        qwen_value=qwen_value,
                        ocr_lines=page_ocr_lines
                    )

                    enriched_page[field_name] = {
                        "value": qwen_value,
                        "llm_confidence": field_data.get("llm_confidence"),
                        "logprob": field_data.get("logprob"),
                        "review_required": field_data.get("review_required"),
                        "ocr_match_text": ocr_info["ocr_match_text"],
                        "ocr_confidence": ocr_info["ocr_confidence"],
                        "ocr_bbox": ocr_info["ocr_bbox"],
                        "ocr_bbox_list": ocr_info.get("ocr_bbox_list"),
                        "ocr_match_score": ocr_info["ocr_match_score"],
                    }

            final_result[page_name] = enriched_page

        return final_result

    def flatten_for_table(self, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Flatten nested result to row-wise table records for Streamlit.
        """
        rows: List[Dict[str, Any]] = []

        for page, page_data in result.items():
            if "raw_output" in page_data:
                rows.append({
                    "page": page,
                    "field": "raw_output",
                    "item_index": None,
                    "value": page_data["raw_output"],
                    "llm_confidence": None,
                    "logprob": None,
                    "ocr_match_text": None,
                    "ocr_confidence": None,
                    "ocr_match_score": None,
                    "review_required": None,
                    "ocr_bbox": None,
                })
                continue

            for field, details in page_data.items():
                if isinstance(details, list):
                    for idx, item in enumerate(details, start=1):
                        rows.append({
                            "page": page,
                            "field": field,
                            "item_index": idx,
                            "value": item.get("value"),
                            "llm_confidence": item.get("llm_confidence"),
                            "logprob": item.get("logprob"),
                            "ocr_match_text": item.get("ocr_match_text"),
                            "ocr_confidence": item.get("ocr_confidence"),
                            "ocr_match_score": item.get("ocr_match_score"),
                            "review_required": item.get("review_required"),
                            "ocr_bbox": item.get("ocr_bbox"),
                            "ocr_bbox_list": item.get("ocr_bbox_list"),
                        })
                else:
                    rows.append({
                        "page": page,
                        "field": field,
                        "item_index": None,
                        "value": details.get("value"),
                        "llm_confidence": details.get("llm_confidence"),
                        "logprob": details.get("logprob"),
                        "ocr_match_text": details.get("ocr_match_text"),
                        "ocr_confidence": details.get("ocr_confidence"),
                        "ocr_match_score": details.get("ocr_match_score"),
                        "review_required": details.get("review_required"),
                        "ocr_bbox": details.get("ocr_bbox"),
                    })

        return rows


if __name__ == "__main__":
    extractor = HybridExtractor()

    result = extractor.extract_data(
        "/home/rohit.sahu/Qwen_model/samples_nonstandard_data/Document_4.pdf"
    )

    print(json.dumps(result, indent=2))