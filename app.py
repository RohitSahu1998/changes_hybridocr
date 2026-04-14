import json
import os
import tempfile
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw

from hybrid_engine import HybridExtractor, load_images


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Hybrid Extraction Review UI",
    page_icon="📄",
    layout="wide",
)


# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
.main-title {
    font-size: 30px;
    font-weight: 700;
    margin-bottom: 4px;
}
.sub-title {
    font-size: 14px;
    color: #666;
    margin-bottom: 16px;
}
.viewer-card {
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 12px;
    background: #ffffff;
}
.right-card {
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 12px;
    background: #ffffff;
}
.row-box {
    border: 1px solid #edf0f2;
    border-radius: 10px;
    padding: 10px;
    margin-bottom: 8px;
    background: #fafafa;
}
.small-text {
    color: #555;
    font-size: 13px;
}
.review-yes {
    color: #b91c1c;
    font-weight: 700;
}
.review-no {
    color: #166534;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# CACHE EXTRACTOR
# ============================================================
@st.cache_resource
def load_extractor():
    return HybridExtractor(
        qwen_model_path="/home/rohit.sahu/Qwen_model/qwen_models/Qwen2.5-VL-3B-Instruct"
    )


# ============================================================
# HELPERS
# ============================================================
def save_uploaded_file(uploaded_file):
    suffix = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        return tmp.name


def page_name_to_index(page_name: str) -> int:
    try:
        return int(page_name.split("_")[-1]) - 1
    except Exception:
        return 0


def index_to_page_name(page_idx: int) -> str:
    return f"page_{page_idx + 1}"


def draw_highlight_on_image(
    image: Image.Image,
    bbox: Optional[List[List[float]]] = None,
    bbox_list: Optional[List[List[List[float]]]] = None,
    outline_color: str = "red",
    fill_color=(255, 0, 0, 45),
    line_width: int = 4,
) -> Image.Image:
    """
    Draw highlight box(es) on the document image.

    For regular fields  : a single bbox is drawn.
    For address fields  : bbox_list (one box per OCR line) is drawn so each
                          line of the address gets its own tight highlight.
    bbox_list takes priority over bbox when both are provided.
    """
    img = image.copy().convert("RGBA")
    overlay = Image.new("RGBA", img.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)

    # Prefer the per-line list for address fields; fall back to single bbox
    boxes_to_draw = bbox_list if bbox_list else ([bbox] if bbox else [])

    if not boxes_to_draw:
        return img.convert("RGB")

    for box in boxes_to_draw:
        if not box:
            continue
        polygon = [(int(pt[0]), int(pt[1])) for pt in box]
        draw.polygon(polygon, fill=fill_color, outline=outline_color)
        draw.line(polygon + [polygon[0]], fill=outline_color, width=line_width)

    combined = Image.alpha_composite(img, overlay)
    return combined.convert("RGB")


def flatten_result_for_actions(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []

    for page_name, page_data in result.items():
        if "raw_output" in page_data:
            continue

        for field_name, field_data in page_data.items():
            if isinstance(field_data, list):
                for idx, item in enumerate(field_data, start=1):
                    rows.append({
                        "page": page_name,
                        "field": field_name,
                        "item_index": idx,
                        "value": item.get("value"),
                        "llm_confidence": item.get("llm_confidence"),
                        "logprob": item.get("logprob"),  # kept internally, not shown
                        "ocr_match_text": item.get("ocr_match_text"),
                        "ocr_confidence": item.get("ocr_confidence"),
                        "ocr_match_score": item.get("ocr_match_score"),
                        "ocr_bbox": item.get("ocr_bbox"),
                        "ocr_bbox_list": item.get("ocr_bbox_list"),
                        "review_required": item.get("review_required"),
                        # CPT validation — only populated for cpt_codes rows
                        "cpt_valid":    item.get("cpt_valid"),
                        "cpt_category": item.get("cpt_category"),
                        "cpt_reason":   item.get("cpt_reason"),
                    })
            else:
                rows.append({
                    "page": page_name,
                    "field": field_name,
                    "item_index": None,
                    "value": field_data.get("value"),
                    "llm_confidence": field_data.get("llm_confidence"),
                    "logprob": field_data.get("logprob"),  # kept internally, not shown
                    "ocr_match_text": field_data.get("ocr_match_text"),
                    "ocr_confidence": field_data.get("ocr_confidence"),
                    "ocr_match_score": field_data.get("ocr_match_score"),
                    "ocr_bbox": field_data.get("ocr_bbox"),
                    "ocr_bbox_list": field_data.get("ocr_bbox_list"),
                    "review_required": field_data.get("review_required"),
                })

    return rows


def initialize_state():
    defaults = {
        "result": None,
        "images": None,
        "current_page_idx": 0,
        "selected_bbox": None,
        "selected_bbox_list": None,
        "selected_field": None,
        "selected_value": None,
        "uploaded_file_path": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ============================================================
# INIT STATE
# ============================================================
initialize_state()


# ============================================================
# HEADER
# ============================================================
st.markdown('<div class="main-title">📄 Hybrid Extraction Review UI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Left side shows the document page. Right side shows extracted fields. Qwen confidence and PaddleOCR confidence are displayed separately.</div>',
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.header("Options")
show_only_flagged = st.sidebar.checkbox("Show only review-required rows", value=False)
show_only_unmatched = st.sidebar.checkbox("Show only OCR-unmatched rows", value=False)
show_debug_json = st.sidebar.checkbox("Show raw JSON below", value=False)
show_preview_table = st.sidebar.checkbox("Show flattened table below", value=False)


# ============================================================
# FILE UPLOAD
# ============================================================
uploaded_file = st.file_uploader(
    "Upload a PDF or image",
    type=["pdf", "png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    file_path = save_uploaded_file(uploaded_file)
    st.session_state["uploaded_file_path"] = file_path

    extractor = load_extractor()

    top_left, top_mid, top_right = st.columns([2, 1, 1])
    with top_left:
        st.info(f"Uploaded file: {uploaded_file.name}")

    with top_mid:
        run_clicked = st.button("Run Extraction", type="primary", use_container_width=True)

    with top_right:
        clear_highlight = st.button("Clear Highlight", use_container_width=True)

    if clear_highlight:
        st.session_state["selected_bbox"] = None
        st.session_state["selected_bbox_list"] = None
        st.session_state["selected_field"] = None
        st.session_state["selected_value"] = None

    if run_clicked:
        with st.spinner("Running extraction..."):
            result = extractor.extract_data(file_path)
            images = load_images(file_path)

        st.session_state["result"] = result
        st.session_state["images"] = images
        st.session_state["current_page_idx"] = 0
        st.session_state["selected_bbox"] = None
        st.session_state["selected_bbox_list"] = None
        st.session_state["selected_field"] = None
        st.session_state["selected_value"] = None

    if st.session_state["result"] is not None and st.session_state["images"] is not None:
        result = st.session_state["result"]
        images = st.session_state["images"]

        rows = flatten_result_for_actions(result)

        if show_only_flagged:
            rows = [r for r in rows if r.get("review_required") is True]

        if show_only_unmatched:
            rows = [r for r in rows if r.get("ocr_match_text") is None]

        nav1, nav2, nav3, nav4, nav5 = st.columns([1, 1, 2, 1, 1])

        with nav1:
            if st.button("Previous", use_container_width=True):
                st.session_state["current_page_idx"] = max(0, st.session_state["current_page_idx"] - 1)
                st.session_state["selected_bbox"] = None

        with nav2:
            if st.button("Next", use_container_width=True):
                st.session_state["current_page_idx"] = min(len(images) - 1, st.session_state["current_page_idx"] + 1)
                st.session_state["selected_bbox"] = None

        with nav3:
            st.markdown(
                f"<div style='text-align:center; padding-top:8px; font-weight:600;'>Page {st.session_state['current_page_idx'] + 1} / {len(images)}</div>",
                unsafe_allow_html=True
            )

        current_page_name = index_to_page_name(st.session_state["current_page_idx"])

        left_col, right_col = st.columns([1.25, 1.0], gap="large")

        with left_col:
            st.markdown('<div class="viewer-card">', unsafe_allow_html=True)
            st.subheader("Document Viewer")

            current_image = images[st.session_state["current_page_idx"]]
            highlighted_image = draw_highlight_on_image(
                current_image,
                bbox=st.session_state["selected_bbox"],
                bbox_list=st.session_state["selected_bbox_list"],
            )

            st.image(
                highlighted_image,
                caption=f"Showing {current_page_name}",
                use_container_width=True
            )

            if st.session_state["selected_field"] is not None:
                st.markdown(
                    f"""
                    <div class="row-box">
                        <b>Selected Field:</b> {st.session_state["selected_field"]}<br>
                        <b>Selected Value:</b> {st.session_state["selected_value"]}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.markdown('</div>', unsafe_allow_html=True)

        with right_col:
            st.markdown('<div class="right-card">', unsafe_allow_html=True)
            st.subheader("Review Panel")

            page_rows = [r for r in rows if r["page"] == current_page_name]

            if not page_rows:
                st.warning("No rows available for this page with the current filters.")
            else:
                header_cols = st.columns([1.7, 2.5, 1.0, 1.0, 1.1])
                header_cols[0].markdown("**Field**")
                header_cols[1].markdown("**Value / OCR Match**")
                header_cols[2].markdown("**Qwen Conf**")
                header_cols[3].markdown("**OCR Conf**")
                header_cols[4].markdown("**Action**")

                st.markdown("---")

                for idx, row in enumerate(page_rows):
                    c1, c2, c3, c4, c5 = st.columns([1.7, 2.5, 1.0, 1.0, 1.1])

                    field_label = row["field"]
                    if row["item_index"] is not None:
                        field_label = f"{field_label} [{row['item_index']}]"

                    with c1:
                        st.write(field_label)
                        if row.get("review_required"):
                            st.caption("⚠ Review required")

                    with c2:
                        st.write(row.get("value"))
                        ocr_text = row.get("ocr_match_text")
                        if ocr_text is not None:
                            st.caption(f"OCR: {ocr_text}")
                        else:
                            st.caption("OCR: Not matched")
                        # CPT validation badge
                        if row.get("field") == "cpt_codes" and row.get("cpt_valid") is not None:
                            if row["cpt_valid"]:
                                st.caption(f"✅ {row['cpt_category']}")
                            else:
                                st.caption(f"⛔ Invalid CPT — {row.get('cpt_reason', '')}")

                    with c3:
                        st.write(row.get("llm_confidence"))

                    with c4:
                        st.write(row.get("ocr_confidence"))

                    with c5:
                        action_key = f"action_{current_page_name}_{idx}"
                        if st.button("Highlight", key=action_key, use_container_width=True):
                            st.session_state["current_page_idx"] = page_name_to_index(row["page"])
                            st.session_state["selected_bbox"] = row.get("ocr_bbox")
                            st.session_state["selected_bbox_list"] = row.get("ocr_bbox_list")
                            st.session_state["selected_field"] = row.get("field")
                            st.session_state["selected_value"] = row.get("value")
                            st.rerun()

            st.markdown('</div>', unsafe_allow_html=True)

        if show_preview_table:
            st.subheader("Flattened Action Table")
            df = pd.DataFrame(rows)

            preferred_cols = [
                "page",
                "field",
                "item_index",
                "value",
                "llm_confidence",
                "ocr_confidence",
                "ocr_match_text",
                "ocr_match_score",
                "review_required",
            ]
            final_cols = [c for c in preferred_cols if c in df.columns]
            df = df[final_cols]

            st.dataframe(df, use_container_width=True)

        if show_debug_json:
            st.subheader("Raw JSON Output")
            st.json(result)

        st.subheader("Download Result")
        st.download_button(
            label="Download JSON",
            data=json.dumps(result, indent=2),
            file_name="hybrid_review_result.json",
            mime="application/json"
        )