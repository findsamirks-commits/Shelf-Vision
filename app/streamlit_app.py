"""Streamlit Interactive Application for Shelf-Vision.

Planogram Compliance & Out-of-Stock (OOS) Detection for Retail Shelves.
Provides live shelf audit visualizations, Hungarian matching diagnostics,
brand share-of-shelf charts, and downloadable compliance JSON reports.
"""

import json
import os
import cv2
import pandas as pd
from PIL import Image
import streamlit as st

from src.inference import ShelfVisionPipeline
from src.data_prep import generate_synthetic_shelf_image, load_planogram

# -----------------------------------------------------------------------------
# Streamlit Page Setup
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Shelf-Vision 🛒👁️ | Planogram Compliance & OOS",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #0F172A;
        margin-bottom: 0.1rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.2rem;
    }
    .status-compliant { color: #16A34A; font-weight: bold; }
    .status-oos { color: #DC2626; font-weight: bold; }
    .status-misplaced { color: #D97706; font-weight: bold; }
    .status-encroached { color: #9333EA; font-weight: bold; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------
st.markdown('<div class="main-title">🛒👁️ Shelf-Vision: Planogram Compliance & OOS Detection</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">'
    "Autonomous Computer Vision audit pipeline for quick-commerce dark stores & supermarket shelves. "
    "Powered by <b>YOLOv8 Object Detection</b> and <b>Bipartite Hungarian Spatial Matching</b>."
    "</div>",
    unsafe_allow_html=True,
)

# Paths
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONFIG_PATH = os.path.join(ROOT_DIR, "configs", "planogram_sample.json")
DEMO_IMG_PATH = os.path.join(ROOT_DIR, "assets", "demo_shelf.jpg")

# -----------------------------------------------------------------------------
# Sidebar: Shelf Audit Configuration & Controls
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Shelf Audit Setup")

    st.subheader("1. Planogram Specification")
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        planogram_data = json.load(f)

    shelf_id = planogram_data.get("shelf_id", "MILKBASKET-HUB-A4-BAY2")
    category = planogram_data.get("category", "Dairy & Breakfast")
    st.info(f"**Shelf:** `{shelf_id}`\n\n**Category:** `{category}`\n\n**Target Slots:** `{len(planogram_data['slots'])} facings`")

    st.markdown("---")
    st.subheader("2. Test Scenarios (Synthetic Generator)")
    st.caption("Inject operational failure modes to test pipeline resilience:")

    test_oos_1 = st.checkbox("Simulate OOS: Mother Dairy Butter (S1-P03)", value=True)
    test_oos_2 = st.checkbox("Simulate OOS: Nestle Milkmaid (S2-P04)", value=True)
    test_misplace = st.checkbox("Simulate Misplacement: S2-P02 (Nestle in Amul slot)", value=True)
    test_encroach = st.checkbox("Simulate Encroachment: S3-P04 (Rogue Competitor Ghee)", value=True)

    if st.button("🔄 Regenerate Audit Image", use_container_width=True):
        oos_slots = []
        if test_oos_1: oos_slots.append("S1-P03")
        if test_oos_2: oos_slots.append("S2-P04")

        misplaced_dict = {"S2-P02": "Nestle Everyday Dairy Whitener 1kg"} if test_misplace else {}
        encroached_dict = {"S3-P04": "Competitor Rogue Ghee 1L"} if test_encroach else {}

        generate_synthetic_shelf_image(
            planogram=planogram_data,
            output_image_path=DEMO_IMG_PATH,
            oos_slot_ids=oos_slots,
            misplaced_slot_ids=misplaced_dict,
            encroached_slot_ids=encroached_dict,
        )
        st.success("New shelf audit photo rendered!")

    st.markdown("---")
    st.markdown("### 🏷️ Legend")
    st.markdown("🟩 **Compliant:** SKU in expected slot")
    st.markdown("⬜ **OOS:** Expected SKU missing / empty space")
    st.markdown("🟨 **Misplaced:** Wrong SKU in slot")
    st.markdown("🟥 **Encroached:** Competitor/rogue SKU in slot")

    st.markdown("---")
    st.markdown("### 👨‍💻 Developer & Architect")
    st.markdown("**Samir Sandhwar** | Category Head & AI Engineer")
    st.markdown("[![GitHub](https://img.shields.io/badge/GitHub-findsamirks--commits-black?style=flat&logo=github)](https://github.com/findsamirks-commits/Shelf-Vision)")
    st.markdown("[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-findsamirks-FFD21E?style=flat&logo=huggingface&logoColor=black)](https://huggingface.co/findsamirks)")
    st.markdown("[![LinkedIn](https://img.shields.io/badge/LinkedIn-samirsandhwar-0077B5?style=flat&logo=linkedin)](https://www.linkedin.com/in/samirsandhwar)")

# -----------------------------------------------------------------------------
# Main Application Flow
# -----------------------------------------------------------------------------
# Initialize Pipeline
pipeline = ShelfVisionPipeline(planogram_path=CONFIG_PATH)

# Ensure demo image exists
if not os.path.exists(DEMO_IMG_PATH):
    generate_synthetic_shelf_image(
        planogram=planogram_data,
        output_image_path=DEMO_IMG_PATH,
        oos_slot_ids=["S1-P03", "S2-P04"],
        misplaced_slot_ids={"S2-P02": "Nestle Everyday Dairy Whitener 1kg"},
        encroached_slot_ids={"S3-P04": "Competitor Rogue Ghee 1L"},
    )

# Run Inference & Audit
report, annotated_cv2_img = pipeline.audit_shelf(DEMO_IMG_PATH)
annotated_rgb = cv2.cvtColor(annotated_cv2_img, cv2.COLOR_BGR2RGB)

# -----------------------------------------------------------------------------
# Executive KPI Scorecard
# -----------------------------------------------------------------------------
st.markdown("### 📊 Executive Shelf Audit Scorecard")
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

compliance_color = "normal" if report["compliance_score"] >= 80 else "inverse"
kpi1.metric("Compliance Index", f"{report['compliance_score']}%", delta=f"{report['compliant_facings']}/{report['total_slots']} Facings")
kpi2.metric("Out-of-Stock (OOS)", f"{report['oos_facings']} Facings", delta=f"-{report['oos_rate_pct']}% OOS", delta_color="inverse")
kpi3.metric("Misplaced SKUs", f"{report['misplaced_facings']} Slots")
kpi4.metric("Encroached Space", f"{report['encroached_facings']} Slots")
kpi5.metric("Inference Latency", f"{report['inference_latency_ms']} ms", help="Evaluated on 1280x720 shelf frame")

st.markdown("---")

# -----------------------------------------------------------------------------
# Visual Inspection: Side-by-Side (Raw Image vs. Annotated Detections)
# -----------------------------------------------------------------------------
col_left, col_right = st.columns([1.1, 0.9])

with col_left:
    st.subheader("🖼️ Computer Vision Audit Inspection")
    view_option = st.radio("Display View:", ["Annotated Detections (YOLOv8 + Bipartite)", "Raw Shelf Photo"], horizontal=True)

    if view_option.startswith("Annotated"):
        st.image(annotated_rgb, caption="Shelf-Vision Automated Detection Overlay (Bounding Boxes + Compliance Flags)", use_container_width=True)
    else:
        raw_img = Image.open(DEMO_IMG_PATH)
        st.image(raw_img, caption="Raw Dark-Store Audit Photo (Before Detection)", use_container_width=True)

with col_right:
    st.subheader("📈 Brand Share-of-Shelf (SoS)")
    sos_data = report.get("share_of_shelf_pct", {})
    if sos_data:
        df_sos = pd.DataFrame(list(sos_data.items()), columns=["Brand", "Share %"]).sort_values(by="Share %", ascending=False)
        st.bar_chart(df_sos.set_index("Brand"))
        st.caption("Distribution of on-shelf facings across competing brands.")

    st.subheader("🚨 Priority Operational Action Items")
    if report["oos_slots"]:
        for item in report["oos_slots"]:
            st.error(f"🔴 **OOS ALERT:** `{item['slot_id']}` is empty! Expected: **{item['expected_sku']}** ({item['expected_brand']})")
    if report["misplaced_slots"]:
        for item in report["misplaced_slots"]:
            st.warning(f"🟡 **MISPLACED:** `{item['slot_id']}` contains *{item['detected_sku']}*, expected: **{item['expected_sku']}**")
    if report["encroached_slots"]:
        for item in report["encroached_slots"]:
            st.error(f"🟣 **ENCROACHMENT:** `{item['slot_id']}` encroached by *{item['detected_sku']}*!")

# -----------------------------------------------------------------------------
# Granular Slot-by-Slot Audit Table
# -----------------------------------------------------------------------------
st.markdown("---")
st.subheader("📋 Granular Slot-by-Slot Hungarian Assignment Matrix")

table_rows = []
for m in report["matched_slots"]:
    status = m["status"]
    table_rows.append({
        "Slot ID": m["slot_id"],
        "Tier": f"Tier {m['shelf_tier']}",
        "Expected SKU": m["expected_sku"],
        "Brand": m["expected_brand"],
        "Detected Item": m["detected_sku"] if m["detected_sku"] else "— (EMPTY) —",
        "Spatial IoU": m["iou"],
        "Detection Conf": f"{m['confidence']:.2f}" if m["confidence"] > 0 else "0.00",
        "Audit Status": status,
    })

df_table = pd.DataFrame(table_rows)
st.dataframe(df_table, use_container_width=True, hide_index=True)

# -----------------------------------------------------------------------------
# Export / Download JSON Report
# -----------------------------------------------------------------------------
st.markdown("---")
col_d1, col_d2 = st.columns([0.7, 0.3])
with col_d1:
    st.markdown("### 📥 Automated Integration Payload (ERP / WMS)")
    st.caption("This JSON audit payload directly triggers automated Purchase Orders in `Agentic-Inventory-Optimizer` or task lists for pickers.")
with col_d2:
    st.download_button(
        label="💾 Download Audit Report (JSON)",
        data=json.dumps(report, indent=2),
        file_name=f"shelf_audit_{shelf_id}.json",
        mime="application/json",
        use_container_width=True,
    )

with st.expander("🔍 View Raw JSON Audit Payload", expanded=False):
    st.json(report)
