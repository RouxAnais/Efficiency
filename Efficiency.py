import streamlit as st
import pandas as pd
import plotly.express as px
from io import BytesIO
import tempfile

from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image,
    Table, TableStyle, PageBreak
)
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Efficiency Analysis",
    layout="wide"
)

UPWARD_COLOR = "#FF0A05"      # rouge
DOWNWARD_COLOR = "#0088FF"    # bleu
BUTTON_COLOR = "#034418"


# ============================================================
# STYLE
# ============================================================

st.markdown(f"""
<style>

.stButton button {{
    background-color: {BUTTON_COLOR};
    color: white;
    border-radius: 10px;
    border: none;
    padding: 10px 18px;
    width: 100%;
}}

.stDownloadButton button {{
    background-color: {BUTTON_COLOR};
    color: white;
    border-radius: 10px;
    border: none;
    padding: 10px 18px;
}}

section[data-testid="stSidebar"] {{
    width: 300px !important;
}}

section[data-testid="stSidebar"] > div {{
    width: 300px !important;
}}

</style>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR FIXE
# ============================================================

with st.sidebar:

    st.markdown("## Efficiency Analysis")
    st.markdown("### NTN Europe")

    st.divider()

    uploaded_files = st.file_uploader(
        "Upload your CSV files",
        type=["csv"],
        accept_multiple_files=True
    )

    if uploaded_files:
        st.success(f"{len(uploaded_files)} file(s) uploaded")

    st.divider()


# ============================================================
# TITRE
# ============================================================

st.markdown("# Efficiency Analysis")
st.markdown("### Diversification - NTN Europe")


# ============================================================
# FONCTIONS
# ============================================================

def load_file(uploaded_file):

    raw_df = pd.read_csv(
        uploaded_file,
        sep=";",
        decimal=",",
        header=None,
        dtype=str
    )

    time_row = raw_df[raw_df[0] == "Temps"].index

    if len(time_row) == 0:
        return None, None

    info_lines = []
    info_df = raw_df.iloc[1:time_row[0]]

    for _, row in info_df.iterrows():
        values = row.dropna().astype(str).tolist()

        if values:
            info_lines.append(" ".join(values).strip())

    uploaded_file.seek(0)

    df = pd.read_csv(
        uploaded_file,
        sep=";",
        decimal=",",
        skiprows=time_row[0]
    )

    df.columns = [
        "Time",
        "Torque_0_5",
        "Torque_0_50",
        "Force",
        "Displacement",
        "Raw_upward_efficiency",
        "Raw_downward_efficiency",
        "Upward",
        "Downward",
        "Upward_efficiency_stats",
        "Downward_efficiency_stats"
    ]

    for col in ["Time", "Upward", "Downward"]:
        df[col] = df[col].astype(str).str.replace(",", ".", regex=False)
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.loc[df["Upward"] > 1, "Upward"] = None
    df.loc[df["Downward"] > 1, "Downward"] = None

    return df, info_lines


def clean_mean(series):

    series = series.dropna()
    series = series[(series > 0) & (series <= 1)]

    if series.empty:
        return None

    q_low = series.quantile(0.05)
    q_high = series.quantile(0.95)

    series_clean = series[
        (series >= q_low) &
        (series <= q_high)
    ]

    return series_clean.mean() * 100


def add_pass_numbers(df):

    df = df.copy()

    df["Upward_pass"] = None
    df["Downward_pass"] = None

    for col, pass_col in [
        ("Upward", "Upward_pass"),
        ("Downward", "Downward_pass")
    ]:
        valid = df[col].notna() & (df[col] > 0) & (df[col] <= 1)
        starts = valid & ~valid.shift(fill_value=False)
        pass_number = starts.cumsum()

        df.loc[valid, pass_col] = pass_number[valid]

    return df


def calculate_stats(df):

    df = add_pass_numbers(df)

    # Moyenne générale sans la première pass
    df_up = df[df["Upward_pass"] >= 2]
    df_down = df[df["Downward_pass"] >= 2]

    average_upward = clean_mean(df_up["Upward"])
    average_downward = clean_mean(df_down["Downward"])

    # Moyenne de chaque pass, y compris la première
    upward_by_pass = (
        df[df["Upward_pass"].notna()]
        .groupby("Upward_pass")["Upward"]
        .apply(clean_mean)
        .reset_index()
    )

    downward_by_pass = (
        df[df["Downward_pass"].notna()]
        .groupby("Downward_pass")["Downward"]
        .apply(clean_mean)
        .reset_index()
    )

    upward_by_pass.columns = ["Pass", "Average upward efficiency (%)"]
    downward_by_pass.columns = ["Pass", "Average downward efficiency (%)"]

    evolution_df = pd.merge(
        upward_by_pass,
        downward_by_pass,
        on="Pass",
        how="outer"
    )

    evolution_df["Pass"] = evolution_df["Pass"].astype(int)

    return average_upward, average_downward, evolution_df


def build_pdf(results_df, files_data):

    pdf_buffer = BytesIO()

    doc = SimpleDocTemplate(
        pdf_buffer,
        pagesize=A4,
        rightMargin=25,
        leftMargin=25,
        topMargin=25,
        bottomMargin=25
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "TitleStyle",
        parent=styles["Title"],
        textColor=colors.HexColor("#1f77b4"),
        fontSize=20,
        leading=24,
        alignment=1
    )

    heading_style = ParagraphStyle(
        "HeadingStyle",
        parent=styles["Heading1"],
        textColor=colors.HexColor("#1f77b4"),
        fontSize=14,
        leading=18
    )

    elements = []

    elements.append(Paragraph("Efficiency Analysis Report", title_style))
    elements.append(Spacer(1, 12))

    table_data = [list(results_df.columns)]

    for _, row in results_df.iterrows():
        table_data.append([row[col] for col in results_df.columns])

    col_width = 520 / len(results_df.columns)

    summary_table = Table(
        table_data,
        colWidths=[col_width] * len(results_df.columns),
        repeatRows=1
    )

    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f77b4")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f5f8fb")),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d0d7de")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 6),
        ("ALIGN", (1, 1), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))

    elements.append(summary_table)

    for item in files_data:

        elements.append(PageBreak())
        elements.append(Paragraph(item["name"], heading_style))
        elements.append(Spacer(1, 10))

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
            item["fig"].write_image(
                tmpfile.name,
                width=900,
                height=430,
                scale=2
            )
            main_chart_path = tmpfile.name

        img_main = Image(main_chart_path)
        img_main.drawWidth = 520
        img_main.drawHeight = 240

        elements.append(Paragraph("Main efficiency curve", heading_style))
        elements.append(img_main)
        elements.append(Spacer(1, 15))

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
            item["fig_evolution"].write_image(
                tmpfile.name,
                width=900,
                height=430,
                scale=2
            )
            evolution_chart_path = tmpfile.name

        img_evolution = Image(evolution_chart_path)
        img_evolution.drawWidth = 520
        img_evolution.drawHeight = 240

        elements.append(Paragraph("Efficiency evolution by round trip", heading_style))
        elements.append(img_evolution)

    doc.build(elements)

    return pdf_buffer.getvalue()


# ============================================================
# APPLICATION
# ============================================================

if uploaded_files:

    results = []
    files_data = []

    for file in uploaded_files:

        df, metadata = load_file(file)

        if df is None:
            st.error(f"Unable to read file: {file.name}")
            continue

        average_upward, average_downward, evolution_df = calculate_stats(df)

        fig = px.line(
            df,
            x="Time",
            y=["Upward", "Downward"],
            labels={
                "Time": "Time (s)",
                "value": "Efficiency (%)",
                "variable": "Direction"
            },
            title=file.name,
            color_discrete_map={
                "Upward": UPWARD_COLOR,
                "Downward": DOWNWARD_COLOR
            }
        )

        fig_evolution = px.line(
            evolution_df,
            x="Pass",
            y=[
                "Average upward efficiency (%)",
                "Average downward efficiency (%)"
            ],
            markers=True,
            title="Efficiency evolution by round trip",
            color_discrete_map={
                "Average upward efficiency (%)": UPWARD_COLOR,
                "Average downward efficiency (%)": DOWNWARD_COLOR
            }
        )

        result_row = {
            "File": file.name,
            "Average upward without first pass (%)": round(average_upward, 2),
            "Average downward without first pass (%)": round(average_downward, 2)
        }

        for _, row in evolution_df.iterrows():
            pass_number = int(row["Pass"])

            result_row[f"Upward pass {pass_number} (%)"] = round(
                row["Average upward efficiency (%)"], 2
            )

            result_row[f"Downward pass {pass_number} (%)"] = round(
                row["Average downward efficiency (%)"], 2
            )

        results.append(result_row)

        files_data.append({
            "name": file.name,
            "metadata": metadata,
            "df": df,
            "fig": fig,
            "fig_evolution": fig_evolution,
            "average_upward": average_upward,
            "average_downward": average_downward,
            "evolution_df": evolution_df
        })

    results_df = pd.DataFrame(results)

    file_names = [item["name"] for item in files_data]

    if "selected_file" not in st.session_state:
        st.session_state.selected_file = file_names[0]

    if st.session_state.selected_file not in file_names:
        st.session_state.selected_file = file_names[0]

    with st.sidebar:

        st.markdown("### Choose your file")

        for file_name in file_names:
            if st.button(file_name, key=f"button_{file_name}"):
                st.session_state.selected_file = file_name

    selected_file = st.session_state.selected_file

    selected_data = next(
        item for item in files_data
        if item["name"] == selected_file
    )

    # ========================================================
    # AFFICHAGE PRINCIPAL
    # ========================================================

    left_col, right_col = st.columns([0.8, 2.2])

    with left_col:

        st.subheader("Test information")

        for line in selected_data["metadata"]:
            st.write(line)

        st.subheader("Global results")

        st.markdown(
            f"""
            <div style="
                background:{UPWARD_COLOR};
                padding:18px;
                border-radius:14px;
                margin-bottom:12px;
                text-align:center;
            ">
                <p style="margin:0;">Average upward efficiency</p>
                <h1 style="margin:0;">{selected_data['average_upward']:.2f} %</h1>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div style="
                background:{DOWNWARD_COLOR};
                padding:18px;
                border-radius:14px;
                margin-bottom:18px;
                text-align:center;
            ">
                <p style="margin:0;">Average downward efficiency</p>
                <h1 style="margin:0;">{selected_data['average_downward']:.2f} %</h1>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.subheader("Efficiency evolution")

        st.plotly_chart(
            selected_data["fig_evolution"],
            use_container_width=True,
            key=f"small_evolution_{selected_file}"
        )

    with right_col:

        st.subheader("Main efficiency curve")

        st.plotly_chart(
            selected_data["fig"],
            use_container_width=True,
            key=f"main_curve_{selected_file}"
        )

        st.subheader("Average efficiency by pass")

        for _, row in selected_data["evolution_df"].iterrows():

            pass_number = int(row["Pass"])
            upward_value = row["Average upward efficiency (%)"]
            downward_value = row["Average downward efficiency (%)"]

            st.write(f"### Pass {pass_number}")

            col_up, col_down = st.columns(2)

            with col_up:
                st.metric(
                    label="🔴 Upward",
                    value=f"{upward_value:.2f} %"
                )

            with col_down:
                if pd.isna(downward_value):
                    st.metric(
                        label="🔵 Downward",
                        value="No value"
                    )
                else:
                    st.metric(
                        label="🔵 Downward",
                        value=f"{downward_value:.2f} %"
                    )

    # ========================================================
    # TABLEAU DE SYNTHÈSE
    # ========================================================

    st.subheader("Complete summary table")

    upward_columns = [
        col for col in results_df.columns
        if "upward" in col.lower()
    ]

    downward_columns = [
        col for col in results_df.columns
        if "downward" in col.lower()
    ]

    styled_results_df = results_df.style.applymap(
        lambda v: f"color: {UPWARD_COLOR};",
        subset=upward_columns
    ).applymap(
        lambda v: f"color: {DOWNWARD_COLOR};",
        subset=downward_columns
    )

    st.dataframe(
        styled_results_df,
        use_container_width=True,
        hide_index=True
    )

    # ========================================================
    # EXPORT EXCEL
    # ========================================================

    excel_buffer = BytesIO()

    with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
        results_df.to_excel(
            writer,
            index=False,
            sheet_name="Results"
        )

    st.download_button(
        label="📥 Download Excel summary",
        data=excel_buffer.getvalue(),
        file_name="efficiency_results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    # ========================================================
    # EXPORT PDF
    # ========================================================

    pdf_data = build_pdf(results_df, files_data)

    st.download_button(
        label="📄 Download full PDF report",
        data=pdf_data,
        file_name="efficiency_analysis_report.pdf",
        mime="application/pdf"
    )

else:
    st.info("Upload one or more CSV files to start the analysis.")