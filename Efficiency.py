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

UPWARD_COLOR = "#FF0A05"
DOWNWARD_COLOR = "#40A0F5"
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
    width: 250px !important;
}}

section[data-testid="stSidebar"] > div {{
    width: 250px !important;
}}

</style>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## Diversification")
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

if "selected_file" in st.session_state:
    st.markdown(
        f"### Selected file : {st.session_state.selected_file}"
    )

st.markdown("<br><br>", unsafe_allow_html=True)

# ============================================================
# CHARGEMENT DU FICHIER
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
        "Upward_raw",
        "Downward_raw",
        "Upward_filtered",
        "Downward_filtered",
        "Upward_stats",
        "Downward_stats"
    ]

    for col in ["Time", "Upward_raw", "Downward_raw"]:
        df[col] = df[col].astype(str).str.replace(",", ".", regex=False)
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Time"]).reset_index(drop=True)

    # Courbes utilisées pour l'analyse :
    # on garde seulement les rendements compris entre 0 et 1
    df["Upward_curve"] = df["Upward_raw"].where(
        (df["Upward_raw"] > 0) &
        (df["Upward_raw"] < 1)
    )

    df["Downward_curve"] = df["Downward_raw"].where(
        (df["Downward_raw"] > 0) &
        (df["Downward_raw"] < 1)
    )

    return df, info_lines


# ============================================================
# MOYENNE NETTOYÉE
# ============================================================

def clean_mean(series):

    series = series.dropna()
    series = series[(series > 0) & (series < 1)]

    if series.empty:
        return None

    q_low = series.quantile(0.05)
    q_high = series.quantile(0.95)

    series_clean = series[
        (series >= q_low) &
        (series <= q_high)
    ]

    if series_clean.empty:
        return None

    return series_clean.mean() * 100


# ============================================================
# DÉTECTION DES PASS
# ============================================================

def add_pass_numbers(df):

    df = df.copy()

    # Direction active selon la courbe valide
    df["Direction"] = None

    df.loc[df["Upward_curve"].notna(), "Direction"] = "Upward"
    df.loc[df["Downward_curve"].notna(), "Direction"] = "Downward"

    # Si aucune direction n'est valide, on laisse vide
    valid_direction = df["Direction"].notna()

    # Détection des segments successifs
    df["Segment"] = None

    df.loc[valid_direction, "Segment"] = (
        df.loc[valid_direction, "Direction"]
        != df.loc[valid_direction, "Direction"].shift()
    ).cumsum()

    # Suppression des segments trop courts
    # Ces segments sont souvent des pics au changement de direction
    min_points = 20

    segment_sizes = df[df["Segment"].notna()].groupby("Segment").size()
    valid_segments = segment_sizes[segment_sizes >= min_points].index

    invalid_segments = ~df["Segment"].isin(valid_segments)

    df.loc[invalid_segments, "Direction"] = None
    df.loc[invalid_segments, "Upward_curve"] = None
    df.loc[invalid_segments, "Downward_curve"] = None

    # Recalcul après suppression des petits segments
    valid_direction = df["Direction"].notna()

    df["Clean_segment"] = None

    df.loc[valid_direction, "Clean_segment"] = (
        df.loc[valid_direction, "Direction"]
        != df.loc[valid_direction, "Direction"].shift()
    ).cumsum()

    df["Upward_pass"] = None
    df["Downward_pass"] = None

    upward_count = 0
    downward_count = 0

    for segment in df["Clean_segment"].dropna().unique():

        segment_df = df[df["Clean_segment"] == segment]
        direction = segment_df["Direction"].iloc[0]

        if direction == "Upward":
            upward_count += 1
            df.loc[df["Clean_segment"] == segment, "Upward_pass"] = upward_count

        elif direction == "Downward":
            downward_count += 1
            df.loc[df["Clean_segment"] == segment, "Downward_pass"] = downward_count

    return df


# ============================================================
# CALCUL DES RÉSULTATS
# ============================================================

def calculate_stats(df):

    df = add_pass_numbers(df)

    # Moyennes générales sans la première pass
    df_up_global = df[df["Upward_pass"] >= 2]
    df_down_global = df[df["Downward_pass"] >= 2]

    average_upward = clean_mean(df_up_global["Upward_curve"])
    average_downward = clean_mean(df_down_global["Downward_curve"])

    # Moyenne de chaque montée, y compris la première
    upward_by_pass = (
        df[df["Upward_pass"].notna()]
        .groupby("Upward_pass")["Upward_curve"]
        .apply(clean_mean)
        .reset_index()
    )

    # Moyenne de chaque descente, y compris la première
    downward_by_pass = (
        df[df["Downward_pass"].notna()]
        .groupby("Downward_pass")["Downward_curve"]
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

    return average_upward, average_downward, evolution_df, df


# ============================================================
# PDF
# ============================================================

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
        fontSize=18,
        leading=22,
        alignment=1
    )

    heading_style = ParagraphStyle(
        "HeadingStyle",
        parent=styles["Heading2"],
        textColor=colors.HexColor("#1f77b4"),
        fontSize=12,
        leading=15
    )

    elements = []

    # ========================================================
    # UNE PAGE PAR FICHIER
    # ========================================================

    for index, item in enumerate(files_data):

        if index > 0:
            elements.append(PageBreak())

        elements.append(
            Paragraph(
                f"Efficiency Analysis : {item['name']}",
                title_style
            )
        )

        elements.append(Spacer(1, 10))

        # ---------------- COURBE PRINCIPALE ----------------

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
            item["fig"].write_image(
                tmpfile.name,
                width=900,
                height=420,
                scale=2
            )
            main_chart_path = tmpfile.name

        img_main = Image(main_chart_path)
        img_main.drawWidth = 520
        img_main.drawHeight = 230

        elements.append(Paragraph("Main efficiency curve", heading_style))
        elements.append(img_main)
        elements.append(Spacer(1, 8))

        # ---------------- MOYENNES GENERALES ----------------

        global_table_data = [
            ["Indicator", "Value"],
            ["Average upward efficiency", f"{item['average_upward']:.1f} %"],
            ["Average downward efficiency", f"{item['average_downward']:.1f} %"]
        ]

        global_table = Table(
            global_table_data,
            colWidths=[260, 120]
        )

        global_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f77b4")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d0d7de")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("ALIGN", (1, 1), (-1, -1), "CENTER"),

            ("TEXTCOLOR", (1, 1), (1, 1), colors.HexColor(UPWARD_COLOR)),
            ("TEXTCOLOR", (1, 2), (1, 2), colors.HexColor(DOWNWARD_COLOR)),
            ("FONTNAME", (1, 1), (1, 2), "Helvetica-Bold"),
        ]))

        elements.append(global_table)
        elements.append(Spacer(1, 12))

        # ---------------- COURBE EVOLUTION PASS ----------------

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
            item["fig_evolution"].write_image(
                tmpfile.name,
                width=900,
                height=380,
                scale=2
            )
            evolution_chart_path = tmpfile.name

        img_evolution = Image(evolution_chart_path)
        img_evolution.drawWidth = 520
        img_evolution.drawHeight = 210

        elements.append(Paragraph("Evolution by pass", heading_style))
        elements.append(img_evolution)
        elements.append(Spacer(1, 8))

        # ---------------- TABLEAU DES PASS ----------------

        pass_table_data = [
            ["Pass", "Upward (%)", "Downward (%)"]
        ]

        for _, row in item["evolution_df"].iterrows():

            pass_number = int(row["Pass"])

            upward_value = row["Average upward efficiency (%)"]
            downward_value = row["Average downward efficiency (%)"]

            pass_table_data.append([
                f"Pass {pass_number}",
                "-" if pd.isna(upward_value) else f"{upward_value:.1f}",
                "-" if pd.isna(downward_value) else f"{downward_value:.1f}"
            ])

        pass_table = Table(
            pass_table_data,
            colWidths=[120, 160, 160]
        )

        pass_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f77b4")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d0d7de")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),

            ("TEXTCOLOR", (1, 1), (1, -1), colors.HexColor(UPWARD_COLOR)),
            ("TEXTCOLOR", (2, 1), (2, -1), colors.HexColor(DOWNWARD_COLOR)),
            ("FONTNAME", (1, 1), (2, -1), "Helvetica-Bold"),
        ]))

        elements.append(pass_table)

    # ========================================================
    # DERNIERE PAGE : TABLEAU GENERAL
    # ========================================================

    elements.append(PageBreak())

    elements.append(
        Paragraph(
            "General Summary Table",
            title_style
        )
    )

    elements.append(Spacer(1, 14))

    summary_df = results_df.copy().round(1)

    table_data = [list(summary_df.columns)]

    for _, row in summary_df.iterrows():
        table_data.append([
            row[col] for col in summary_df.columns
        ])

    col_width = 520 / len(summary_df.columns)

    summary_table = Table(
        table_data,
        colWidths=[col_width] * len(summary_df.columns),
        repeatRows=1
    )

    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f77b4")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f5f8fb")),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d0d7de")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 6),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))

    elements.append(summary_table)

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

        average_upward, average_downward, evolution_df, df = calculate_stats(df)

        # Graphique principal : uniquement les courbes coupées à 1
        fig = px.line(
            df,
            x="Time",
            y=["Upward_curve", "Downward_curve"],
            labels={
                "Time": "Time (s)",
                "value": "Efficiency",
                "variable": ""
            },
            title=file.name,
            color_discrete_map={
                "Upward_curve": UPWARD_COLOR,
                "Downward_curve": DOWNWARD_COLOR
            }
        )

        fig.update_yaxes(range=[0.4, 1])

        fig.update_layout(
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1,
                xanchor="center",
                x=0.5
            )
        )

        fig_evolution = px.line(
            evolution_df,
            x="Pass",
            y=[
                "Average upward efficiency (%)",
                "Average downward efficiency (%)"
            ],
            labels={
                "Pass": "Pass",
                "value": "Efficiency (%)",
                "variable": ""
            },
            markers=True,
            title="Evolution by round trip",
            color_discrete_map={
                "Average upward efficiency (%)": UPWARD_COLOR,
                "Average downward efficiency (%)": DOWNWARD_COLOR
            }
        )

        fig_evolution.update_layout(
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1,
                xanchor="center",
                x=0.5
            )
        )

        result_row = {
            "File": file.name,
            "Average upward without first pass (%)": round(average_upward, 2),
            "Average downward without first pass (%)": round(average_downward, 2)
        }

        for _, row in evolution_df.iterrows():
            pass_number = int(row["Pass"])

            result_row[f"Upward pass {pass_number} (%)"] = round(
                row["Average upward efficiency (%)"], 1
            )

            result_row[f"Downward pass {pass_number} (%)"] = round(
                row["Average downward efficiency (%)"], 1
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

    left_col, col_space, right_col = st.columns([1, 0.1, 3])

    with left_col:

        with st.container(border=True):
            st.subheader("Test information")
            st.write("")

            for line in selected_data["metadata"]:
                st.write(line)

        with st.container(border=True):
            st.subheader("Efficiency")
            
            st.write("")

            st.markdown(
                f"""
                <div style="color:{UPWARD_COLOR}; font-size:20px; font-weight:bold;">
                    <p style="margin:0;">Average upward efficiency</p>
                    <h1 style="margin:0;">{selected_data['average_upward']:.1f} %</h1>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.write("")

            st.markdown(
                f"""
                <div style="color:{DOWNWARD_COLOR}; font-size:20px; font-weight:bold;">
                    <p style="margin:0;">Average downward efficiency</p>
                    <h1 style="margin:0;">{selected_data['average_downward']:.1f} %</h1>
                </div>
                """,
                unsafe_allow_html=True
            )

        with st.container(border=True):
            st.subheader("Efficiency evolution")

            st.plotly_chart(
                selected_data["fig_evolution"],
                use_container_width=True,
                key=f"small_evolution_{selected_file}"
            )

    with right_col:

        with st.container(border=True):
            st.subheader("Main efficiency curve")

            st.plotly_chart(
                selected_data["fig"],
                use_container_width=True,
                key=f"main_curve_{selected_file}"
            )

        with st.container(border=True):
            st.subheader("Average efficiency by pass")

            st.write("")

            for _, row in selected_data["evolution_df"].iterrows():

                pass_number = int(row["Pass"])
                upward_value = row["Average upward efficiency (%)"]
                downward_value = row["Average downward efficiency (%)"]

                st.write(f"### Pass {pass_number}")

                col_up, col_down = st.columns(2)

                with col_up:
                    st.metric(
                        label="🔴 Upward",
                        value="No value" if pd.isna(upward_value) else f"{upward_value:.1f} %"
                    )

                with col_down:
                    st.metric(
                        label="🔵 Downward",
                        value="No value" if pd.isna(downward_value) else f"{downward_value:.1f} %"
                    )

                st.write("")

    st.markdown("<br><br>", unsafe_allow_html=True)

    st.subheader("Complete summary table")

    # ============================================================
    # FORMATAGE DES COLONNES
    # ============================================================

    results_df.columns = [

        col.replace(
            "Average upward without first pass (%)",
            "Avg Up"
        )

        .replace(
            "Average downward without first pass (%)",
            "Avg Down"
        )

        .replace(
            "Upward pass ",
            "Up "
        )

        .replace(
            "Downward pass ",
            "Down "
        )

        .replace("(%)", "")

        for col in results_df.columns
    ]

    # ============================================================
    # COULEURS
    # ============================================================

    upward_columns = [
        col for col in results_df.columns
        if "up" in col.lower()
    ]

    downward_columns = [
        col for col in results_df.columns
        if "down" in col.lower()
    ]

    styled_results_df = (
        results_df.style

        # 1 chiffre après la virgule max
        .format(
            lambda x: f"{x:.1f}" if isinstance(x, (int, float)) else x,
            na_rep="-"
        )

        .applymap(
            lambda v: f"color: {UPWARD_COLOR}; font-weight:bold;",
            subset=upward_columns
        )

        .applymap(
            lambda v: f"color: {DOWNWARD_COLOR}; font-weight:bold;",
            subset=downward_columns
        )
    )

    # ============================================================
    # AFFICHAGE
    # ============================================================

    st.dataframe(
        styled_results_df,
        use_container_width=True,
        hide_index=True
    )

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

    pdf_data = build_pdf(results_df, files_data)

    st.download_button(
        label="📄 Download full PDF report",
        data=pdf_data,
        file_name="efficiency_analysis_report.pdf",
        mime="application/pdf"
    )

else:

    st.info("Upload one or more CSV files from the sidebar to start the analysis.")

    st.markdown("## How to use this interface")

    st.write(
        """
        This interface allows you to evaluate the efficiency of ball screws.

        Before uploading your file, save your raw data file as:
        CSV UTF-8 (Comma delimited) (*.csv)

        Rename your document using the following format:
        BallScrewType_Grease.csv

        You can analyze several efficiency files at the same time and download
        the results afterwards in order to keep them.
        """
    )

    st.markdown("---")

    st.markdown("## Comment utiliser cette interface")

    st.write(
        """
        Cette interface vous permet d’évaluer le rendement des vis à billes.

        Avant le téléchargement, enregistrez votre fichier de données brutes au format :
        CSV UTF-8 (Comma delimited) (*.csv)

        Renommez ensuite votre document sous le format :
        TypeVAB_Graisse.csv

        Vous pouvez analyser plusieurs rendements en même temps puis télécharger
        les résultats à la fin de l’analyse afin de les conserver.
        """
    )