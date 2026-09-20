# ============================================================
# RETENTIA
# Employee Attrition Analytics Dashboard
#
# Multi-page version. Pages:
#   - Home            : welcome / what Retentia does
#   - Analyze         : upload data, cleaning, model, patterns, findings
#   - Employee Lookup : search a specific employee + full ranked table
#   - About           : what Retentia is, its limits, and who built it
#
# Data and results from the Analyze page are kept in st.session_state
# so they are still available when the user switches to the
# Employee Lookup page, without needing to re-upload.
#
# Visual theme lives in two places:
#   - .streamlit/config.toml sets the base dark theme and accent color
#     (this is what Streamlit uses for buttons, sliders, the sidebar
#     radio, etc.)
#   - The CSS block below adds custom touches config.toml can't do:
#     the gradient wordmark, fonts, section accent bars, and the
#     colored risk badges.
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

# ------------------------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------------------------

st.set_page_config(
    page_title="Retentia | Employee Attrition Analytics",
    page_icon="📊",
    layout="wide"
)

# ------------------------------------------------------------
# BRAND STYLING
# ------------------------------------------------------------

st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500;600&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        /* ---------- Wordmark ---------- */
        .brand-row {
            display: flex;
            align-items: baseline;
            gap: 14px;
            margin-bottom: 2px;
        }

        .main-title {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 44px;
            font-weight: 700;
            letter-spacing: -0.5px;
            background: linear-gradient(90deg, #F2F4F3 0%, #F2F4F3 55%, #34D399 100%);
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
            margin: 0;
        }

        .subtitle {
            font-size: 16px;
            color: #8A928F;
            margin-bottom: 28px;
            max-width: 620px;
        }

        /* ---------- Section headers ---------- */
        .section-title {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 22px;
            font-weight: 700;
            margin-top: 34px;
            margin-bottom: 6px;
            padding-left: 14px;
            border-left: 3px solid #10B981;
            color: #F2F4F3;
        }

        /* ---------- Info / welcome box ---------- */
        .info-box {
            padding: 20px 22px;
            border-radius: 10px;
            background-color: #131615;
            border: 1px solid #1F2422;
            margin-bottom: 22px;
            color: #F2F4F3;
        }

        /* ---------- Sidebar ---------- */
        section[data-testid="stSidebar"] {
            background-color: #0A0B0A;
            border-right: 1px solid #1F2422;
        }

        .sidebar-brand {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 24px;
            font-weight: 700;
            background: linear-gradient(90deg, #F2F4F3 40%, #34D399 100%);
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
            margin-bottom: 4px;
        }

        .sidebar-tagline {
            font-size: 12px;
            color: #6B726F;
            margin-bottom: 18px;
        }

        /* ---------- Risk badges ---------- */
        .risk-badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 13px;
            font-weight: 600;
            font-family: 'Inter', sans-serif;
        }

        .risk-high {
            background-color: rgba(239, 68, 68, 0.15);
            color: #F87171;
            border: 1px solid rgba(239, 68, 68, 0.35);
        }

        .risk-medium {
            background-color: rgba(245, 158, 11, 0.15);
            color: #FBBF24;
            border: 1px solid rgba(245, 158, 11, 0.35);
        }

        .risk-low {
            background-color: rgba(16, 185, 129, 0.15);
            color: #34D399;
            border: 1px solid rgba(16, 185, 129, 0.35);
        }
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------------------------------------------------
# HEADER (shown on every page)
# ------------------------------------------------------------

st.markdown(
    '<div class="main-title">Retentia</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Employee attrition analytics that helps organizations understand '
    'retention patterns and take evidence-based action.'
    '</div>',
    unsafe_allow_html=True
)

# ------------------------------------------------------------
# SESSION STATE
# ------------------------------------------------------------

if "analyzed" not in st.session_state:
    st.session_state.analyzed = False

# ------------------------------------------------------------
# SIDEBAR NAVIGATION
# ------------------------------------------------------------

st.sidebar.markdown(
    '<div class="sidebar-brand">Retentia</div>'
    '<div class="sidebar-tagline">Attrition analytics</div>',
    unsafe_allow_html=True
)

page = st.sidebar.radio(
    "Navigate",
    ["Home", "Analyze", "Employee Lookup", "About"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")

if st.session_state.analyzed:
    st.sidebar.success("Data loaded and analyzed.")
else:
    st.sidebar.caption(
        "No data analyzed yet. Go to the Analyze page to upload a CSV."
    )


def risk_badge_html(risk_level):
    """
    Returns an HTML span styled as a colored badge for a risk level,
    used wherever risk level is displayed so it's easy to scan
    (green = low, amber = medium, red = high).
    """

    css_class = {
        "High risk": "risk-high",
        "Medium risk": "risk-medium",
        "Low risk": "risk-low"
    }.get(risk_level, "risk-medium")

    return f'<span class="risk-badge {css_class}">{risk_level}</span>'

# ------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------

def is_text_column(series):
    """
    Detects text/string columns. Newer pandas versions can read CSV
    text columns in as a "string" dtype instead of the classic
    "object" dtype, so we check for both to avoid silently missing
    columns like Attrition, OverTime, etc.
    """
    return series.dtype == "object" or pd.api.types.is_string_dtype(series)


def find_column(df, possible_names):
    """
    Finds a column even when the uploaded dataset uses
    slightly different capitalization or spacing.
    """
    normalized = {
        str(col).strip().lower().replace(" ", "").replace("_", ""): col
        for col in df.columns
    }

    for name in possible_names:
        key = name.lower().replace(" ", "").replace("_", "")

        if key in normalized:
            return normalized[key]

    return None


def convert_binary_columns(df):
    """
    Converts common Yes/No and True/False columns into 1/0.
    """

    yes_values = {
        "yes": 1,
        "y": 1,
        "true": 1,
        "1": 1
    }

    no_values = {
        "no": 0,
        "n": 0,
        "false": 0,
        "0": 0
    }

    for column in df.columns:

        if is_text_column(df[column]):

            cleaned = (
                df[column]
                .astype(str)
                .str.strip()
                .str.lower()
            )

            unique_values = set(cleaned.dropna().unique())

            if unique_values and unique_values.issubset(
                set(yes_values.keys()) | set(no_values.keys())
            ):
                df[column] = cleaned.map(
                    {**yes_values, **no_values}
                )

    return df


def prepare_target(df):
    """
    Finds the attrition/left column and converts it to 0/1.
    """

    target_column = find_column(
        df,
        [
            "Attrition",
            "Left",
            "EmployeeAttrition",
            "Exited",
            "Turnover"
        ]
    )

    if target_column is None:
        return None, None

    target = df[target_column].copy()

    if is_text_column(target):

        target = (
            target
            .astype(str)
            .str.strip()
            .str.lower()
            .map({
                "yes": 1,
                "no": 0,
                "true": 1,
                "false": 0,
                "left": 1,
                "stayed": 0,
                "1": 1,
                "0": 0
            })
        )

    else:
        target = pd.to_numeric(target, errors="coerce")

    return target_column, target


def clean_dataset(raw_df):
    """
    Performs basic automated cleaning.
    """

    df = raw_df.copy()

    original_rows, original_columns = df.shape

    # Remove completely empty columns
    empty_columns = [
        col for col in df.columns
        if df[col].isna().all()
    ]

    df = df.drop(columns=empty_columns)

    # Remove duplicate rows
    duplicate_count = df.duplicated().sum()

    df = df.drop_duplicates()

    # Convert common binary fields
    df = convert_binary_columns(df)

    # Try to convert numeric-looking columns
    for column in df.columns:

        if is_text_column(df[column]):

            converted = pd.to_numeric(
                df[column],
                errors="coerce"
            )

            non_missing_original = df[column].notna().sum()

            if non_missing_original > 0:

                successful_conversion = (
                    converted.notna().sum()
                    / non_missing_original
                )

                if successful_conversion >= 0.85:
                    df[column] = converted

    return (
        df,
        original_rows,
        original_columns,
        duplicate_count,
        empty_columns
    )


def build_model_data(df, target_column):
    """
    Separates target from predictors and removes
    columns that should not be used as predictive features.
    """

    X = df.drop(columns=[target_column]).copy()

    # Common identifier columns are not useful for learning
    irrelevant_names = [
        "employeenumber",
        "employeecount",
        "standardhours",
        "over18",
        "id",
        "employeeid"
    ]

    columns_to_drop = []

    for column in X.columns:

        normalized = (
            str(column)
            .lower()
            .replace(" ", "")
            .replace("_", "")
        )

        if normalized in irrelevant_names:
            columns_to_drop.append(column)

    X = X.drop(columns=columns_to_drop, errors="ignore")

    return X, columns_to_drop


def make_pipeline(X):
    """
    Builds preprocessing + decision tree pipeline.
    """

    numeric_columns = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_columns = X.select_dtypes(
        exclude=["number"]
    ).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            )
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_columns
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_columns
            )
        ]
    )

    model = DecisionTreeClassifier(
        max_depth=5,
        min_samples_leaf=10,
        random_state=42,
        class_weight="balanced"
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    return pipeline


def get_feature_importance(pipeline):
    """
    Extracts feature importance after preprocessing.
    """

    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    feature_names = preprocessor.get_feature_names_out()

    importances = model.feature_importances_

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    })

    importance_df = importance_df.sort_values(
        "Importance",
        ascending=False
    )

    # Make feature names easier to read
    importance_df["Feature"] = (
        importance_df["Feature"]
        .str.replace(
            "numeric__",
            "",
            regex=False
        )
        .str.replace(
            "categorical__",
            "",
            regex=False
        )
    )

    return importance_df


# ------------------------------------------------------------
# PAGE: HOME
# ------------------------------------------------------------

if page == "Home":

    st.markdown(
        '<div class="info-box">'
        '<strong style="color:#34D399;">Turn "why are people leaving?" '
        'into an answer.</strong><br><br>'
        'Retentia analyzes your workforce data to show which employees '
        'are at risk of leaving, why, and what to do about it — before '
        'they hand in notice.'
        '</div>',
        unsafe_allow_html=True
    )

    st.subheader("What Retentia does")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("1", "Clean data")
        st.caption("Upload a CSV and Retentia automatically cleans it.")

    with col2:
        st.metric("2", "Train a model")
        st.caption("A decision-tree model learns what predicts attrition.")

    with col3:
        st.metric("3", "Understand patterns")
        st.caption("See what drives risk, company-wide and per employee.")

    st.markdown("---")

    st.subheader("Getting started")

    st.write(
        "Go to the **Analyze** page from the sidebar to upload your "
        "employee data. Once analyzed, you can explore individual "
        "employees on the **Employee Lookup** page at any time, "
        "without re-uploading."
    )

    st.subheader("Expected data")

    st.write(
        "Your CSV should contain an attrition/left column and employee "
        "attributes such as department, salary, tenure, overtime, "
        "performance, promotion history, attendance, or similar fields."
    )


# ------------------------------------------------------------
# PAGE: ANALYZE
# ------------------------------------------------------------

elif page == "Analyze":

    st.markdown(
        '<div class="section-title">Analyze employee data</div>',
        unsafe_allow_html=True
    )

    st.info(
        "Retentia is designed as an HR analytics and decision-support "
        "tool. Predictions are presented at an aggregate level and "
        "should not be used to make employment decisions about "
        "individual employees."
    )

    uploaded_file = st.file_uploader(
        "Upload a CSV file",
        type=["csv"]
    )

    st.caption(
        "For testing, you can generate a synthetic dataset using "
        "`generate_data.py`."
    )

    if uploaded_file is None:

        st.write(
            "Upload a CSV above to run the analysis. Results will also "
            "become available on the Employee Lookup page once this "
            "finishes."
        )

        st.stop()

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    try:
        raw_data = pd.read_csv(uploaded_file)
    except Exception as error:
        st.error(f"Could not read the CSV file: {error}")
        st.stop()

    # --------------------------------------------------------
    # CLEAN DATA
    # --------------------------------------------------------

    (
        data,
        original_rows,
        original_columns,
        duplicate_count,
        empty_columns
    ) = clean_dataset(raw_data)

    # --------------------------------------------------------
    # FIND TARGET
    # --------------------------------------------------------

    target_column, target = prepare_target(data)

    if target_column is None:
        st.error(
            "I could not find an attrition target column. "
            "Please include a column such as Attrition, Left, "
            "EmployeeAttrition, Exited, or Turnover."
        )
        st.stop()

    data[target_column] = target

    before_target_cleanup = len(data)
    data = data.dropna(subset=[target_column])
    target_rows_removed = before_target_cleanup - len(data)

    unique_target_values = set(data[target_column].unique())

    if not unique_target_values.issubset({0, 1}):
        st.error(
            "The attrition column must represent two classes, "
            "such as Yes/No or 1/0."
        )
        st.stop()

    # --------------------------------------------------------
    # DATA SUMMARY
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Data overview</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Employees", len(data))

    with col2:
        st.metric("Features", len(data.columns) - 1)

    with col3:
        attrition_rate = data[target_column].mean() * 100
        st.metric("Observed attrition", f"{attrition_rate:.1f}%")

    with col4:
        st.metric("Duplicate rows removed", duplicate_count)

    with st.expander("View cleaning summary"):

        st.write(
            f"Original dataset: **{original_rows} rows × "
            f"{original_columns} columns**."
        )
        st.write(
            f"Final dataset: **{len(data)} rows × "
            f"{len(data.columns)} columns**."
        )
        st.write(f"Duplicate rows removed: **{duplicate_count}**.")
        st.write(
            f"Rows with missing target values removed: "
            f"**{target_rows_removed}**."
        )

        if empty_columns:
            st.write(
                "Completely empty columns removed: "
                + ", ".join(map(str, empty_columns))
            )
        else:
            st.write("No completely empty columns were found.")

        st.write(
            "Missing values in predictor columns are handled "
            "automatically during model training using median "
            "imputation for numeric variables and most-frequent "
            "imputation for categorical variables."
        )

    # --------------------------------------------------------
    # PREPARE MODEL DATA
    # --------------------------------------------------------

    X, dropped_columns = build_model_data(data, target_column)
    y = data[target_column]

    if len(X) < 30:
        st.warning(
            "The dataset is quite small. Model metrics may be unstable. "
            "For a realistic test, use several hundred synthetic "
            "records or more."
        )

    # --------------------------------------------------------
    # TRAIN / TEST SPLIT
    # --------------------------------------------------------

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
    except ValueError:
        st.error(
            "The dataset does not contain enough examples of both "
            "attrition classes for a stratified train/test split."
        )
        st.stop()

    # --------------------------------------------------------
    # TRAIN MODEL
    # --------------------------------------------------------

    pipeline = make_pipeline(X)
    pipeline.fit(X_train, y_train)

    # --------------------------------------------------------
    # MODEL EVALUATION
    # --------------------------------------------------------

    predictions = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)

    st.markdown(
        '<div class="section-title">Model performance</div>',
        unsafe_allow_html=True
    )

    st.caption(
        "These metrics describe performance on the held-out test data. "
        "They should not be interpreted as proof that the model will "
        "perform identically on future employees."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Accuracy", f"{accuracy:.1%}")
    with col2:
        st.metric("Precision", f"{precision:.1%}")
    with col3:
        st.metric("Recall", f"{recall:.1%}")

    # --------------------------------------------------------
    # FEATURE IMPORTANCE
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">What drives the model?</div>',
        unsafe_allow_html=True
    )

    importance_df = get_feature_importance(pipeline)

    top_features = importance_df.head(10).copy()
    top_features = top_features.sort_values("Importance", ascending=True)

    st.bar_chart(top_features.set_index("Feature")["Importance"])

    st.caption(
        "Feature importance indicates which variables the decision tree "
        "used most strongly. It does not prove that a factor causes "
        "employees to leave."
    )

    # --------------------------------------------------------
    # AGGREGATE ATTRITION ANALYSIS
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Attrition patterns</div>',
        unsafe_allow_html=True
    )

    department_column = find_column(
        data, ["Department", "BusinessUnit", "JobRole"]
    )

    if department_column:

        department_summary = (
            data.groupby(department_column)[target_column]
            .agg(Employees="count", Attrition_Rate="mean")
            .reset_index()
        )
        department_summary["Attrition_Rate"] *= 100
        department_summary = department_summary.sort_values(
            "Attrition_Rate", ascending=False
        )

        st.subheader(f"Attrition by {department_column}")

        st.dataframe(
            department_summary.style.format({"Attrition_Rate": "{:.1f}%"}),
            use_container_width=True,
            hide_index=True
        )

    else:
        st.write("No department or comparable grouping column was found.")

    # --------------------------------------------------------
    # NUMERIC ATTRITION PATTERNS
    # --------------------------------------------------------

    numeric_columns = data.select_dtypes(include=["number"]).columns.tolist()
    numeric_columns = [c for c in numeric_columns if c != target_column]

    pattern_df = pd.DataFrame(columns=["Feature", "Stayed average", "Left average", "Difference"])

    if numeric_columns:

        pattern_rows = []

        for column in numeric_columns:

            stayed_values = data.loc[data[target_column] == 0, column].dropna()
            left_values = data.loc[data[target_column] == 1, column].dropna()

            if len(stayed_values) > 0 and len(left_values) > 0:
                pattern_rows.append({
                    "Feature": column,
                    "Stayed average": stayed_values.mean(),
                    "Left average": left_values.mean(),
                    "Difference": left_values.mean() - stayed_values.mean()
                })

        if pattern_rows:

            pattern_df = pd.DataFrame(pattern_rows)
            pattern_df["Absolute Difference"] = pattern_df["Difference"].abs()
            pattern_df = (
                pattern_df
                .sort_values("Absolute Difference", ascending=False)
                .drop(columns=["Absolute Difference"])
                .head(10)
            )

            st.subheader("Largest numeric differences between groups")

            st.dataframe(
                pattern_df,
                use_container_width=True,
                hide_index=True
            )

    # --------------------------------------------------------
    # OVERALL FINDINGS
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Overall findings</div>',
        unsafe_allow_html=True
    )

    top_feature_names = importance_df.head(3)["Feature"].tolist()
    feature_text = ", ".join(top_feature_names)

    findings_text = f"""
The uploaded dataset contains **{len(data)} employees**, with an observed
attrition rate of **{attrition_rate:.1f}%**.

The decision-tree model achieved **{accuracy:.1%} accuracy**, with
**{precision:.1%} precision** and **{recall:.1%} recall** on the held-out
test set.

The variables that contributed most to the model's predictions were
**{feature_text}**.

These results describe patterns in this particular dataset. Feature
importance does not establish that any individual factor causes
attrition, and model predictions should be reviewed alongside
employee feedback, organizational context, and other evidence.
"""

    st.write(findings_text)

    # --------------------------------------------------------
    # RECOMMENDATIONS
    # --------------------------------------------------------

    st.subheader("Recommended management actions")

    recommendations = [
        "Review the strongest aggregate attrition patterns and investigate the underlying workplace causes.",
        "Use employee surveys, stay interviews, and manager feedback to validate the patterns identified by the model.",
        "Review workload, overtime, career-development opportunities, compensation, and promotion processes where the data indicates potential retention concerns.",
        "Develop targeted retention initiatives at the team or organizational level rather than treating model predictions as conclusions about individual employees.",
        "Monitor attrition rates over time and retrain the model periodically as workforce conditions and organizational policies change."
    ]

    for number, recommendation in enumerate(recommendations, start=1):
        st.write(f"**{number}.** {recommendation}")

    # --------------------------------------------------------
    # DATA PREVIEW
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Cleaned data preview</div>',
        unsafe_allow_html=True
    )

    st.dataframe(data.head(20), use_container_width=True)

    # --------------------------------------------------------
    # COMPUTE + SAVE EMPLOYEE-LEVEL RESULTS FOR THE LOOKUP PAGE
    # --------------------------------------------------------

    all_risk_scores = pipeline.predict_proba(X)[:, 1]
    results_df = data.copy()
    results_df["RiskScore"] = all_risk_scores * 100

    def risk_label(score):
        if score >= 66:
            return "High risk"
        elif score >= 33:
            return "Medium risk"
        else:
            return "Low risk"

    results_df["RiskLevel"] = results_df["RiskScore"].apply(risk_label)

    id_column = find_column(
        data, ["EmployeeID", "EmployeeNumber", "ID", "Employee Id"]
    )

    if id_column is None:
        results_df["RowNumber"] = range(1, len(results_df) + 1)
        id_column = "RowNumber"

    # Save everything the Employee Lookup page needs
    st.session_state.analyzed = True
    st.session_state.results_df = results_df
    st.session_state.pattern_df = pattern_df
    st.session_state.id_column = id_column
    st.session_state.department_column = department_column
    st.session_state.target_column = target_column
    st.session_state.data_columns = list(data.columns)

    st.success(
        "Analysis complete. Go to the Employee Lookup page to explore "
        "individual employees, or upload a new file above to re-analyze."
    )


# ------------------------------------------------------------
# PAGE: EMPLOYEE LOOKUP
# ------------------------------------------------------------

elif page == "Employee Lookup":

    st.markdown(
        '<div class="section-title">Employee risk scores</div>',
        unsafe_allow_html=True
    )

    if not st.session_state.analyzed:

        st.warning(
            "No analyzed data yet. Go to the Analyze page and upload a "
            "CSV first — results will then be available here."
        )

    else:

        results_df = st.session_state.results_df
        pattern_df = st.session_state.pattern_df
        id_column = st.session_state.id_column
        department_column = st.session_state.department_column
        target_column = st.session_state.target_column

        st.caption(
            "Risk scores are calculated for every employee in the "
            "uploaded dataset using the trained model. Scores reflect "
            "patterns found in this dataset and should be reviewed "
            "alongside other evidence, not used as the sole basis for "
            "decisions about individual employees."
        )

        st.subheader("Look up an employee")

        selected_id = st.selectbox(
            "Select an employee",
            options=results_df[id_column].astype(str).tolist()
        )

        employee_row = results_df[
            results_df[id_column].astype(str) == selected_id
        ].iloc[0]

        lookup_col1, lookup_col2 = st.columns(2)

        with lookup_col1:
            st.metric("Risk score", f"{employee_row['RiskScore']:.0f}%")

        with lookup_col2:
            st.write("")
            st.markdown(
                risk_badge_html(employee_row["RiskLevel"]),
                unsafe_allow_html=True
            )

        detail_columns = [
            column for column in st.session_state.data_columns
            if column not in [target_column, id_column]
        ][:10]

        st.write("**Employee details:**")

        details_table = pd.DataFrame({
            "Attribute": detail_columns,
            "Value": [employee_row[column] for column in detail_columns]
        })

        st.dataframe(details_table, use_container_width=True, hide_index=True)

        if not pattern_df.empty:

            st.write("**Top factors for this employee:**")

            top_pattern_features = pattern_df["Feature"].head(3).tolist()

            for feature in top_pattern_features:

                if feature in employee_row.index:

                    employee_value = employee_row[feature]
                    feature_pattern = pattern_df[
                        pattern_df["Feature"] == feature
                    ].iloc[0]

                    stayed_avg = feature_pattern["Stayed average"]
                    left_avg = feature_pattern["Left average"]

                    closer_to_left = (
                        abs(employee_value - left_avg)
                        < abs(employee_value - stayed_avg)
                    )

                    direction = (
                        "closer to the pattern seen in employees who left"
                        if closer_to_left
                        else "closer to the pattern seen in employees who stayed"
                    )

                    st.write(
                        f"- **{feature}**: this employee's value is "
                        f"**{employee_value:.1f}** ({direction}). "
                        f"Average for employees who left: {left_avg:.1f}, "
                        f"average for employees who stayed: {stayed_avg:.1f}."
                    )

        st.markdown("---")

        st.subheader("All employees ranked by risk")

        ranked_columns = [id_column, "RiskScore", "RiskLevel"]

        if department_column and department_column in results_df.columns:
            ranked_columns.insert(1, department_column)

        ranked_table = results_df[ranked_columns].sort_values(
            "RiskScore", ascending=False
        )

        def color_risk_level(value):
            colors = {
                "High risk": "color: #F87171; font-weight: 600;",
                "Medium risk": "color: #FBBF24; font-weight: 600;",
                "Low risk": "color: #34D399; font-weight: 600;"
            }
            return colors.get(value, "")

        st.dataframe(
            ranked_table.style
                .format({"RiskScore": "{:.0f}%"})
                .applymap(color_risk_level, subset=["RiskLevel"]),
            use_container_width=True,
            hide_index=True
        )


# ------------------------------------------------------------
# PAGE: ABOUT
# ------------------------------------------------------------

elif page == "About":

    st.markdown(
        '<div class="section-title">About Retentia</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Retentia is an employee attrition analytics tool. It helps "
        "organizations understand which patterns are associated with "
        "employees leaving, using their own workforce data, and turns "
        "those patterns into plain-English findings and recommended "
        "actions."
    )

    st.subheader("How it works")

    st.write(
        "Retentia cleans an uploaded employee dataset, trains a "
        "decision-tree model to distinguish employees who left from "
        "those who stayed, and reports which factors the model relied "
        "on most. It also estimates a risk score for every employee "
        "in the uploaded dataset."
    )

    st.subheader("Limitations")

    st.write(
        "Retentia is a decision-support tool, not a decision-making "
        "tool. Predictions reflect statistical patterns in the "
        "uploaded dataset and should always be reviewed alongside "
        "employee feedback, organizational context, and other "
        "evidence — especially before any action is taken regarding "
        "an individual employee."
    )

    st.markdown("---")

    st.caption(
        "Retentia • Employee attrition analytics • "
        "Synthetic or appropriately authorized data recommended for testing"
    )
