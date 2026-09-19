# ============================================================
# RETENTIA
# Employee Attrition Analytics Dashboard
#
# This version is designed for HR analytics and decision support.
# It avoids ranking individual employees for management action.
# Instead, it provides:
#   - Data cleaning
#   - Decision-tree attrition model
#   - Model performance
#   - Feature importance
#   - Aggregate risk patterns
#   - Plain-English findings
#   - Management recommendations
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
# SIMPLE PRODUCT STYLING
# ------------------------------------------------------------

st.markdown(
    """
    <style>
        .main-title {
            font-size: 42px;
            font-weight: 700;
            margin-bottom: 5px;
        }

        .subtitle {
            font-size: 18px;
            color: #666;
            margin-bottom: 25px;
        }

        .section-title {
            font-size: 25px;
            font-weight: 650;
            margin-top: 30px;
        }

        .info-box {
            padding: 18px;
            border-radius: 10px;
            background-color: #f5f7fb;
            margin-bottom: 20px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------------------------------------------------
# HEADER
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

st.info(
    "Retentia is designed as an HR analytics and decision-support tool. "
    "Predictions are presented at an aggregate level and should not be "
    "used to make employment decisions about individual employees."
)

# ------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------

st.sidebar.header("Upload employee data")

uploaded_file = st.sidebar.file_uploader(
    "Upload a CSV file",
    type=["csv"]
)

st.sidebar.markdown("---")

st.sidebar.caption(
    "For testing, you can generate a synthetic dataset using "
    "`generate_data.py`."
)

# ------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------

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

        if df[column].dtype == "object":

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

    if target.dtype == "object":

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

        if df[column].dtype == "object":

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
# MAIN APPLICATION
# ------------------------------------------------------------

if uploaded_file is None:

    st.markdown(
        '<div class="info-box">'
        '<strong>Welcome to Retentia.</strong><br><br>'
        'Upload an employee CSV from the sidebar to begin analyzing '
        'attrition patterns.'
        '</div>',
        unsafe_allow_html=True
    )

    st.subheader("What Retentia does")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("1", "Clean data")

    with col2:
        st.metric("2", "Train model")

    with col3:
        st.metric("3", "Understand patterns")

    st.markdown("---")

    st.subheader("Expected data")

    st.write(
        "Your CSV should contain an attrition/left column and employee "
        "attributes such as department, salary, tenure, overtime, "
        "performance, promotion history, attendance, or similar fields."
    )

    st.stop()


# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

try:

    raw_data = pd.read_csv(uploaded_file)

except Exception as error:

    st.error(
        f"Could not read the CSV file: {error}"
    )

    st.stop()


# ------------------------------------------------------------
# CLEAN DATA
# ------------------------------------------------------------

(
    data,
    original_rows,
    original_columns,
    duplicate_count,
    empty_columns
) = clean_dataset(raw_data)


# ------------------------------------------------------------
# FIND TARGET
# ------------------------------------------------------------

target_column, target = prepare_target(data)

if target_column is None:

    st.error(
        "I could not find an attrition target column. "
        "Please include a column such as Attrition, Left, "
        "EmployeeAttrition, Exited, or Turnover."
    )

    st.stop()


data[target_column] = target

# Remove rows where target cannot be determined
before_target_cleanup = len(data)

data = data.dropna(
    subset=[target_column]
)

target_rows_removed = (
    before_target_cleanup - len(data)
)


# Make sure target is binary
unique_target_values = set(
    data[target_column].unique()
)

if not unique_target_values.issubset({0, 1}):

    st.error(
        "The attrition column must represent two classes, "
        "such as Yes/No or 1/0."
    )

    st.stop()


# ------------------------------------------------------------
# DATA SUMMARY
# ------------------------------------------------------------

st.markdown(
    '<div class="section-title">Data overview</div>',
    unsafe_allow_html=True
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Employees",
        len(data)
    )

with col2:
    st.metric(
        "Features",
        len(data.columns) - 1
    )

with col3:
    attrition_rate = data[target_column].mean() * 100

    st.metric(
        "Observed attrition",
        f"{attrition_rate:.1f}%"
    )

with col4:
    st.metric(
        "Duplicate rows removed",
        duplicate_count
    )


# ------------------------------------------------------------
# CLEANING SUMMARY
# ------------------------------------------------------------

with st.expander(
    "View cleaning summary"
):

    st.write(
        f"Original dataset: **{original_rows} rows × "
        f"{original_columns} columns**."
    )

    st.write(
        f"Final dataset: **{len(data)} rows × "
        f"{len(data.columns)} columns**."
    )

    st.write(
        f"Duplicate rows removed: **{duplicate_count}**."
    )

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

        st.write(
            "No completely empty columns were found."
        )

    st.write(
        "Missing values in predictor columns are handled "
        "automatically during model training using median "
        "imputation for numeric variables and most-frequent "
        "imputation for categorical variables."
    )


# ------------------------------------------------------------
# PREPARE MODEL DATA
# ------------------------------------------------------------

X, dropped_columns = build_model_data(
    data,
    target_column
)

y = data[target_column]


if len(X) < 30:

    st.warning(
        "The dataset is quite small. Model metrics may be unstable. "
        "For a realistic test, use several hundred synthetic records "
        "or more."
    )


# ------------------------------------------------------------
# TRAIN / TEST SPLIT
# ------------------------------------------------------------

try:

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

except ValueError:

    st.error(
        "The dataset does not contain enough examples of both "
        "attrition classes for a stratified train/test split."
    )

    st.stop()


# ------------------------------------------------------------
# TRAIN MODEL
# ------------------------------------------------------------

pipeline = make_pipeline(X)

pipeline.fit(
    X_train,
    y_train
)


# ------------------------------------------------------------
# MODEL EVALUATION
# ------------------------------------------------------------

predictions = pipeline.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions
)

precision = precision_score(
    y_test,
    predictions,
    zero_division=0
)

recall = recall_score(
    y_test,
    predictions,
    zero_division=0
)


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
    st.metric(
        "Accuracy",
        f"{accuracy:.1%}"
    )

with col2:
    st.metric(
        "Precision",
        f"{precision:.1%}"
    )

with col3:
    st.metric(
        "Recall",
        f"{recall:.1%}"
    )


# ------------------------------------------------------------
# FEATURE IMPORTANCE
# ------------------------------------------------------------

st.markdown(
    '<div class="section-title">What drives the model?</div>',
    unsafe_allow_html=True
)

importance_df = get_feature_importance(
    pipeline
)

top_features = importance_df.head(10).copy()

top_features = top_features.sort_values(
    "Importance",
    ascending=True
)

st.bar_chart(
    top_features.set_index("Feature")["Importance"]
)

st.caption(
    "Feature importance indicates which variables the decision tree "
    "used most strongly. It does not prove that a factor causes "
    "employees to leave."
)


# ------------------------------------------------------------
# AGGREGATE ATTRITION ANALYSIS
# ------------------------------------------------------------

st.markdown(
    '<div class="section-title">Attrition patterns</div>',
    unsafe_allow_html=True
)


# Find useful grouping columns
department_column = find_column(
    data,
    [
        "Department",
        "BusinessUnit",
        "JobRole"
    ]
)

if department_column:

    department_summary = (
        data.groupby(department_column)[target_column]
        .agg(
            Employees="count",
            Attrition_Rate="mean"
        )
        .reset_index()
    )

    department_summary["Attrition_Rate"] *= 100

    department_summary = department_summary.sort_values(
        "Attrition_Rate",
        ascending=False
    )

    st.subheader(
        f"Attrition by {department_column}"
    )

    st.dataframe(
        department_summary.style.format({
            "Attrition_Rate": "{:.1f}%"
        }),
        use_container_width=True,
        hide_index=True
    )

else:

    st.write(
        "No department or comparable grouping column was found."
    )


# ------------------------------------------------------------
# NUMERIC ATTRITION PATTERNS
# ------------------------------------------------------------

numeric_columns = data.select_dtypes(
    include=["number"]
).columns.tolist()

numeric_columns = [
    column
    for column in numeric_columns
    if column != target_column
]


if numeric_columns:

    pattern_rows = []

    for column in numeric_columns:

        stayed_values = data.loc[
            data[target_column] == 0,
            column
        ].dropna()

        left_values = data.loc[
            data[target_column] == 1,
            column
        ].dropna()

        if len(stayed_values) > 0 and len(left_values) > 0:

            pattern_rows.append({
                "Feature": column,
                "Stayed average": stayed_values.mean(),
                "Left average": left_values.mean(),
                "Difference": (
                    left_values.mean()
                    - stayed_values.mean()
                )
            })

    if pattern_rows:

        pattern_df = pd.DataFrame(
            pattern_rows
        )

        pattern_df["Absolute Difference"] = (
            pattern_df["Difference"].abs()
        )

        pattern_df = (
            pattern_df
            .sort_values(
                "Absolute Difference",
                ascending=False
            )
            .drop(
                columns=["Absolute Difference"]
            )
            .head(10)
        )

        st.subheader(
            "Largest numeric differences between groups"
        )

        st.dataframe(
            pattern_df,
            use_container_width=True,
            hide_index=True
        )


# ------------------------------------------------------------
# OVERALL FINDINGS
# ------------------------------------------------------------

st.markdown(
    '<div class="section-title">Overall findings</div>',
    unsafe_allow_html=True
)


top_feature_names = (
    importance_df
    .head(3)["Feature"]
    .tolist()
)


feature_text = ", ".join(
    top_feature_names
)


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

st.write(
    findings_text
)


# ------------------------------------------------------------
# RECOMMENDATIONS
# ------------------------------------------------------------

st.subheader(
    "Recommended management actions"
)

recommendations = [
    "Review the strongest aggregate attrition patterns and investigate the underlying workplace causes.",
    "Use employee surveys, stay interviews, and manager feedback to validate the patterns identified by the model.",
    "Review workload, overtime, career-development opportunities, compensation, and promotion processes where the data indicates potential retention concerns.",
    "Develop targeted retention initiatives at the team or organizational level rather than treating model predictions as conclusions about individual employees.",
    "Monitor attrition rates over time and retrain the model periodically as workforce conditions and organizational policies change."
]

for number, recommendation in enumerate(
    recommendations,
    start=1
):

    st.write(
        f"**{number}.** {recommendation}"
    )


# ------------------------------------------------------------
# DATA PREVIEW
# ------------------------------------------------------------

st.markdown(
    '<div class="section-title">Cleaned data preview</div>',
    unsafe_allow_html=True
)

st.dataframe(
    data.head(20),
    use_container_width=True
)


# ------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------

st.markdown("---")

st.caption(
    "Retentia • Employee attrition analytics • "
    "Synthetic or appropriately authorized data recommended for testing"
)
