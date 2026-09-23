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
import time

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

from supabase import create_client, Client

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
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500;600&family=Caveat:wght@600&display=swap');

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

        /* ---------- Handwritten accent text ---------- */
        .handwritten-accent {
            font-family: 'Caveat', cursive;
            font-size: 30px;
            color: #34D399;
            line-height: 1.2;
            transform: rotate(-2deg);
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

        /* ---------- Page icon badge (top of each page) ---------- */
        .page-icon-badge {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 48px;
            height: 48px;
            border-radius: 12px;
            background: linear-gradient(135deg, #10B981, #0D9488);
            font-size: 24px;
            margin-bottom: 10px;
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

        /* ---------- Sidebar nav (radio, restyled as a nav list) ---------- */
        section[data-testid="stSidebar"] div[role="radiogroup"] {
            gap: 4px;
        }

        section[data-testid="stSidebar"] div[role="radiogroup"] label {
            padding: 10px 14px;
            border-radius: 10px;
            width: 100%;
            transition: background-color 0.15s ease;
        }

        section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
            background-color: #131615;
        }

        section[data-testid="stSidebar"] div[role="radiogroup"] label[data-checked="true"],
        section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
            background-color: rgba(16, 185, 129, 0.14);
            border: 1px solid rgba(16, 185, 129, 0.35);
        }

        /* ---------- Sidebar footer panel ---------- */
        .sidebar-footer {
            border-radius: 12px;
            background: linear-gradient(160deg, #131615 0%, #0D1F19 100%);
            border: 1px solid #1F2422;
            padding: 16px 16px;
            margin-top: 18px;
        }

        .sidebar-footer .line1 {
            color: #F2F4F3;
            font-size: 13px;
            font-weight: 600;
        }

        .sidebar-footer .line2 {
            color: #34D399;
            font-size: 13px;
            font-weight: 700;
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
# SUPABASE CLIENT (accounts + saved data)
# ------------------------------------------------------------
# Credentials come from Streamlit's private Secrets area, never from
# a file in the GitHub repository.

@st.cache_resource
def get_supabase_client():
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)


supabase: Client = get_supabase_client()

# ------------------------------------------------------------
# SESSION STATE
# ------------------------------------------------------------

if "user" not in st.session_state:
    st.session_state.user = None

if "analyzed" not in st.session_state:
    st.session_state.analyzed = False

if "preferences" not in st.session_state:
    st.session_state.preferences = {
        "company_name": "",
        "high_risk_threshold": 66,
        "medium_risk_threshold": 33
    }

if "splash_seen" not in st.session_state:
    st.session_state.splash_seen = False

# ------------------------------------------------------------
# SPLASH SCREEN
# ------------------------------------------------------------
# Shown once per browser session, before the login screen or
# anything else - just the logo, briefly, the way most apps open.
#
# This uses a base64-embedded copy of the logo inside a fixed,
# full-screen overlay - rather than st.image() inside st.columns -
# because that combination doesn't reliably vertically center in
# Streamlit. position: fixed takes it completely out of Streamlit's
# normal page flow, which guarantees true centering regardless of
# whatever else is on the page.

SPLASH_LOGO_B64 = "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KoYxgxtTYPJYkTosBwZ0cjCF5dNwo+CbIQZMPyJ4nkpkC+U66tjjjh79r2eeOuTpJ2qeYOZFRMSfCzCqqwWuukrXUhX2/OOpf+15yM7naFisG1vZsqVgxdAacHIOSob2agaQZZ3gUBIJICXh2U8IInqxsLDwSH8hi3kuBGWyfN06yivncawhb3ZLoo0vhWt/NJ+klE6XlnbffOihB9/3wAMPtIdlLq3FlClTYlT0AQMG8OrVq+m2225bzszLP88F3n///fL555+Pgdatt976Qpc1JiKcffbZ9rRp05QQohMg/vrXv7b8129m5rzfe9JFf/3b32wAOO+885x999338TlzFhe8N/f93ea88/ZqAO+N23PPj3764x/vO3z48HPHjxu3tV1QCCeXUwBJj6GoYUocmbc9Zu6HyM8p2DTooJTYRSHVJKmwYZPBefNXQggoV6FP7z58+qmnnvXYQw/97fP+MZVVl1l1mYwLYMw+t571h14H73pIR2NOkZsVwhIUKEcRiNhRyu5W1LdgzLZlHR8srUNlpUBtrUo+kkn8f5GwX77DMW3aNHvKlCnO66+++tBe++xzTK6jzSWQFVPnRmS7YAhkxwZXdbhrj34WVwY1UIoZSmsUFBXjtdded8sOnHBA35495w4cONCZM2dOsOL+OwVqgc9PtulqAPOzhj81wqIiZBfZljI+E/LfPB/5JSghBI477rjC2traAHT7Vf+q+uyJh0w8d9999x0EgDva2mBJQSHjMKCDUxxomLugxHfxPZkissYcFTPnPS8w9gveW4KwbbexocG64vLLT75l2rT7qqurrYwHNl3GqMrKVH1tba54aP8xoy87cma/w3frk93U7MLVFhMgOboWDUCnpeO6WXvhH5/+0Zo7X/krqsssJMOySXyOSFh23+HNhs+yG9qRzaXz1ijEHeOC1IhDyjNRZ/M4ypcPCmdhAxp1XENBKddyOzqWr169uu2II45QPoC4/095SBuP+//+dRXqMx6rjZU6/3cqD5z+3fPz76TUWova2tr2yspKycxSCLEuc1Xmmv3222+3u++665olS5bogqIicplZ6ShJjc07GdT2aFrXVA/vXFUjZkNFw9AcpHhmCGMYl4SAk8uJnr168f7lE34NQE6dOlV91ua0bGa1VV9bm9vh1PFjxv7xhBd6HrJrn+z6ZkcobREBQgfnrqMN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"

if not st.session_state.splash_seen:

    st.markdown(
        f"""
        <style>
            .splash-overlay {{
                position: fixed;
                top: 0;
                left: 0;
                width: 100vw;
                height: 100vh;
                background-color: #0A0B0A;
                display: flex;
                justify-content: center;
                align-items: center;
                z-index: 9999;
            }}

            .splash-overlay img {{
                width: 300px;
                max-width: 70vw;
            }}
        </style>

        <div class="splash-overlay">
            <img src="data:image/png;base64,{SPLASH_LOGO_B64}" />
        </div>
        """,
        unsafe_allow_html=True
    )

    time.sleep(1.6)
    st.session_state.splash_seen = True
    st.rerun()


# ------------------------------------------------------------
# ONBOARDING CAROUSEL
# ------------------------------------------------------------
# Shown once per browser session, after the splash screen and
# before login/sign up - a few swipeable-feeling slides introducing
# what Retentia does, with a Skip option, the way many apps open.

ONBOARDING_SLIDES = [
    {"image": "onboarding_1.png", "alt": "Smarter Insights. Stronger Teams."},
    {"image": "onboarding_2.png", "alt": "Data-Driven Better Decisions."},
    {"image": "onboarding_3.png", "alt": "Better People. Bigger Possibilities."},
]

if "onboarding_seen" not in st.session_state:
    st.session_state.onboarding_seen = False

if "onboarding_index" not in st.session_state:
    st.session_state.onboarding_index = 0

if st.session_state.splash_seen and not st.session_state.onboarding_seen:

    st.markdown(
        """
        <style>
            .onboarding-wrap {
                max-width: 380px;
                margin: 0 auto;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    left_pad, center_pad, right_pad = st.columns([1, 2, 1])

    with center_pad:

        st.markdown('<div class="onboarding-wrap">', unsafe_allow_html=True)

        current_slide = ONBOARDING_SLIDES[st.session_state.onboarding_index]
        st.image(current_slide["image"], use_container_width=True)

        # Simple dot indicator showing which slide this is
        dots = "".join(
            "● " if i == st.session_state.onboarding_index else "○ "
            for i in range(len(ONBOARDING_SLIDES))
        )
        st.markdown(
            f'<p style="text-align:center; color:#34D399; '
            f'letter-spacing:4px;">{dots}</p>',
            unsafe_allow_html=True
        )

        nav_col1, nav_col2 = st.columns(2)

        is_last_slide = (
            st.session_state.onboarding_index == len(ONBOARDING_SLIDES) - 1
        )

        with nav_col1:
            if st.button("Skip", use_container_width=True):
                st.session_state.onboarding_seen = True
                st.rerun()

        with nav_col2:
            button_label = "Get Started" if is_last_slide else "Next"
            if st.button(
                button_label, use_container_width=True, type="primary"
            ):
                if is_last_slide:
                    st.session_state.onboarding_seen = True
                else:
                    st.session_state.onboarding_index += 1
                st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

    st.stop()


def load_saved_results(user_id):
    """
    Looks up this user's most recently saved analysis in Supabase
    and, if one exists, loads it into session state so the Employee
    Lookup page works immediately without re-uploading a CSV.
    """

    try:
        response = (
            supabase.table("retentia_results")
            .select("*")
            .eq("user_id", user_id)
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )

        if response.data:

            row = response.data[0]

            st.session_state.results_df = pd.DataFrame(row["results_json"])
            st.session_state.pattern_df = pd.DataFrame(row["pattern_json"])
            st.session_state.id_column = row["id_column"]
            st.session_state.department_column = row["department_column"]
            st.session_state.target_column = row["target_column"]
            st.session_state.data_columns = row["data_columns"]
            st.session_state.accuracy = row.get("accuracy")
            st.session_state.precision = row.get("precision_score")
            st.session_state.recall = row.get("recall_score")
            importance_data = row.get("importance_json")
            st.session_state.importance_df = (
                pd.DataFrame(importance_data) if importance_data else None
            )
            st.session_state.analyzed = True

    except Exception:
        # No saved data yet, or the lookup failed. Not a critical
        # error - the user can just analyze data fresh instead.
        pass


def save_results_to_db(user_id):
    """
    Saves the current analysis results to Supabase, tied to this
    user's account, so they're still there next time they log in.
    """

    import json

    try:
        # Round-trip through pandas' own JSON conversion (rather than
        # .to_dict()) so every value is guaranteed to be a plain,
        # JSON-safe type before it's sent to the database - this
        # avoids errors from pandas/numpy number types that plain
        # .to_dict() can sometimes leave behind.
        results_json = json.loads(
            st.session_state.results_df.to_json(orient="records")
        )
        pattern_json = json.loads(
            st.session_state.pattern_df.to_json(orient="records")
        )

        importance_json = None
        if st.session_state.get("importance_df") is not None:
            importance_json = json.loads(
                st.session_state.importance_df.to_json(orient="records")
            )

        supabase.table("retentia_results").insert({
            "user_id": user_id,
            "results_json": results_json,
            "pattern_json": pattern_json,
            "id_column": st.session_state.id_column,
            "department_column": st.session_state.department_column,
            "target_column": st.session_state.target_column,
            "data_columns": st.session_state.data_columns,
            "accuracy": st.session_state.get("accuracy"),
            "precision_score": st.session_state.get("precision"),
            "recall_score": st.session_state.get("recall"),
            "importance_json": importance_json,
        }).execute()

    except Exception as error:
        st.warning(f"Could not save your results to your account: {error}")


def load_preferences(user_id):
    """
    Loads this user's saved preferences (company name, risk
    thresholds). Falls back to sensible defaults if none are saved
    yet - this is normal for a brand new account.
    """

    defaults = {
        "company_name": "",
        "high_risk_threshold": 66,
        "medium_risk_threshold": 33
    }

    try:
        response = (
            supabase.table("user_preferences")
            .select("*")
            .eq("user_id", user_id)
            .limit(1)
            .execute()
        )

        if response.data:
            row = response.data[0]
            st.session_state.preferences = {
                "company_name": row.get("company_name") or "",
                "high_risk_threshold": row.get("high_risk_threshold", 66),
                "medium_risk_threshold": row.get(
                    "medium_risk_threshold", 33
                )
            }
        else:
            st.session_state.preferences = defaults

    except Exception:
        st.session_state.preferences = defaults


def save_preferences(user_id, company_name, high_threshold, medium_threshold):
    """
    Saves (or updates) this user's preferences in Supabase using
    upsert - insert if it's their first time, update otherwise.
    """

    try:
        supabase.table("user_preferences").upsert({
            "user_id": user_id,
            "company_name": company_name,
            "high_risk_threshold": high_threshold,
            "medium_risk_threshold": medium_threshold
        }).execute()

        st.session_state.preferences = {
            "company_name": company_name,
            "high_risk_threshold": high_threshold,
            "medium_risk_threshold": medium_threshold
        }

        return True

    except Exception as error:
        st.warning(f"Could not save preferences: {error}")
        return False


def load_analysis_history(user_id):
    """
    Loads a lightweight summary of every past analysis for this
    user (not the full data - just enough to list them), most
    recent first.
    """

    try:
        response = (
            supabase.table("retentia_results")
            .select("id, created_at, results_json")
            .eq("user_id", user_id)
            .order("created_at", desc=True)
            .execute()
        )

        return response.data or []

    except Exception:
        return []


def load_specific_analysis(row_id):
    """
    Loads one specific past analysis (by its database row id) into
    session state, so the user can revisit an older upload instead
    of only ever seeing the latest one.
    """

    try:
        response = (
            supabase.table("retentia_results")
            .select("*")
            .eq("id", row_id)
            .limit(1)
            .execute()
        )

        if response.data:
            row = response.data[0]

            st.session_state.results_df = pd.DataFrame(row["results_json"])
            st.session_state.pattern_df = pd.DataFrame(row["pattern_json"])
            st.session_state.id_column = row["id_column"]
            st.session_state.department_column = row["department_column"]
            st.session_state.target_column = row["target_column"]
            st.session_state.data_columns = row["data_columns"]
            st.session_state.accuracy = row.get("accuracy")
            st.session_state.precision = row.get("precision_score")
            st.session_state.recall = row.get("recall_score")
            importance_data = row.get("importance_json")
            st.session_state.importance_df = (
                pd.DataFrame(importance_data) if importance_data else None
            )
            st.session_state.analyzed = True

            return True

        return False

    except Exception as error:
        st.warning(f"Could not load that analysis: {error}")
        return False


# ------------------------------------------------------------
# LOGIN / SIGN UP GATE
# ------------------------------------------------------------
# Nothing else in the app renders until the user is logged in.
# This screen is deliberately centered and card-styled, unlike the
# rest of the app, since it's the first thing anyone ever sees.

if st.session_state.user is None:

    st.markdown(
        """
        <style>
            .login-card {
                max-width: 420px;
                margin: 40px auto 0 auto;
                padding: 36px 34px 28px 34px;
                background-color: #131615;
                border: 1px solid #1F2422;
                border-radius: 14px;
            }

            .login-title {
                font-family: 'Space Grotesk', sans-serif;
                font-size: 30px;
                font-weight: 700;
                text-align: center;
                background: linear-gradient(90deg, #F2F4F3 40%, #34D399 100%);
                -webkit-background-clip: text;
                background-clip: text;
                color: transparent;
                margin-bottom: 4px;
            }

            .login-subtitle {
                text-align: center;
                font-size: 14px;
                color: #8A928F;
                margin-bottom: 26px;
            }

            div[data-testid="stForm"] {
                border: none;
                padding: 0;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    left_spacer, center_col, right_spacer = st.columns([1, 1.3, 1])

    with center_col:

        st.markdown('<div class="login-card">', unsafe_allow_html=True)

        logo_col1, logo_col2, logo_col3 = st.columns([1, 2, 1])
        with logo_col2:
            st.image("logo.png", use_container_width=True)

        st.markdown(
            '<div class="login-subtitle">'
            'Log in or create an account to continue'
            '</div>'
            '</div>',
            unsafe_allow_html=True
        )

        auth_mode = st.radio(
            "Choose an option",
            ["Log in", "Sign up"],
            horizontal=True,
            label_visibility="collapsed"
        )

        with st.form("auth_form"):

            email = st.text_input("Email", placeholder="you@company.com")
            password = st.text_input(
                "Password", type="password", placeholder="••••••••"
            )

            submitted = st.form_submit_button(
                auth_mode,
                use_container_width=True
            )

        if submitted:

            if auth_mode == "Log in":

                try:
                    res = supabase.auth.sign_in_with_password({
                        "email": email,
                        "password": password
                    })

                    st.session_state.user = res.user
                    load_saved_results(res.user.id)
                    load_preferences(res.user.id)
                    st.rerun()

                except Exception as error:
                    st.error(f"Could not log in: {error}")

            else:

                try:
                    supabase.auth.sign_up({
                        "email": email,
                        "password": password
                    })

                    st.success(
                        "Account created. Depending on your project's "
                        "settings, you may need to confirm your email "
                        "before logging in - check your inbox, then "
                        "switch to \"Log in\" above."
                    )

                except Exception as error:
                    st.error(f"Could not sign up: {error}")

        st.caption(
            "Your data is kept private to your account and is never "
            "visible to other users."
        )

    st.stop()


# ------------------------------------------------------------
# HEADER (shown on every page once logged in)
# ------------------------------------------------------------

st.image("logo.png", width=230)

st.markdown(
    '<div class="subtitle">'
    'Employee attrition analytics that helps organizations understand '
    'retention patterns and take evidence-based action.'
    '</div>',
    unsafe_allow_html=True
)

# ------------------------------------------------------------
# SIDEBAR NAVIGATION (only reached once logged in)
# ------------------------------------------------------------

st.sidebar.image("logo.png", width=140)
st.sidebar.markdown(
    '<div class="sidebar-tagline">Attrition analytics</div>',
    unsafe_allow_html=True
)

company_name = st.session_state.preferences.get("company_name", "")

if company_name:
    st.sidebar.caption(f"{company_name}")

st.sidebar.caption(f"Logged in as {st.session_state.user.email}")

if "nav_page" not in st.session_state:
    st.session_state.nav_page = "Home"

# Buttons elsewhere in the app (like "Get Started" or a feature card)
# can't set st.session_state.nav_page directly once this radio widget
# has been drawn - Streamlit blocks that. Instead, they set
# pending_nav, which gets applied here, BEFORE the widget below is
# instantiated, which is allowed.
if st.session_state.get("pending_nav"):
    st.session_state.nav_page = st.session_state.pending_nav
    st.session_state.pending_nav = None

NAV_ICONS = {
    "Home": "🏠",
    "Analyze": "📤",
    "Employee Lookup": "👥",
    "History": "🕐",
    "Settings": "⚙️",
    "About": "ℹ️",
}

page = st.sidebar.radio(
    "Navigate",
    ["Home", "Analyze", "Employee Lookup", "History", "Settings", "About"],
    label_visibility="collapsed",
    key="nav_page",
    format_func=lambda option: f"{NAV_ICONS.get(option, '')}  {option}"
)

st.sidebar.markdown("---")

if st.session_state.analyzed:
    st.sidebar.success("Data loaded and analyzed.")
else:
    st.sidebar.caption(
        "No data analyzed yet. Go to the Analyze page to upload a CSV."
    )

if st.sidebar.button("Log out"):
    st.session_state.user = None
    st.session_state.analyzed = False
    st.rerun()

st.sidebar.markdown(
    '<div class="sidebar-footer">'
    '<div class="line1">🍃 Better People.</div>'
    '<div class="line2">Bigger Possibilities.</div>'
    '</div>',
    unsafe_allow_html=True
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



def get_recommendation_details(feature_name):
    """
    Maps a data column name to a tailored recommendation, made up of
    a short "why it matters" explanation and 2 concrete next steps a
    manager can actually act on - not just a single generic sentence.
    Falls back to a generic-but-still-useful version for factors
    that don't match a known category.
    """

    name = str(feature_name).lower()

    if "promot" in name:
        return {
            "why": (
                "A lack of recent promotion is one of the strongest "
                "predictors of leaving in this data - employees who "
                "feel stuck often start looking elsewhere."
            ),
            "steps": [
                "Schedule a career conversation to discuss their "
                "growth path and a realistic promotion timeline.",
                "Compare their progression against peers with similar "
                "tenure to check if they've genuinely stalled."
            ]
        }

    if "training" in name:
        return {
            "why": (
                "Limited training or development activity can signal "
                "an employee who feels their growth has plateaued."
            ),
            "steps": [
                "Offer a specific training or development opportunity "
                "relevant to their role or next career step.",
                "Ask them directly what skills or growth areas they're "
                "interested in pursuing."
            ]
        }

    if "overtime" in name or "workload" in name or "hours" in name:
        return {
            "why": (
                "High workload or overtime is strongly linked to "
                "burnout, and burnt-out employees are far more likely "
                "to leave."
            ),
            "steps": [
                "Review their current workload with their manager and "
                "look for tasks that can be redistributed.",
                "Ask them directly whether their workload feels "
                "sustainable right now."
            ]
        }

    if any(word in name for word in ["salary", "income", "pay", "compensation"]):
        return {
            "why": (
                "Compensation that falls behind role or market "
                "expectations is a common, concrete reason employees "
                "leave for another offer."
            ),
            "steps": [
                "Benchmark their current pay against similar roles, "
                "both internally and in the market.",
                "If a raise isn't immediately possible, be transparent "
                "with them about the timeline and path to one."
            ]
        }

    if "satisfaction" in name:
        return {
            "why": (
                "Low reported satisfaction often reflects something "
                "specific and fixable - but only if it gets surfaced "
                "and addressed."
            ),
            "steps": [
                "Have a direct, informal conversation about what's "
                "going well and what isn't for them right now.",
                "Follow up on anything they raise within a set "
                "timeframe, so the conversation doesn't feel one-off."
            ]
        }

    if "worklife" in name or "work_life" in name or "balance" in name:
        return {
            "why": (
                "Poor work-life balance is a common, often "
                "under-discussed driver of attrition that rarely "
                "shows up until someone's already decided to leave."
            ),
            "steps": [
                "Check in on their current balance and any flexibility "
                "or scheduling needs they may have.",
                "Look at whether recent deadlines or projects have "
                "made this worse than usual."
            ]
        }

    if "tenure" in name or "years" in name:
        return {
            "why": (
                "Employees at this tenure stage, without a recent "
                "change in role or recognition, are statistically more "
                "likely to start considering other options."
            ),
            "steps": [
                "Schedule a dedicated check-in focused on their "
                "long-term path at the company, not just current work.",
                "Make sure their contributions to date have been "
                "clearly recognized."
            ]
        }

    if "attendance" in name:
        return {
            "why": (
                "A change in attendance patterns can be an early "
                "signal of disengagement or personal challenges, "
                "often before someone starts actively job-hunting."
            ),
            "steps": [
                "Check in personally rather than through a formal HR "
                "process, to understand what's behind the change.",
                "Watch whether the pattern is recent and worsening, or "
                "long-standing."
            ]
        }

    if "performance" in name:
        return {
            "why": (
                "Performance concerns can either reflect a genuine "
                "mismatch or unclear, unfair feedback - both increase "
                "the chance of someone leaving."
            ),
            "steps": [
                "Review recent performance feedback with their manager "
                "to confirm it's been clear and constructive.",
                "Ask the employee directly whether they feel their "
                "feedback has been fair and actionable."
            ]
        }

    if "environment" in name:
        return {
            "why": (
                "How someone experiences their day-to-day team and "
                "environment often matters as much as the work itself."
            ),
            "steps": [
                "Gather direct feedback from them about their team and "
                "working environment.",
                "Check whether this feeling is shared by others on "
                "their team, or specific to them."
            ]
        }

    return {
        "why": (
            f"This employee's {feature_name} stands out compared to "
            f"peers, and is one of the stronger factors behind their "
            f"risk score."
        ),
        "steps": [
            f"Review their {feature_name} more closely with their "
            f"manager to understand what's driving it.",
            "Use this as a starting point for a direct conversation, "
            "rather than a conclusion on its own."
        ]
    }

def format_time_ago(timestamp_str):
    """
    Converts a Supabase timestamp string into a short "time ago"
    label like "2h ago" or "3d ago", for the Recent Activity feed.
    Falls back to the raw string if it can't be parsed.
    """

    from datetime import datetime, timezone

    try:
        clean = timestamp_str.replace("Z", "+00:00")
        then = datetime.fromisoformat(clean)

        if then.tzinfo is None:
            then = then.replace(tzinfo=timezone.utc)

        now = datetime.now(timezone.utc)
        diff = now - then

        seconds = diff.total_seconds()

        if seconds < 60:
            return "just now"
        elif seconds < 3600:
            return f"{int(seconds // 60)}m ago"
        elif seconds < 86400:
            return f"{int(seconds // 3600)}h ago"
        else:
            return f"{int(seconds // 86400)}d ago"

    except Exception:
        return timestamp_str
# ------------------------------------------------------------
# PAGE: HOME
# ------------------------------------------------------------

if page == "Home":

    st.markdown(
        """
        <style>
            .hero-badge {
                display: inline-block;
                padding: 6px 16px;
                border-radius: 20px;
                background-color: rgba(16, 185, 129, 0.12);
                border: 1px solid rgba(16, 185, 129, 0.3);
                color: #34D399;
                font-size: 13px;
                font-weight: 600;
                margin-bottom: 18px;
            }

            .hero-heading {
                font-family: 'Space Grotesk', sans-serif;
                font-size: 44px;
                font-weight: 700;
                line-height: 1.15;
                color: #F2F4F3;
                margin-bottom: 14px;
            }

            .hero-heading .accent {
                background: linear-gradient(90deg, #34D399, #10B981);
                -webkit-background-clip: text;
                background-clip: text;
                color: transparent;
            }

            .hero-subtext {
                font-size: 15px;
                color: #8A928F;
                max-width: 520px;
                margin-bottom: 22px;
            }

            .feature-card {
                background-color: #131615;
                border: 1px solid #1F2422;
                border-radius: 12px;
                padding: 20px;
                height: 100%;
            }

            .feature-card h4 {
                font-family: 'Space Grotesk', sans-serif;
                font-size: 17px;
                margin: 10px 0 6px 0;
                color: #F2F4F3;
            }

            .feature-card p {
                font-size: 13px;
                color: #8A928F;
                margin-bottom: 0;
            }

            .stat-card {
                background-color: #131615;
                border: 1px solid #1F2422;
                border-radius: 12px;
                padding: 18px 20px;
            }

            .stat-number {
                font-family: 'Space Grotesk', sans-serif;
                font-size: 30px;
                font-weight: 700;
                color: #F2F4F3;
            }

            .stat-label {
                font-size: 13px;
                color: #8A928F;
                margin-bottom: 4px;
            }

            .activity-item {
                display: flex;
                justify-content: space-between;
                padding: 10px 0;
                border-bottom: 1px solid #1F2422;
                font-size: 13px;
            }

            .activity-item:last-child {
                border-bottom: none;
            }

            .activity-title {
                color: #F2F4F3;
                font-weight: 600;
            }

            .activity-sub {
                color: #8A928F;
                font-size: 12px;
            }

            .activity-time {
                color: #6B726F;
                font-size: 12px;
                white-space: nowrap;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # HERO
    # --------------------------------------------------------

    display_name = st.session_state.preferences.get("company_name", "")
    if not display_name:
        display_name = st.session_state.user.email.split("@")[0]

    hero_col, mascot_col = st.columns([3, 2])

    with hero_col:

        st.markdown(
            '<div class="hero-badge">Employee Attrition Early Warning '
            'System</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="hero-heading">Welcome back,<br>'
            f'<span class="accent">{display_name}!</span></div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="hero-subtext">Retentia helps you predict, '
            'understand, and reduce employee attrition — so you can '
            'build a healthier, more engaged workforce.</div>',
            unsafe_allow_html=True
        )

        if st.button("Get Started →", type="primary"):
            st.session_state.pending_nav = "Analyze"
            st.rerun()

    with mascot_col:
        st.image("mascot_laptop.png", use_container_width=True)
        st.markdown(
            '<div class="handwritten-accent" style="text-align:right;">'
            'Happy Teams.<br>Stronger Businesses.</div>',
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # --------------------------------------------------------
    # FEATURE CARDS
    # --------------------------------------------------------

    feature_cards = [
        ("Analyze", "Upload your employee data and let Retentia find attrition risks."),
        ("Employee Lookup", "Check individual attrition risk factors in seconds."),
        ("History", "View past analyses and track changes over time."),
        ("Settings", "Manage your account, preferences, and risk thresholds."),
        ("About", "Learn more about Retentia and how it works."),
    ]

    card_cols = st.columns(5)

    for i, (title, description) in enumerate(feature_cards):
        with card_cols[i]:
            st.markdown(
                f'<div class="feature-card"><h4>{title}</h4>'
                f'<p>{description}</p></div>',
                unsafe_allow_html=True
            )
            if st.button("Open →", key=f"home_nav_{title}", use_container_width=True):
                st.session_state.pending_nav = title
                st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # --------------------------------------------------------
    # WORKFORCE AT A GLANCE + RECENT ACTIVITY
    # --------------------------------------------------------

    glance_col, activity_col = st.columns([3, 2])

    with glance_col:

        st.markdown(
            '<div class="section-title" style="margin-top:0;">'
            'Your workforce at a glance</div>',
            unsafe_allow_html=True
        )

        if st.session_state.analyzed:

            results_df = st.session_state.results_df
            total_employees = len(results_df)
            at_risk = (results_df["RiskLevel"] == "High risk").sum()
            attrition_rate = (
                results_df[st.session_state.target_column].mean() * 100
            )
            retention_rate = 100 - attrition_rate

            stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)

            with stat_col1:
                st.markdown(
                    f'<div class="stat-card">'
                    f'<div class="stat-label">Total Employees</div>'
                    f'<div class="stat-number">{total_employees}</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

            with stat_col2:
                st.markdown(
                    f'<div class="stat-card">'
                    f'<div class="stat-label">At Risk of Leaving</div>'
                    f'<div class="stat-number">{at_risk}</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

            with stat_col3:
                st.markdown(
                    f'<div class="stat-card">'
                    f'<div class="stat-label">Retention Rate</div>'
                    f'<div class="stat-number">{retention_rate:.0f}%</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

            with stat_col4:
                st.markdown(
                    f'<div class="stat-card">'
                    f'<div class="stat-label">Attrition Rate</div>'
                    f'<div class="stat-number">{attrition_rate:.1f}%</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

        else:

            st.markdown(
                '<div class="feature-card">No data analyzed yet. '
                'Upload a CSV on the Analyze page to see your '
                'workforce stats here.</div>',
                unsafe_allow_html=True
            )

    with activity_col:

        st.markdown(
            '<div class="section-title" style="margin-top:0;">'
            'Recent activity</div>',
            unsafe_allow_html=True
        )

        history = load_analysis_history(st.session_state.user.id)

        if not history:

            st.markdown(
                '<div class="feature-card">No activity yet. Your '
                'analyses will show up here once you upload data.'
                '</div>',
                unsafe_allow_html=True
            )

        else:

            activity_html = '<div class="feature-card">'

            for entry in history[:4]:

                employee_count = len(entry.get("results_json") or [])
                time_ago = format_time_ago(entry.get("created_at", ""))

                activity_html += (
                    '<div class="activity-item">'
                    '<div>'
                    '<div class="activity-title">Analysis completed</div>'
                    f'<div class="activity-sub">{employee_count} '
                    'employees analyzed</div>'
                    '</div>'
                    f'<div class="activity-time">{time_ago}</div>'
                    '</div>'
                )

            activity_html += '</div>'

            st.markdown(activity_html, unsafe_allow_html=True)

    st.caption(
        "Note: the notification bell and account menu in the top "
        "corner of the reference design aren't wired up yet - this "
        "page focuses on real data from your account."
    )


# ------------------------------------------------------------
# PAGE: ANALYZE
# ------------------------------------------------------------

elif page == "Analyze":

    st.markdown('<div class="page-icon-badge">📤</div>', unsafe_allow_html=True)
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

    # ----------------------------------------------------------------
    # Decide what to show: a fresh upload, a previously saved analysis
    # (so navigating away and back doesn't lose everything), or
    # nothing yet.
    # ----------------------------------------------------------------

    show_cached = False

    if uploaded_file is None:

        if st.session_state.analyzed:
            show_cached = True
        else:
            st.write(
                "Upload a CSV above to run the analysis. Results will "
                "also become available on the Employee Lookup page "
                "once this finishes."
            )
            st.stop()

    cleaning_stats_available = False

    if uploaded_file is not None:

        # ------------------------------------------------------------
        # LOADING SCREEN (shown once per newly uploaded file)
        # ------------------------------------------------------------

        file_identifier = f"{uploaded_file.name}_{uploaded_file.size}"

        if st.session_state.get("last_loading_shown_for") != file_identifier:

            st.markdown(
                """
                <style>
                    @keyframes gentle-bob {
                        0%, 100% { transform: translateY(0px); }
                        50% { transform: translateY(-10px); }
                    }
                    .loading-mascot img {
                        animation: gentle-bob 2.2s ease-in-out infinite;
                    }
                </style>
                """,
                unsafe_allow_html=True
            )

            loading_left, loading_center, loading_right = st.columns(
                [1, 2, 1]
            )

            with loading_center:
                st.markdown(
                    '<div class="loading-mascot">', unsafe_allow_html=True
                )
                st.image("mascot_laptop.png", use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                st.markdown(
                    '<p style="text-align:center; color:#8A928F;">'
                    'Analyzing your data…</p>',
                    unsafe_allow_html=True
                )

            time.sleep(6)
            st.session_state.last_loading_shown_for = file_identifier
            st.rerun()

        # ------------------------------------------------------------
        # LOAD DATA
        # ------------------------------------------------------------

        try:
            raw_data = pd.read_csv(uploaded_file)
        except Exception as error:
            st.error(f"Could not read the CSV file: {error}")
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

        cleaning_stats_available = True

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

        # ------------------------------------------------------------
        # PREPARE MODEL DATA
        # ------------------------------------------------------------

        X, dropped_columns = build_model_data(data, target_column)
        y = data[target_column]

        if len(X) < 30:
            st.warning(
                "The dataset is quite small. Model metrics may be "
                "unstable. For a realistic test, use several hundred "
                "synthetic records or more."
            )

        # ------------------------------------------------------------
        # TRAIN / TEST SPLIT
        # ------------------------------------------------------------

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

        # ------------------------------------------------------------
        # TRAIN MODEL
        # ------------------------------------------------------------

        pipeline = make_pipeline(X)
        pipeline.fit(X_train, y_train)

        # ------------------------------------------------------------
        # MODEL EVALUATION
        # ------------------------------------------------------------

        predictions = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, zero_division=0)
        recall = recall_score(y_test, predictions, zero_division=0)

        # ------------------------------------------------------------
        # FEATURE IMPORTANCE
        # ------------------------------------------------------------

        importance_df = get_feature_importance(pipeline)

        # ------------------------------------------------------------
        # DEPARTMENT COLUMN
        # ------------------------------------------------------------

        department_column = find_column(
            data, ["Department", "BusinessUnit", "JobRole"]
        )

        # ------------------------------------------------------------
        # NUMERIC ATTRITION PATTERNS
        # ------------------------------------------------------------

        numeric_columns = data.select_dtypes(
            include=["number"]
        ).columns.tolist()
        numeric_columns = [c for c in numeric_columns if c != target_column]

        pattern_df = pd.DataFrame(
            columns=["Feature", "Stayed average", "Left average", "Difference"]
        )

        if numeric_columns:

            pattern_rows = []

            for column in numeric_columns:

                stayed_values = data.loc[
                    data[target_column] == 0, column
                ].dropna()
                left_values = data.loc[
                    data[target_column] == 1, column
                ].dropna()

                if len(stayed_values) > 0 and len(left_values) > 0:
                    pattern_rows.append({
                        "Feature": column,
                        "Stayed average": stayed_values.mean(),
                        "Left average": left_values.mean(),
                        "Difference": (
                            left_values.mean() - stayed_values.mean()
                        )
                    })

            if pattern_rows:

                pattern_df = pd.DataFrame(pattern_rows)
                pattern_df["Absolute Difference"] = (
                    pattern_df["Difference"].abs()
                )
                pattern_df = (
                    pattern_df
                    .sort_values("Absolute Difference", ascending=False)
                    .drop(columns=["Absolute Difference"])
                    .head(10)
                )

        attrition_rate = data[target_column].mean() * 100

    else:

        # ------------------------------------------------------------
        # SHOWING A CACHED (previously analyzed) RESULT
        # ------------------------------------------------------------

        st.info(
            "Showing your most recently analyzed data. Upload a new "
            "file above to run a new analysis."
        )

        results_df = st.session_state.results_df
        data = results_df.drop(columns=["RiskScore", "RiskLevel"])
        target_column = st.session_state.target_column
        department_column = st.session_state.department_column
        pattern_df = st.session_state.pattern_df
        accuracy = st.session_state.get("accuracy")
        precision = st.session_state.get("precision")
        recall = st.session_state.get("recall")
        importance_df = st.session_state.get("importance_df")
        attrition_rate = data[target_column].mean() * 100

    # ----------------------------------------------------------------
    # SHARED DISPLAY - works for both a fresh upload and a cached one
    # ----------------------------------------------------------------

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
        st.metric("Observed attrition", f"{attrition_rate:.1f}%")

    with col4:
        if cleaning_stats_available:
            st.metric("Duplicate rows removed", duplicate_count)
        else:
            st.metric("Duplicate rows removed", "—")

    if cleaning_stats_available:

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

    # ------------------------------------------------------------
    # MODEL PERFORMANCE
    # ------------------------------------------------------------

    st.markdown(
        '<div class="section-title">Model performance</div>',
        unsafe_allow_html=True
    )

    if accuracy is not None:

        st.caption(
            "These metrics describe performance on the held-out test "
            "data. They should not be interpreted as proof that the "
            "model will perform identically on future employees."
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Accuracy", f"{accuracy:.1%}")
        with col2:
            st.metric("Precision", f"{precision:.1%}")
        with col3:
            st.metric("Recall", f"{recall:.1%}")

    else:
        st.write(
            "Model performance metrics aren't available for this "
            "analysis - upload a file above to run a fresh analysis."
        )

    # ------------------------------------------------------------
    # FEATURE IMPORTANCE
    # ------------------------------------------------------------

    st.markdown(
        '<div class="section-title">What drives the model?</div>',
        unsafe_allow_html=True
    )

    if importance_df is not None and not importance_df.empty:

        top_features = importance_df.head(10).copy()
        top_features = top_features.sort_values("Importance", ascending=True)

        st.bar_chart(top_features.set_index("Feature")["Importance"])

        st.caption(
            "Feature importance indicates which variables the decision "
            "tree used most strongly. It does not prove that a factor "
            "causes employees to leave."
        )

    else:
        st.write("Feature importance isn't available for this analysis.")

    # ------------------------------------------------------------
    # AGGREGATE ATTRITION ANALYSIS
    # ------------------------------------------------------------

    st.markdown(
        '<div class="section-title">Attrition patterns</div>',
        unsafe_allow_html=True
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

    if pattern_df is not None and not pattern_df.empty:

        st.subheader("Largest numeric differences between groups")

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

    if importance_df is not None and not importance_df.empty and accuracy is not None:

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

    else:
        st.write(
            f"The uploaded dataset contains **{len(data)} employees**, "
            f"with an observed attrition rate of "
            f"**{attrition_rate:.1f}%**."
        )

    # ------------------------------------------------------------
    # RECOMMENDATIONS
    # ------------------------------------------------------------

    st.subheader("Recommended organization-wide actions")

    st.caption(
        "These are company-wide, systemic actions based on overall "
        "patterns. For specific actions about an individual employee, "
        "see the Employee Lookup page."
    )

    recommendations = [
        "Review the strongest aggregate attrition patterns and investigate the underlying workplace causes.",
        "Use employee surveys, stay interviews, and manager feedback to validate the patterns identified by the model.",
        "Review workload, overtime, career-development opportunities, compensation, and promotion processes where the data indicates potential retention concerns.",
        "Develop targeted retention initiatives at the team or organizational level rather than treating model predictions as conclusions about individual employees.",
        "Monitor attrition rates over time and retrain the model periodically as workforce conditions and organizational policies change."
    ]

    for number, recommendation in enumerate(recommendations, start=1):
        st.write(f"**{number}.** {recommendation}")

    # ------------------------------------------------------------
    # DATA PREVIEW
    # ------------------------------------------------------------

    st.markdown(
        '<div class="section-title">Cleaned data preview</div>',
        unsafe_allow_html=True
    )

    st.dataframe(data.head(20), use_container_width=True)

    # ------------------------------------------------------------
    # COMPUTE + SAVE EMPLOYEE-LEVEL RESULTS (fresh upload only)
    # ------------------------------------------------------------

    if uploaded_file is not None:

        all_risk_scores = pipeline.predict_proba(X)[:, 1]
        results_df = data.copy()
        results_df["RiskScore"] = all_risk_scores * 100

        high_threshold = st.session_state.preferences.get(
            "high_risk_threshold", 66
        )
        medium_threshold = st.session_state.preferences.get(
            "medium_risk_threshold", 33
        )

        def risk_label(score):
            if score >= high_threshold:
                return "High risk"
            elif score >= medium_threshold:
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

        # Save everything the Employee Lookup page - and this page,
        # on a future visit - needs.
        st.session_state.analyzed = True
        st.session_state.results_df = results_df
        st.session_state.pattern_df = pattern_df
        st.session_state.id_column = id_column
        st.session_state.department_column = department_column
        st.session_state.target_column = target_column
        st.session_state.data_columns = list(data.columns)
        st.session_state.accuracy = accuracy
        st.session_state.precision = precision
        st.session_state.recall = recall
        st.session_state.importance_df = importance_df

        # Also save it to this user's account in Supabase, so it's
        # still here the next time they log in - not just for the
        # rest of this browser session.
        save_results_to_db(st.session_state.user.id)

        st.success(
            "Analysis complete and saved to your account. Go to the "
            "Employee Lookup page to explore individual employees, or "
            "upload a new file above to re-analyze."
        )


elif page == "Employee Lookup":

    st.markdown('<div class="page-icon-badge">👥</div>', unsafe_allow_html=True)
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

        search_tab, ranking_tab = st.tabs(["🔍 Search", "📋 Full ranking"])

        # ----------------------------------------------------
        # TAB: SEARCH
        # ----------------------------------------------------

        with search_tab:

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

            st.dataframe(
                details_table, use_container_width=True, hide_index=True
            )

            if not pattern_df.empty:

                st.write("**Top factors for this employee:**")

                top_pattern_features = pattern_df["Feature"].head(3).tolist()
                risk_driving_factors = []

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

                        if closer_to_left:
                            risk_driving_factors.append(feature)

                        direction = (
                            "closer to the pattern seen in employees "
                            "who left"
                            if closer_to_left
                            else "closer to the pattern seen in "
                            "employees who stayed"
                        )

                        st.write(
                            f"- **{feature}**: this employee's value is "
                            f"**{employee_value:.1f}** ({direction}). "
                            f"Average for employees who left: "
                            f"{left_avg:.1f}, average for employees who "
                            f"stayed: {stayed_avg:.1f}."
                        )

                # ------------------------------------------------
                # PERSONALIZED RECOMMENDATION FOR THIS EMPLOYEE
                # ------------------------------------------------

                st.write("**Recommended action for this employee:**")

                if employee_row["RiskLevel"] == "Low risk":

                    st.write(
                        "No urgent action needed based on this data. "
                        "Continue regular check-ins as part of normal "
                        "management practice."
                    )

                elif risk_driving_factors:

                    for feature in risk_driving_factors:

                        details = get_recommendation_details(feature)

                        st.markdown(f"**{feature}**")
                        st.write(details["why"])

                        for step in details["steps"]:
                            st.write(f"- {step}")

                        st.write("")

                    st.caption(
                        "These suggestions are based on the factors "
                        "most associated with this specific employee's "
                        "risk score, not the company-wide averages "
                        "shown in the Analyze page. Use them as a "
                        "starting point for a conversation, not a "
                        "final decision."
                    )

                else:
                    st.write(
                        "This employee's risk score isn't clearly "
                        "explained by the top overall factors - a "
                        "direct check-in is the best next step."
                    )

        # ----------------------------------------------------
        # TAB: FULL RANKING
        # ----------------------------------------------------

        with ranking_tab:

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

            try:
                styled_table = (
                    ranked_table.style
                    .format({"RiskScore": "{:.0f}%"})
                    .map(color_risk_level, subset=["RiskLevel"])
                )
            except AttributeError:
                # Older pandas versions use .applymap() instead of
                # .map() on a Styler object - fall back to that if
                # .map() isn't available in this environment.
                styled_table = (
                    ranked_table.style
                    .format({"RiskScore": "{:.0f}%"})
                    .applymap(color_risk_level, subset=["RiskLevel"])
                )

            st.dataframe(
                styled_table,
                use_container_width=True,
                hide_index=True
            )


# ------------------------------------------------------------
# PAGE: HISTORY
# ------------------------------------------------------------

elif page == "History":

    st.markdown('<div class="page-icon-badge">🕐</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-title">Your analysis history</div>',
        unsafe_allow_html=True
    )

    st.caption(
        "Every dataset you've analyzed is saved here, most recent "
        "first. Load any past analysis to explore it again on the "
        "Employee Lookup page."
    )

    history = load_analysis_history(st.session_state.user.id)

    if not history:

        st.info(
            "No past analyses yet. Go to the Analyze page to upload "
            "your first dataset."
        )

    else:

        for entry in history:

            employee_count = len(entry.get("results_json") or [])
            created_at = entry.get("created_at", "Unknown date")

            with st.container():

                col1, col2, col3 = st.columns([3, 2, 1])

                with col1:
                    st.write(f"**{created_at}**")

                with col2:
                    st.write(f"{employee_count} employees")

                with col3:
                    if st.button("Load", key=f"load_{entry['id']}"):
                        if load_specific_analysis(entry["id"]):
                            st.success(
                                "Loaded. Go to Employee Lookup to "
                                "explore it."
                            )
                            st.rerun()

                st.markdown("---")


# ------------------------------------------------------------
# PAGE: SETTINGS
# ------------------------------------------------------------

elif page == "Settings":

    st.markdown('<div class="page-icon-badge">⚙️</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-title">Settings</div>',
        unsafe_allow_html=True
    )

    st.caption(
        "These preferences are saved to your account and applied "
        "every time you analyze new data."
    )

    current_company = st.session_state.preferences.get("company_name", "")
    current_high = st.session_state.preferences.get(
        "high_risk_threshold", 66
    )
    current_medium = st.session_state.preferences.get(
        "medium_risk_threshold", 33
    )

    company_name_input = st.text_input(
        "Company or display name",
        value=current_company,
        help="Shown in the sidebar throughout the app."
    )

    st.write("**Risk thresholds**")

    st.caption(
        "Employees at or above the high threshold are labeled "
        "\"High risk.\" Below that but at or above the medium "
        "threshold, they're \"Medium risk.\" Below the medium "
        "threshold, they're \"Low risk.\""
    )

    medium_threshold_input = st.slider(
        "Medium risk threshold (%)",
        min_value=0,
        max_value=100,
        value=int(current_medium)
    )

    high_threshold_input = st.slider(
        "High risk threshold (%)",
        min_value=0,
        max_value=100,
        value=int(current_high)
    )

    if high_threshold_input <= medium_threshold_input:

        st.warning(
            "The high risk threshold should be greater than the "
            "medium risk threshold, or the labels won't make sense."
        )

    if st.button("Save settings"):

        saved = save_preferences(
            st.session_state.user.id,
            company_name_input,
            high_threshold_input,
            medium_threshold_input
        )

        if saved:
            st.success(
                "Settings saved. These will apply the next time you "
                "analyze data."
            )
            st.rerun()


# ------------------------------------------------------------
# PAGE: ABOUT
# ------------------------------------------------------------

elif page == "About":

    st.markdown('<div class="page-icon-badge">ℹ️</div>', unsafe_allow_html=True)
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
