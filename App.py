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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

from supabase import create_client, Client
import plotly.graph_objects as go
from fpdf import FPDF
from datetime import date

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
            background: linear-gradient(90deg, #1A1D1C 0%, #1A1D1C 55%, #10B981 100%);
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
            margin: 0;
        }

        .subtitle {
            font-size: 16px;
            color: #5B635F;
            margin-bottom: 28px;
            max-width: 620px;
        }

        /* ---------- Handwritten accent text ---------- */
        .handwritten-accent {
            font-family: 'Caveat', cursive;
            font-size: 30px;
            color: #0D9488;
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
            color: #1A1D1C;
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
            background-color: #FFFFFF;
            border: 1px solid #E2E8E5;
            box-shadow: 0 1px 3px rgba(0,0,0,0.04);
            margin-bottom: 22px;
            color: #1A1D1C;
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
            background-color: rgba(239, 68, 68, 0.10);
            color: #DC2626;
            border: 1px solid rgba(239, 68, 68, 0.3);
        }

        .risk-medium {
            background-color: rgba(245, 158, 11, 0.10);
            color: #B45309;
            border: 1px solid rgba(245, 158, 11, 0.3);
        }

        .risk-low {
            background-color: rgba(16, 185, 129, 0.10);
            color: #047857;
            border: 1px solid rgba(16, 185, 129, 0.3);
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

if not st.session_state.splash_seen:

    st.markdown(
        """
        <style>
            .splash-overlay {
                position: fixed;
                top: 0;
                left: 0;
                width: 100vw;
                height: 100vh;
                background-color: #0A0B0A;
                display: flex;
                flex-direction: column;
                justify-content: center;
                align-items: center;
                z-index: 9999;
            }

            .liquid-loader {
                position: relative;
                width: 140px;
                height: 140px;
                border-radius: 50%;
                overflow: hidden;
                border: 3px solid rgba(52, 211, 153, 0.5);
                background: #0A0B0A;
                box-shadow: 0 0 30px rgba(16, 185, 129, 0.25);
            }

            .liquid-wave {
                position: absolute;
                width: 200%;
                height: 200%;
                left: -50%;
                border-radius: 42%;
                background: linear-gradient(180deg, #10B981, #0D9488);
                animation:
                    wave-rotate 4s linear infinite,
                    wave-fill 1.7s ease-in-out forwards;
            }

            .liquid-wave.wave2 {
                background: rgba(52, 211, 153, 0.55);
                animation:
                    wave-rotate 6s linear infinite reverse,
                    wave-fill 1.7s ease-in-out forwards;
            }

            .splash-logo {
                margin-top: 26px;
                width: 220px;
                max-width: 60vw;
            }

            @keyframes wave-rotate {
                from { transform: rotate(0deg); }
                to { transform: rotate(360deg); }
            }

            @keyframes wave-fill {
                from { top: 100%; }
                to { top: 12%; }
            }
        </style>

        <div class="splash-overlay">
            <div class="liquid-loader">
                <div class="liquid-wave wave1"></div>
                <div class="liquid-wave wave2"></div>
            </div>
            <img class="splash-logo" src="data:image/png;base64,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" />
        </div>
        """,
        unsafe_allow_html=True
    )

    time.sleep(1.9)
    st.session_state.splash_seen = True
    st.rerun()


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
        "medium_risk_threshold": 33,
        "onboarding_seen": False
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
                ),
                "onboarding_seen": row.get("onboarding_seen", False)
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


def mark_onboarding_seen(user_id):
    """
    Records that this account has seen the onboarding carousel, so
    it isn't shown again on future logins - only leaves company_name
    and thresholds untouched since only onboarding_seen is included.
    """

    try:
        supabase.table("user_preferences").upsert({
            "user_id": user_id,
            "onboarding_seen": True
        }).execute()

        st.session_state.preferences["onboarding_seen"] = True

    except Exception:
        pass


def load_analysis_history(user_id):
    """
    Loads a lightweight summary of every past analysis for this
    user (not the full data - just enough to list them), most
    recent first.
    """

    try:
        response = (
            supabase.table("retentia_results")
            .select("id, created_at, results_json, pattern_json")
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
                background-color: #FFFFFF;
                border: 1px solid #E2E8E5;
                border-radius: 14px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.04);
            }

            .login-title {
                font-family: 'Space Grotesk', sans-serif;
                font-size: 30px;
                font-weight: 700;
                text-align: center;
                background: linear-gradient(90deg, #1A1D1C 40%, #10B981 100%);
                -webkit-background-clip: text;
                background-clip: text;
                color: transparent;
                margin-bottom: 4px;
            }

            .login-subtitle {
                text-align: center;
                font-size: 14px;
                color: #5B635F;
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
            st.image("logo_dark.png", use_container_width=True)

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

        if auth_mode == "Log in":

            with st.expander("Forgot password?"):

                st.caption(
                    "Enter your email below, then click the button to "
                    "get a password reset link sent to it."
                )

                reset_email = st.text_input(
                    "Email for password reset", key="reset_email_input"
                )

                if st.button("Send password reset email"):

                    if not reset_email:
                        st.warning("Enter your email above first.")
                    else:
                        try:
                            supabase.auth.reset_password_email(
                                reset_email
                            )
                            st.success(
                                "If an account exists for that email, "
                                "a password reset link has been sent."
                            )
                        except Exception as error:
                            st.error(
                                f"Could not send reset email: {error}"
                            )

        st.caption(
            "Your data is kept private to your account and is never "
            "visible to other users."
        )

    st.stop()


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

if not st.session_state.preferences.get("onboarding_seen", False):

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
                mark_onboarding_seen(st.session_state.user.id)
                st.rerun()

        with nav_col2:
            button_label = "Get Started" if is_last_slide else "Next"
            if st.button(
                button_label, use_container_width=True, type="primary"
            ):
                if is_last_slide:
                    mark_onboarding_seen(st.session_state.user.id)
                else:
                    st.session_state.onboarding_index += 1
                st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

    st.stop()


# ------------------------------------------------------------
# HEADER (shown on every page once logged in)
# ------------------------------------------------------------

st.image("logo_dark.png", width=230)

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
display_profile_name = (
    company_name if company_name
    else st.session_state.user.email.split("@")[0]
)

with st.sidebar.popover(
    f"👤 {display_profile_name}", use_container_width=True
):
    st.write(f"**{display_profile_name}**")
    st.caption(st.session_state.user.email)
    st.markdown("---")

    if st.button(
        "⚙️ My Account", use_container_width=True, key="profile_menu_account"
    ):
        st.session_state.pending_nav = "Settings"
        st.rerun()

    if st.button(
        "🚪 Log out", use_container_width=True, key="profile_menu_logout"
    ):
        st.session_state.user = None
        st.session_state.analyzed = False
        st.rerun()

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
    "Retention Action Center": "🎯",
    "History": "🕐",
    "Settings": "⚙️",
    "About": "ℹ️",
}

page = st.sidebar.radio(
    "Navigate",
    ["Home", "Analyze", "Employee Lookup", "Retention Action Center",
     "History", "Settings", "About"],
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


def compute_data_health_report(
    row_count,
    X,
    y,
    duplicate_count,
    empty_columns_removed
):
    """
    Runs a handful of practical checks against the cleaned dataset
    actually used to train the model (X, y - after cleaning, target
    handling, and dropping ID-like columns) and turns them into a
    plain-language health report: a 0-100 score plus a list of
    checks, each labeled "good", "warning", or "issue".

    This is a quick sanity check on the data itself, separate from
    model accuracy - a model can report high accuracy on data that's
    still small, imbalanced, or full of near-useless columns.
    """

    checks = []
    score = 100

    # --- Sample size ---
    if row_count < 30:
        checks.append({
            "status": "issue",
            "title": "Very small dataset",
            "detail": (
                f"Only {row_count} rows remain after cleaning. Model "
                f"metrics and risk scores will be unstable - treat "
                f"results as illustrative, not decision-grade."
            )
        })
        score -= 25
    elif row_count < 100:
        checks.append({
            "status": "warning",
            "title": "Small dataset",
            "detail": (
                f"{row_count} rows remain after cleaning. Results are "
                f"usable but will get more reliable with more data."
            )
        })
        score -= 10
    else:
        checks.append({
            "status": "good",
            "title": "Dataset size",
            "detail": f"{row_count} rows - enough for a stable model."
        })

    # --- Class balance ---
    if y is not None and len(y) > 0:

        positive_rate = y.mean() * 100
        minority_rate = min(positive_rate, 100 - positive_rate)

        if minority_rate < 5:
            checks.append({
                "status": "issue",
                "title": "Severe class imbalance",
                "detail": (
                    f"Only about {minority_rate:.1f}% of employees "
                    f"are in the minority class (stayed or left). "
                    f"The model has very few real examples to learn "
                    f"from for that group."
                )
            })
            score -= 20
        elif minority_rate < 15:
            checks.append({
                "status": "warning",
                "title": "Class imbalance",
                "detail": (
                    f"About {minority_rate:.1f}% of employees are in "
                    f"the minority class. This is common for "
                    f"attrition data and is already accounted for "
                    f"during training, but treat scores with a "
                    f"little extra caution."
                )
            })
            score -= 5
        else:
            checks.append({
                "status": "good",
                "title": "Class balance",
                "detail": (
                    f"About {minority_rate:.1f}% minority class - a "
                    f"reasonable split for the model to learn from."
                )
            })

    # --- Missing values in predictor columns (before imputation) ---
    if X is not None and len(X.columns) > 0:

        missing_shares = X.isna().mean() * 100
        high_missing = missing_shares[missing_shares > 30]
        moderate_missing = missing_shares[
            (missing_shares > 10) & (missing_shares <= 30)
        ]

        if len(high_missing) > 0:
            checks.append({
                "status": "issue",
                "title": "Columns with heavy missing data",
                "detail": (
                    "Over 30% of values are missing in: "
                    + ", ".join(high_missing.index.tolist())
                    + ". These are filled in automatically during "
                    "training, but may add more noise than signal."
                )
            })
            score -= 15
        if len(moderate_missing) > 0:
            checks.append({
                "status": "warning",
                "title": "Columns with some missing data",
                "detail": (
                    "10-30% of values are missing in: "
                    + ", ".join(moderate_missing.index.tolist())
                    + ". Handled automatically, but worth checking "
                    "the source data if this is unexpected."
                )
            })
            score -= 5
        if len(high_missing) == 0 and len(moderate_missing) == 0:
            checks.append({
                "status": "good",
                "title": "Missing data",
                "detail": (
                    "No predictor column has significant missing data."
                )
            })

    # --- Columns that never vary (no predictive value) ---
    constant_columns = []

    if X is not None:
        for column in X.columns:
            non_null = X[column].dropna()
            if len(non_null) > 0 and non_null.nunique() <= 1:
                constant_columns.append(column)

    if constant_columns:
        checks.append({
            "status": "warning",
            "title": "Columns with a single value",
            "detail": (
                "These columns never change, so they can't help "
                "predict attrition: "
                + ", ".join(map(str, constant_columns))
            )
        })
        score -= 5 * min(len(constant_columns), 2)

    # --- Text columns that look like identifiers ---
    high_cardinality_columns = []

    if X is not None and len(X) > 0:

        categorical_columns = X.select_dtypes(
            exclude=["number"]
        ).columns

        for column in categorical_columns:
            unique_count = X[column].nunique(dropna=True)
            if unique_count > max(20, len(X) * 0.5):
                high_cardinality_columns.append(column)

    if high_cardinality_columns:
        checks.append({
            "status": "warning",
            "title": "Columns that look like identifiers",
            "detail": (
                "These text columns have almost as many unique "
                "values as there are employees, which usually means "
                "they're an ID rather than a real pattern: "
                + ", ".join(map(str, high_cardinality_columns))
            )
        })
        score -= 5

    # --- Already-handled cleaning issues, shown as reassurance ---
    if duplicate_count > 0:
        checks.append({
            "status": "good",
            "title": "Duplicate rows removed",
            "detail": (
                f"{duplicate_count} duplicate row(s) were found and "
                f"removed automatically before analysis."
            )
        })

    if empty_columns_removed:
        checks.append({
            "status": "good",
            "title": "Empty columns removed",
            "detail": (
                "These columns had no data at all and were dropped: "
                + ", ".join(map(str, empty_columns_removed))
            )
        })

    score = max(0, min(100, score))

    if score >= 80:
        label = "Good"
    elif score >= 50:
        label = "Fair"
    else:
        label = "Needs attention"

    return {"score": score, "label": label, "checks": checks}


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

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=5,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
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


# ------------------------------------------------------------
# RETENTION ACTION CENTER HELPERS
# ------------------------------------------------------------
# These group high-risk employees by the broad *cause* behind their
# risk (career growth, workload, compensation, satisfaction) rather
# than by the raw column name, so the Retention Action Center page
# can show "12 employees at risk over workload" instead of a wall
# of individual feature names.

CAUSE_CATEGORIES = ["Career Growth", "Workload & Overtime",
                     "Compensation", "Job Satisfaction", "Other"]

CAUSE_META = {
    "Career Growth": {"icon": "📈", "color": "#10B981"},
    "Workload & Overtime": {"icon": "🔥", "color": "#F59E0B"},
    "Compensation": {"icon": "💰", "color": "#3B82F6"},
    "Job Satisfaction": {"icon": "🙂", "color": "#8B5CF6"},
    "Other": {"icon": "🔎", "color": "#6B726F"},
    "Unclear": {"icon": "❔", "color": "#8A928F"},
}


def categorize_feature(feature_name):
    """
    Maps a raw data column name to one of a small set of
    human-readable retention "causes". Mirrors the keyword logic in
    get_recommendation_details() above, so a factor and its
    recommendation always land in the same bucket.
    """

    name = str(feature_name).lower()

    if any(word in name for word in ["promot", "training", "tenure", "years"]):
        return "Career Growth"

    if any(word in name for word in [
        "overtime", "workload", "hours", "worklife", "work_life", "balance"
    ]):
        return "Workload & Overtime"

    if any(word in name for word in ["salary", "income", "pay", "compensation"]):
        return "Compensation"

    if any(word in name for word in [
        "satisfaction", "environment", "performance", "attendance"
    ]):
        return "Job Satisfaction"

    return "Other"


def get_employee_primary_cause(employee_row, pattern_df):
    """
    Returns the single strongest cause category for one employee:
    walks pattern_df (already sorted, strongest company-wide factor
    first) and returns the category of the first factor where this
    employee's own value sits closer to the "left" average than the
    "stayed" average. Returns None if no tracked factor points at
    this employee at all.
    """

    for _, feature_pattern in pattern_df.iterrows():

        feature = feature_pattern["Feature"]

        if feature not in employee_row.index:
            continue

        employee_value = employee_row[feature]
        stayed_avg = feature_pattern["Stayed average"]
        left_avg = feature_pattern["Left average"]

        closer_to_left = (
            abs(employee_value - left_avg) < abs(employee_value - stayed_avg)
        )

        if closer_to_left:
            return categorize_feature(feature)

    return None


def cause_badge_html(cause):
    """
    Returns an HTML span styled as a colored badge for a retention
    cause category (Career Growth, Workload & Overtime, etc.),
    matching the same pill shape as risk_badge_html so the two read
    as one visual system wherever they appear together.
    """

    meta = CAUSE_META.get(cause, CAUSE_META["Unclear"])

    return (
        f'<span class="risk-badge" style="background-color:'
        f'{meta["color"]}1A; color:{meta["color"]}; border:1px solid '
        f'{meta["color"]}4D;">{meta["icon"]} {cause}</span>'
    )


def compute_cause_counts_for_history_entry(entry):
    """
    Recomputes the same primary-cause categories used by the
    Retention Action Center, but for one saved historical analysis
    (a row from load_analysis_history), so cause trends over time can
    be plotted on the History page without changing what's stored in
    Supabase - it just reads the pattern_json saved alongside that
    run's results_json and re-runs the same per-employee logic.

    Returns:
        - a dict of {cause: count} if the run has usable data
        - {} if the run has results but no high-risk employees
        - None if the run predates pattern_json being saved, or is
          otherwise missing what's needed to compute a cause at all
    """

    results_json = entry.get("results_json") or []
    pattern_json = entry.get("pattern_json") or []

    if not results_json or not pattern_json:
        return None

    entry_results_df = pd.DataFrame(results_json)
    entry_pattern_df = pd.DataFrame(pattern_json)

    if "RiskLevel" not in entry_results_df.columns or entry_pattern_df.empty:
        return None

    entry_high_risk_df = entry_results_df[
        entry_results_df["RiskLevel"] == "High risk"
    ]

    if entry_high_risk_df.empty:
        return {}

    causes = entry_high_risk_df.apply(
        lambda row: (
            get_employee_primary_cause(row, entry_pattern_df) or "Unclear"
        ),
        axis=1
    )

    return causes.value_counts().to_dict()


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

def generate_pdf_report(
    company_name,
    total_employees,
    attrition_rate,
    retention_rate,
    top_department,
    top_department_rate,
    top_factor_explanations,
    recommendations,
    top_risk_employees,
    id_column,
    cause_breakdown=None
):
    """
    Builds a short, plain-language 1-2 page PDF report a manager can
    download and forward - no jargon, no raw numbers dump, just what's
    happening and what to do about it.

    Every cell/multi_cell call explicitly resets the cursor to the
    left margin first and uses an explicit width (pdf.epw) rather
    than width=0. Some fpdf2 versions can let the cursor drift right
    after repeated multi_cell calls, eventually leaving too little
    width to render text at all - resetting explicitly avoids that
    regardless of version quirks.

    cause_breakdown (optional): a list of {"cause": str, "count": int}
    dicts - one row per Retention Action Center category among
    high-risk employees (e.g. "Workload & Overtime": 4) - shown as a
    plain-text breakdown so the PDF matches what's on that page
    without needing colors or icons, which fpdf2's default font can't
    render reliably.
    """

    pdf = FPDF()
    pdf.add_page()
    width = pdf.epw

    def write_heading(text, size=14):
        pdf.set_x(pdf.l_margin)
        pdf.set_font("Helvetica", "B", size)
        pdf.cell(width, 8, text)
        pdf.ln(9)

    def write_paragraph(text, size=11, style=""):
        pdf.set_x(pdf.l_margin)
        pdf.set_font("Helvetica", style, size)
        pdf.multi_cell(width, 7, text)

    write_heading("Retentia - Employee Retention Report", size=18)

    if company_name:
        write_paragraph(f"Company: {company_name}")
    write_paragraph(f"Date: {date.today().strftime('%B %d, %Y')}")
    pdf.ln(6)

    write_heading("Overview")
    write_paragraph(
        f"We looked at {total_employees} employees. "
        f"{retention_rate:.0f} out of every 100 are likely to stay. "
        f"{attrition_rate:.0f} out of every 100 are at risk of leaving."
    )
    pdf.ln(6)

    if top_department:
        write_heading("Where to Look First")
        write_paragraph(
            f"{top_department} has the highest share of at-risk "
            f"employees, at about {top_department_rate:.0f}%. Start "
            f"here."
        )
        pdf.ln(6)

    if cause_breakdown:
        write_heading("Where Risk Is Coming From")
        write_paragraph(
            "Among employees flagged as high risk, here is the single "
            "factor most associated with each person's own risk score, "
            "grouped into broad categories:"
        )
        total_high_risk = sum(entry["count"] for entry in cause_breakdown)
        for entry in cause_breakdown:
            share = (
                (entry["count"] / total_high_risk * 100)
                if total_high_risk else 0
            )
            write_paragraph(
                f"- {entry['cause']}: {entry['count']} employee"
                f"{'s' if entry['count'] != 1 else ''} "
                f"(about {share:.0f}% of high-risk employees)"
            )
        pdf.ln(6)

    if top_factor_explanations:
        write_heading("Why Employees Are Leaving")
        for explanation in top_factor_explanations:
            write_paragraph(f"- {explanation}")
        pdf.ln(6)

    if recommendations:
        write_heading("What To Do")
        for number, recommendation in enumerate(recommendations, start=1):
            write_paragraph(f"{number}. {recommendation}")
        pdf.ln(6)

    if top_risk_employees is not None and len(top_risk_employees) > 0:
        write_heading("Employees To Check In With Soon")
        for _, row in top_risk_employees.iterrows():
            write_paragraph(
                f"- {row[id_column]}: about {row['RiskScore']:.0f}% "
                f"risk of leaving"
            )
        pdf.ln(6)

    write_paragraph(
        "This report is a decision-support summary, not a final "
        "conclusion. Review it alongside direct conversations with "
        "employees and managers before making decisions.",
        size=9,
        style="I"
    )

    return bytes(pdf.output())
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
                background-color: rgba(16, 185, 129, 0.10);
                border: 1px solid rgba(16, 185, 129, 0.3);
                color: #0D9488;
                font-size: 13px;
                font-weight: 600;
                margin-bottom: 18px;
            }

            .hero-heading {
                font-family: 'Space Grotesk', sans-serif;
                font-size: 44px;
                font-weight: 700;
                line-height: 1.15;
                color: #1A1D1C;
                margin-bottom: 14px;
            }

            .hero-heading .accent {
                background: linear-gradient(90deg, #0D9488, #10B981);
                -webkit-background-clip: text;
                background-clip: text;
                color: transparent;
            }

            .hero-subtext {
                font-size: 15px;
                color: #5B635F;
                max-width: 520px;
                margin-bottom: 22px;
            }

            .feature-card {
                background-color: #FFFFFF;
                border: 1px solid #E2E8E5;
                border-radius: 12px;
                padding: 20px;
                height: 100%;
                box-shadow: 0 1px 3px rgba(0,0,0,0.04);
            }

            .feature-card h4 {
                font-family: 'Space Grotesk', sans-serif;
                font-size: 17px;
                margin: 10px 0 6px 0;
                color: #1A1D1C;
            }

            .feature-card p {
                font-size: 13px;
                color: #5B635F;
                margin-bottom: 0;
            }

            .stat-card {
                background-color: #FFFFFF;
                border: 1px solid #E2E8E5;
                border-radius: 12px;
                padding: 18px 20px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.04);
            }

            .stat-number {
                font-family: 'Space Grotesk', sans-serif;
                font-size: 30px;
                font-weight: 700;
                color: #1A1D1C;
            }

            .stat-label {
                font-size: 13px;
                color: #5B635F;
                margin-bottom: 4px;
            }

            .activity-item {
                display: flex;
                justify-content: space-between;
                padding: 10px 0;
                border-bottom: 1px solid #E2E8E5;
                font-size: 13px;
            }

            .activity-item:last-child {
                border-bottom: none;
            }

            .activity-title {
                color: #1A1D1C;
                font-weight: 600;
            }

            .activity-sub {
                color: #5B635F;
                font-size: 12px;
            }

            .activity-time {
                color: #8A928F;
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
                    .analyze-liquid-loader {
                        position: relative;
                        width: 140px;
                        height: 140px;
                        border-radius: 50%;
                        overflow: hidden;
                        border: 3px solid rgba(52, 211, 153, 0.5);
                        background: #0A0B0A;
                        box-shadow: 0 0 30px rgba(16, 185, 129, 0.25);
                        margin: 0 auto;
                    }

                    .analyze-liquid-wave {
                        position: absolute;
                        width: 200%;
                        height: 200%;
                        left: -50%;
                        border-radius: 42%;
                        background: linear-gradient(180deg, #10B981, #0D9488);
                        animation:
                            analyze-wave-rotate 9s linear infinite,
                            analyze-wave-fill 5.6s ease-in-out forwards;
                    }

                    .analyze-liquid-wave.wave2 {
                        background: rgba(52, 211, 153, 0.55);
                        animation:
                            analyze-wave-rotate 12s linear infinite reverse,
                            analyze-wave-fill 5.6s ease-in-out forwards;
                    }

                    .analyze-splash-logo {
                        display: block;
                        margin: 22px auto 0 auto;
                        width: 200px;
                        max-width: 60vw;
                    }

                    @keyframes analyze-wave-rotate {
                        from { transform: rotate(0deg); }
                        to { transform: rotate(360deg); }
                    }

                    @keyframes analyze-wave-fill {
                        from { top: 100%; }
                        to { top: 12%; }
                    }
                </style>

                <div style="display:flex; flex-direction:column; align-items:center;">
                    <div class="analyze-liquid-loader">
                        <div class="analyze-liquid-wave wave1"></div>
                        <div class="analyze-liquid-wave wave2"></div>
                    </div>
                    <img class="analyze-splash-logo" src="data:image/png;base64,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" />
                    <p style="text-align:center; color:#5B635F; margin-top:14px;">
                        Analyzing your data…</p>
                </div>
                """,
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

        # --------------------------------------------------------
        # RISK SCORES (computed here so stat cards below can use
        # them for both a fresh upload and this same run)
        # --------------------------------------------------------

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
        id_column = st.session_state.id_column

    # ----------------------------------------------------------------
    # SHARED DISPLAY - works for both a fresh upload and a cached one
    # ----------------------------------------------------------------

    st.markdown(
        '<div class="section-title">Analysis results</div>',
        unsafe_allow_html=True
    )

    at_risk_count = (results_df["RiskLevel"] == "High risk").sum()
    retention_rate = 100 - attrition_rate

    stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)

    with stat_col1:
        st.markdown(
            '<div class="stat-card">'
            '<div class="stat-label">👥 Total Employees</div>'
            f'<div class="stat-number">{len(results_df)}</div>'
            '</div>',
            unsafe_allow_html=True
        )

    with stat_col2:
        st.markdown(
            '<div class="stat-card">'
            '<div class="stat-label">⚠️ At Risk of Leaving</div>'
            f'<div class="stat-number">{at_risk_count}</div>'
            '</div>',
            unsafe_allow_html=True
        )

    with stat_col3:
        st.markdown(
            '<div class="stat-card">'
            '<div class="stat-label">🛡️ Retention Rate</div>'
            f'<div class="stat-number">{retention_rate:.1f}%</div>'
            '</div>',
            unsafe_allow_html=True
        )

    with stat_col4:
        st.markdown(
            '<div class="stat-card">'
            '<div class="stat-label">📈 Attrition Rate</div>'
            f'<div class="stat-number">{attrition_rate:.1f}%</div>'
            '</div>',
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # --------------------------------------------------------
    # DONUT CHART: Attrition Risk Distribution
    # --------------------------------------------------------

    donut_col, insights_col = st.columns([3, 2])

    with donut_col:

        st.markdown(
            '<div class="section-title">Attrition risk distribution'
            '</div>',
            unsafe_allow_html=True
        )

        risk_counts = (
            results_df["RiskLevel"]
            .value_counts()
            .reindex(["High risk", "Medium risk", "Low risk"])
            .fillna(0)
        )

        risk_colors = {
            "High risk": "#EF4444",
            "Medium risk": "#F59E0B",
            "Low risk": "#34D399"
        }

        donut_fig = go.Figure(data=[go.Pie(
            labels=risk_counts.index.tolist(),
            values=risk_counts.values.tolist(),
            hole=0.65,
            marker=dict(
                colors=[risk_colors[level] for level in risk_counts.index]
            ),
            textinfo="label+percent",
            textfont=dict(color="#1A1D1C", size=12)
        )])

        donut_fig.update_layout(
            showlegend=True,
            legend=dict(font=dict(color="#1A1D1C")),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            annotations=[dict(
                text=f"{(at_risk_count/len(results_df)*100):.1f}%<br>"
                     f"High Risk",
                x=0.5, y=0.5,
                font=dict(size=18, color="#1A1D1C"),
                showarrow=False
            )],
            margin=dict(t=10, b=10, l=10, r=10),
            height=320
        )

        st.plotly_chart(donut_fig, use_container_width=True)

    with insights_col:

        st.markdown(
            '<div class="section-title">Quick insights</div>',
            unsafe_allow_html=True
        )

        insight_cards_html = '<div class="feature-card" style="margin-bottom:10px;">'
        insight_added = False

        if department_column and department_column in results_df.columns:

            dept_risk = (
                results_df.groupby(department_column)["RiskLevel"]
                .apply(lambda s: (s == "High risk").mean() * 100)
                .sort_values(ascending=False)
            )

            if len(dept_risk) > 0 and dept_risk.iloc[0] > 0:
                top_dept = dept_risk.index[0]
                top_dept_rate = dept_risk.iloc[0]
                insight_cards_html += (
                    f'<p>🔴 <strong>{top_dept}</strong> has the highest '
                    f'high-risk share, at {top_dept_rate:.1f}% of its '
                    f'employees.</p>'
                )
                insight_added = True

        if importance_df is not None and not importance_df.empty:
            top_driver = importance_df.iloc[0]["Feature"]
            insight_cards_html += (
                f'<p>📊 <strong>{top_driver}</strong> is the single '
                f'strongest predictor of attrition in this dataset.</p>'
            )
            insight_added = True

        if not pattern_df.empty:
            top_pattern = pattern_df.iloc[0]
            insight_cards_html += (
                f'<p>📈 Employees who left differ most from those who '
                f'stayed on <strong>{top_pattern["Feature"]}</strong> '
                f'(left avg {top_pattern["Left average"]:.1f} vs. '
                f'stayed avg {top_pattern["Stayed average"]:.1f}).</p>'
            )
            insight_added = True

        if not insight_added:
            insight_cards_html += (
                "<p>Not enough patterns in this dataset yet to "
                "generate insights.</p>"
            )

        insight_cards_html += '</div>'

        st.markdown(insight_cards_html, unsafe_allow_html=True)

    # ------------------------------------------------------------
    # DEPARTMENT RISK MAP
    # ------------------------------------------------------------
    # A full risk-composition breakdown for every department, not
    # just the single worst one called out in Quick Insights above -
    # so it's possible to see at a glance where risk is concentrated
    # across the whole organization, and where it isn't.

    if department_column and department_column in results_df.columns:

        st.markdown(
            '<div class="section-title">Department risk map</div>',
            unsafe_allow_html=True
        )

        st.caption(
            "Risk composition for every department, sorted by share "
            "of high-risk employees - worst first."
        )

        risk_map_colors = {
            "High risk": "#EF4444",
            "Medium risk": "#F59E0B",
            "Low risk": "#34D399"
        }

        dept_employee_counts = results_df.groupby(department_column).size()

        dept_composition = (
            results_df.groupby(department_column)["RiskLevel"]
            .value_counts(normalize=True)
            .mul(100)
            .unstack(fill_value=0)
        )

        for level in risk_map_colors:
            if level not in dept_composition.columns:
                dept_composition[level] = 0.0

        dept_composition = dept_composition.sort_values(
            "High risk", ascending=False
        )

        map_fig = go.Figure()

        for level in ["Low risk", "Medium risk", "High risk"]:
            map_fig.add_trace(go.Bar(
                y=dept_composition.index.astype(str).tolist(),
                x=dept_composition[level].tolist(),
                name=level,
                orientation="h",
                marker=dict(color=risk_map_colors[level]),
                hovertemplate=f"{level}: %{{x:.0f}}%<extra></extra>"
            ))

        map_fig.update_layout(
            barmode="stack",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#1A1D1C"),
            height=max(220, 46 * len(dept_composition)),
            margin=dict(t=10, b=10, l=10, r=10),
            legend=dict(orientation="h", y=-0.15),
            xaxis=dict(
                title="Share of employees (%)",
                range=[0, 100],
                showgrid=False
            ),
            yaxis=dict(title="", autorange="reversed")
        )

        st.plotly_chart(map_fig, use_container_width=True)

        dept_summary_table = pd.DataFrame({
            "Department": dept_composition.index.astype(str),
            "Employees": [
                dept_employee_counts[d] for d in dept_composition.index
            ],
            "High risk %": dept_composition["High risk"].round(0),
            "Medium risk %": dept_composition["Medium risk"].round(0),
            "Low risk %": dept_composition["Low risk"].round(0),
        }).reset_index(drop=True)

        with st.expander("View exact numbers"):
            st.dataframe(
                dept_summary_table,
                use_container_width=True,
                hide_index=True
            )

        st.markdown("<br>", unsafe_allow_html=True)

    if cleaning_stats_available:

        # ------------------------------------------------------------
        # DATA HEALTH CHECK
        # ------------------------------------------------------------
        # A quick sanity check on the data itself - separate from
        # model accuracy, since a model can report solid accuracy on
        # data that's still small, imbalanced, or full of near-useless
        # columns. Shown right after a fresh upload, using the same
        # X/y actually fed to the model.

        health_report = compute_data_health_report(
            row_count=len(data),
            X=X,
            y=y,
            duplicate_count=duplicate_count,
            empty_columns_removed=empty_columns
        )

        st.markdown(
            '<div class="section-title">Data health check</div>',
            unsafe_allow_html=True
        )

        health_colors = {
            "Good": "#10B981",
            "Fair": "#F59E0B",
            "Needs attention": "#EF4444"
        }
        health_color = health_colors.get(health_report["label"], "#8A928F")

        st.markdown(
            '<div class="feature-card" style="display:flex; '
            'align-items:center; gap:16px; margin-bottom:14px;">'
            '<div style="font-family:\'Space Grotesk\',sans-serif; '
            f'font-size:32px; font-weight:700; color:{health_color};">'
            f'{health_report["score"]}/100</div>'
            '<div>'
            f'<p style="margin:0; font-weight:600; color:{health_color};">'
            f'{health_report["label"]}</p>'
            '<p style="margin:0; color:#6B726F; font-size:13px;">'
            'How reliable this dataset is for training a model and '
            'trusting the risk scores it produces.</p>'
            '</div></div>',
            unsafe_allow_html=True
        )

        status_icons = {"good": "✅", "warning": "⚠️", "issue": "🔴"}

        for check in health_report["checks"]:

            icon = status_icons.get(check["status"], "•")

            st.markdown(
                '<div class="feature-card" style="margin-bottom:8px;">'
                f'<p style="margin:0; font-weight:600;">{icon} '
                f'{check["title"]}</p>'
                '<p style="margin:4px 0 0 0; font-size:13px; '
                f'color:#5B635F;">{check["detail"]}</p>'
                '</div>',
                unsafe_allow_html=True
            )

        st.caption(
            "This checks the data itself, not the model's predictions "
            "- a model can still report solid accuracy on data that "
            "has some of these issues."
        )

        st.markdown("<br>", unsafe_allow_html=True)

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
        '<div class="section-title">How reliable is this?</div>',
        unsafe_allow_html=True
    )

    if accuracy is not None:

        recall_out_of_100 = round(recall * 100)
        precision_out_of_100 = round(precision * 100)
        accuracy_out_of_100 = round(accuracy * 100)

        plain_language_html = (
            '<div class="feature-card">'
            f'<p>🎯 <strong>Catching real leavers:</strong> out of every '
            f'100 employees who actually left, this model would have '
            f'correctly flagged about <strong>{recall_out_of_100}</strong> '
            f'of them as at-risk beforehand. '
            f'<span style="color:#6B726F;">(This is called "recall.")</span>'
            f'</p>'
            f'<p>✅ <strong>Trusting a "high risk" flag:</strong> when '
            f'this model says an employee is high risk, it turns out to '
            f'be right about <strong>{precision_out_of_100}</strong> '
            f'times out of 100. '
            f'<span style="color:#6B726F;">(This is called "precision.")'
            f'</span></p>'
            f'<p>📊 <strong>Overall correctness:</strong> across all '
            f'employees, what this model predicted matched what '
            f'actually happened about <strong>{accuracy_out_of_100}'
            f'</strong> times out of 100. '
            f'<span style="color:#6B726F;">(This is called "accuracy.")'
            f'</span></p>'
            '</div>'
        )

        st.markdown(plain_language_html, unsafe_allow_html=True)

        st.caption(
            "These numbers come from testing the model on employees it "
            "hadn't seen before. They're a helpful guide, not a "
            "guarantee - always weigh them alongside real conversations "
            "with employees and managers."
        )

    else:
        st.write(
            "Reliability information isn't available for this "
            "analysis - upload a file above to run a fresh analysis."
        )

    # ------------------------------------------------------------
    # FEATURE IMPORTANCE
    # ------------------------------------------------------------

    st.markdown(
        '<div class="section-title">Top risk factors</div>',
        unsafe_allow_html=True
    )

    if importance_df is not None and not importance_df.empty:

        top_features = importance_df.head(6).copy()
        max_importance = top_features["Importance"].max()

        bar_pieces = ['<div class="feature-card">']

        for _, row in top_features.iterrows():

            pct = (
                (row["Importance"] / max_importance * 100)
                if max_importance > 0 else 0
            )

            # Built as one unindented line per bar (no leading whitespace,
            # no blank lines between pieces) - Streamlit's markdown
            # renderer treats indented, blank-line-separated text as a
            # code block and displays it literally, even with
            # unsafe_allow_html=True, so this has to stay compact.
            bar_pieces.append(
                f'<div style="margin-bottom:14px;">'
                f'<div style="display:flex; justify-content:space-between; '
                f'font-size:13px; color:#1A1D1C; margin-bottom:4px;">'
                f'<span>{row["Feature"]}</span>'
                f'<span style="color:#0D9488;">{pct:.0f}%</span>'
                f'</div>'
                f'<div style="background:#E2E8E5; border-radius:6px; height:8px;">'
                f'<div style="background:linear-gradient(90deg,#0D9488,#34D399); '
                f'width:{pct:.0f}%; height:8px; border-radius:6px;"></div>'
                f'</div>'
                f'</div>'
            )

        bar_pieces.append('</div>')
        bars_html = "".join(bar_pieces)

        st.markdown(bars_html, unsafe_allow_html=True)

        st.caption(
            "Feature importance indicates which variables the decision "
            "tree used most strongly, shown here relative to the "
            "single strongest factor. It does not prove that a factor "
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

The random forest model achieved **{accuracy:.1%} accuracy**, with
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

    st.markdown(
        '<div class="section-title">Recommended organization-wide '
        'actions</div>',
        unsafe_allow_html=True
    )

    st.caption(
        "These are company-wide, systemic actions based on overall "
        "patterns. For specific actions about an individual employee, "
        "see the Employee Lookup page."
    )

    recommendations_detailed = [
        {
            "title": "Dig into the biggest patterns",
            "why": (
                "The numbers above point to specific things - like "
                "overtime, promotion timing, or satisfaction - but "
                "numbers alone don't explain why. Understanding the "
                "real reason is what makes the next steps effective."
            ),
            "step": (
                "Pick the single strongest pattern shown above and "
                "have HR or a team lead investigate what's actually "
                "happening on the ground."
            )
        },
        {
            "title": "Check the pattern against real feedback",
            "why": (
                "A pattern in data can point in the right direction "
                "but still miss the full picture. Talking to people "
                "confirms whether the model's guess matches reality."
            ),
            "step": (
                "Run a short survey or a few informal 'stay interviews' "
                "with employees to see if they mention the same issues "
                "the data is flagging."
            )
        },
        {
            "title": "Fix the systems, not just the symptoms",
            "why": (
                "If overtime, pay, or promotion timing keeps showing "
                "up as a risk factor, that's usually a policy issue, "
                "not an individual one."
            ),
            "step": (
                "Review the relevant policy (workload limits, pay "
                "bands, promotion cycles) and identify one concrete "
                "change to test."
            )
        },
        {
            "title": "Act on groups, not just the model's word",
            "why": (
                "The model is a guide, not a judge. Treating its "
                "output as the final say on any one person risks "
                "unfair decisions and misses the bigger picture."
            ),
            "step": (
                "Use these patterns to shape team-wide or "
                "department-wide initiatives, and pair them with "
                "individual conversations from Employee Lookup."
            )
        },
        {
            "title": "Keep checking back",
            "why": (
                "Workforces change. A pattern that's true today may "
                "shift after a policy change, a new hire wave, or "
                "a tough quarter."
            ),
            "step": (
                "Re-run this analysis every few months with fresh "
                "data to see whether the picture is improving."
            )
        }
    ]

    for item in recommendations_detailed:
        st.markdown(
            '<div class="feature-card" style="margin-bottom:10px;">'
            f'<h4 style="margin-top:0;">{item["title"]}</h4>'
            f'<p>{item["why"]}</p>'
            f'<p><strong>Next step:</strong> {item["step"]}</p>'
            '</div>',
            unsafe_allow_html=True
        )

    # A plain-text version for the downloadable PDF report, which
    # doesn't render HTML cards.
    recommendations = [
        f'{item["title"]} - {item["step"]}'
        for item in recommendations_detailed
    ]

    # ------------------------------------------------------------
    # DOWNLOADABLE REPORT
    # ------------------------------------------------------------

    st.markdown(
        '<div class="section-title">Download report</div>',
        unsafe_allow_html=True
    )

    st.write(
        "A short, plain-language summary you can save or forward - "
        "what's happening and what to do about it, without the "
        "underlying data."
    )

    report_top_department = None
    report_top_department_rate = 0

    if department_column and department_column in data.columns:
        dept_risk_for_report = (
            results_df.groupby(department_column)["RiskLevel"]
            .apply(lambda s: (s == "High risk").mean() * 100)
            .sort_values(ascending=False)
        )
        if len(dept_risk_for_report) > 0 and dept_risk_for_report.iloc[0] > 0:
            report_top_department = dept_risk_for_report.index[0]
            report_top_department_rate = dept_risk_for_report.iloc[0]

    report_factor_explanations = []
    if importance_df is not None and not importance_df.empty:
        for feature in importance_df.head(5)["Feature"].tolist():
            details = get_recommendation_details(feature)
            if details["why"] not in report_factor_explanations:
                report_factor_explanations.append(details["why"])
            if len(report_factor_explanations) == 3:
                break

    report_top_risk_employees = results_df.sort_values(
        "RiskScore", ascending=False
    ).head(5)

    # Same primary-cause categories shown on the Retention Action
    # Center page, counted across every high-risk employee, so the
    # PDF a manager forwards tells the same story as that page.
    report_cause_breakdown = []

    if not pattern_df.empty:

        high_risk_for_report = results_df[
            results_df["RiskLevel"] == "High risk"
        ]

        if not high_risk_for_report.empty:

            report_causes = high_risk_for_report.apply(
                lambda row: (
                    get_employee_primary_cause(row, pattern_df)
                    or "Unclear"
                ),
                axis=1
            )

            report_cause_breakdown = [
                {"cause": cause, "count": int(count)}
                for cause, count in report_causes.value_counts().items()
            ]

    pdf_bytes = generate_pdf_report(
        company_name=st.session_state.preferences.get("company_name", ""),
        total_employees=len(results_df),
        attrition_rate=attrition_rate,
        retention_rate=100 - attrition_rate,
        top_department=report_top_department,
        top_department_rate=report_top_department_rate,
        top_factor_explanations=report_factor_explanations,
        recommendations=recommendations,
        top_risk_employees=report_top_risk_employees,
        id_column=id_column,
        cause_breakdown=report_cause_breakdown
    )

    st.download_button(
        label="📄 Download management report (PDF)",
        data=pdf_bytes,
        file_name="retentia_report.pdf",
        mime="application/pdf"
    )

    # ------------------------------------------------------------
    # DATA PREVIEW
    # ------------------------------------------------------------

    st.markdown(
        '<div class="section-title">Cleaned data preview</div>',
        unsafe_allow_html=True
    )

    st.dataframe(data.head(20), use_container_width=True)

    # ------------------------------------------------------------
    # SAVE EMPLOYEE-LEVEL RESULTS (fresh upload only)
    # ------------------------------------------------------------

    if uploaded_file is not None:

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

            # A "View →" button on the Retention Action Center page sets
            # this before switching pages so the search box is
            # pre-filled with that employee's ID - same trick used for
            # pending_nav above, since a widget's value can't be set
            # directly once it's already been drawn once.
            if st.session_state.get("pending_employee_search"):
                st.session_state["employee_search_query"] = (
                    st.session_state.pending_employee_search
                )
                st.session_state.pending_employee_search = None

            search_query = st.text_input(
                "Search by employee ID, department, or other details",
                placeholder="Type to search…",
                key="employee_search_query"
            )

            # Search across the ID column plus any text/category columns,
            # so typing a department name, job level, etc. also works -
            # not just an exact ID.
            searchable_columns = [id_column] + [
                column for column in results_df.columns
                if is_text_column(results_df[column])
                and column not in [id_column, "RiskLevel"]
            ]

            if search_query:

                query_lower = search_query.strip().lower()

                match_mask = results_df[searchable_columns].apply(
                    lambda col: col.astype(str).str.lower().str.contains(
                        query_lower, na=False
                    )
                ).any(axis=1)

                filtered_df = results_df[match_mask]

            else:
                filtered_df = results_df

            if filtered_df.empty:

                st.warning(
                    f"No employees match \"{search_query}\". Try a "
                    f"different search term."
                )
                st.stop()

            if search_query:
                st.caption(
                    f"{len(filtered_df)} employee"
                    f"{'s' if len(filtered_df) != 1 else ''} match."
                )

            selected_id = st.selectbox(
                "Select an employee",
                options=filtered_df[id_column].astype(str).tolist()
            )

            employee_row = results_df[
                results_df[id_column].astype(str) == selected_id
            ].iloc[0]

            detail_columns = [
                column for column in st.session_state.data_columns
                if column not in [target_column, id_column]
            ][:10]

            profile_col, gauge_col, summary_col = st.columns([2, 2, 2])

            # ---------------------------------------------
            # EMPLOYEE CARD (left)
            # ---------------------------------------------

            with profile_col:

                initials = "".join(
                    [c for c in str(selected_id) if c.isalnum()]
                )[:2].upper()

                profile_html = (
                    '<div class="feature-card">'
                    '<div style="display:flex; align-items:center; '
                    'gap:12px; margin-bottom:14px;">'
                    '<div style="width:46px; height:46px; border-radius:50%; '
                    'background:linear-gradient(135deg,#10B981,#0D9488); '
                    'display:flex; align-items:center; justify-content:center; '
                    'font-weight:700; color:#0A0B0A; font-family:\'Space '
                    'Grotesk\',sans-serif;">'
                    f'{initials}</div>'
                    f'<div><h4 style="margin:0;">{selected_id}</h4>'
                    f'<p style="margin:0;">{risk_badge_html(employee_row["RiskLevel"])}'
                )

                # A "primary cause" badge next to the risk level - the
                # same category shown for this employee's group on the
                # Retention Action Center page, so the two pages agree.
                if not pattern_df.empty:
                    employee_cause = get_employee_primary_cause(
                        employee_row, pattern_df
                    )
                    if employee_cause:
                        profile_html += (
                            f' {cause_badge_html(employee_cause)}'
                        )

                profile_html += (
                    '</p></div>'
                    '</div>'
                )

                for column in detail_columns[:6]:
                    profile_html += (
                        f'<p style="margin:4px 0; font-size:13px;">'
                        f'<span style="color:#6B726F;">{column}:</span> '
                        f'<span style="color:#1A1D1C;">'
                        f'{employee_row[column]}</span></p>'
                    )

                profile_html += '</div>'

                st.markdown(profile_html, unsafe_allow_html=True)

            # ---------------------------------------------
            # RISK GAUGE (middle)
            # ---------------------------------------------

            with gauge_col:

                risk_score = employee_row["RiskScore"]
                gauge_color = {
                    "High risk": "#EF4444",
                    "Medium risk": "#F59E0B",
                    "Low risk": "#34D399"
                }.get(employee_row["RiskLevel"], "#8A928F")

                gauge_fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=risk_score,
                    number={
                        "suffix": "%",
                        "font": {"color": "#1A1D1C", "size": 36}
                    },
                    gauge={
                        "axis": {
                            "range": [0, 100],
                            "tickcolor": "#5B635F",
                            "tickfont": {"color": "#5B635F"}
                        },
                        "bar": {"color": gauge_color},
                        "bgcolor": "#F0F4F2",
                        "borderwidth": 0,
                        "steps": [
                            {"range": [0, 33], "color": "rgba(16,185,129,0.15)"},
                            {"range": [33, 66], "color": "rgba(245,158,11,0.15)"},
                            {"range": [66, 100], "color": "rgba(239,68,68,0.15)"},
                        ],
                    }
                ))

                gauge_fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#1A1D1C"),
                    height=230,
                    margin=dict(t=30, b=10, l=30, r=30)
                )

                st.plotly_chart(gauge_fig, use_container_width=True)
                st.markdown(
                    f'<p style="text-align:center; color:#8A928F; '
                    f'font-size:13px; margin-top:-10px;">Chance of '
                    f'leaving</p>',
                    unsafe_allow_html=True
                )

            # ---------------------------------------------
            # QUICK SUMMARY (right)
            # ---------------------------------------------

            with summary_col:

                summary_html = (
                    '<div class="feature-card">'
                    '<h4 style="margin-top:0;">Quick summary</h4>'
                    f'<p style="font-size:13px;">'
                    f'<span style="color:#6B726F;">Risk score:</span> '
                    f'<span style="color:{gauge_color}; font-weight:600;">'
                    f'{risk_score:.0f} / 100</span></p>'
                )

                if not pattern_df.empty:
                    top_summary_features = pattern_df["Feature"].head(3).tolist()
                    for feature in top_summary_features:
                        if feature in employee_row.index:
                            summary_html += (
                                f'<p style="font-size:13px;">'
                                f'<span style="color:#6B726F;">{feature}:'
                                f'</span> <span style="color:#1A1D1C;">'
                                f'{employee_row[feature]:.1f}</span></p>'
                            )

                summary_html += '</div>'

                st.markdown(summary_html, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # ---------------------------------------------
            # TOP FACTORS + PERSONALIZED RECOMMENDATIONS
            # ---------------------------------------------

            if not pattern_df.empty:

                st.markdown(
                    '<div class="section-title">Key insights</div>',
                    unsafe_allow_html=True
                )

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

                factor_col1, factor_col2, factor_col3 = st.columns(3)
                factor_cols = [factor_col1, factor_col2, factor_col3]

                for i, feature in enumerate(top_pattern_features):
                    if feature in employee_row.index and i < 3:

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
                            "closer to employees who left"
                            if closer_to_left
                            else "closer to employees who stayed"
                        )

                        with factor_cols[i]:
                            st.markdown(
                                '<div class="feature-card">'
                                f'<h4 style="margin-top:0;">{feature}</h4>'
                                f'<p>This employee: <strong>{employee_value:.1f}'
                                f'</strong><br>({direction})<br>'
                                f'Left avg: {left_avg:.1f}<br>'
                                f'Stayed avg: {stayed_avg:.1f}</p>'
                                '</div>',
                                unsafe_allow_html=True
                            )

                st.markdown("<br>", unsafe_allow_html=True)

                # ------------------------------------------------
                # PERSONALIZED RECOMMENDATION FOR THIS EMPLOYEE
                # ------------------------------------------------

                st.markdown(
                    '<div class="section-title">Recommended management '
                    'actions</div>',
                    unsafe_allow_html=True
                )

                if employee_row["RiskLevel"] == "Low risk":

                    st.markdown(
                        '<div class="feature-card">No urgent action '
                        'needed based on this data. Continue regular '
                        'check-ins as part of normal management '
                        'practice.</div>',
                        unsafe_allow_html=True
                    )

                elif risk_driving_factors:

                    urgency_text = (
                        "This employee is flagged as high risk - the "
                        "factors below are likely contributing now, and "
                        "acting sooner rather than later matters."
                        if employee_row["RiskLevel"] == "High risk"
                        else
                        "This employee is at moderate risk - not "
                        "urgent, but worth addressing before it "
                        "escalates."
                    )

                    st.write(urgency_text)

                    for number, feature in enumerate(
                        risk_driving_factors, start=1
                    ):

                        details = get_recommendation_details(feature)

                        steps_html = "".join(
                            f'<p style="margin:4px 0;">• {step}</p>'
                            for step in details["steps"]
                        )

                        st.markdown(
                            '<div class="feature-card" style="margin-bottom:10px;">'
                            '<div style="display:flex; gap:12px;">'
                            '<div style="width:28px; height:28px; '
                            'border-radius:50%; background:rgba(16,185,129,0.15); '
                            'color:#047857; display:flex; align-items:center; '
                            'justify-content:center; font-weight:700; '
                            'flex-shrink:0;">'
                            f'{number}</div>'
                            f'<div><h4 style="margin:0 0 4px 0;">{feature}'
                            f'</h4><p style="margin:0 0 8px 0;">'
                            f'{details["why"]}</p>{steps_html}</div>'
                            '</div></div>',
                            unsafe_allow_html=True
                        )

                    st.caption(
                        "These suggestions are based on the factors "
                        "most associated with this specific employee's "
                        "risk score, not the company-wide averages "
                        "shown in the Analyze page. Use them as a "
                        "starting point for a conversation, not a "
                        "final decision."
                    )

                else:
                    st.markdown(
                        '<div class="feature-card">This employee\'s '
                        'risk score isn\'t clearly explained by the '
                        'top overall factors - a direct check-in is '
                        'the best next step.</div>',
                        unsafe_allow_html=True
                    )

        # ----------------------------------------------------
        # TAB: FULL RANKING
        # ----------------------------------------------------

        with ranking_tab:

            st.subheader("All employees ranked by risk")

            ranked_columns = [id_column, "RiskScore", "RiskLevel"]

            if department_column and department_column in results_df.columns:
                ranked_columns.insert(1, department_column)

            ranked_table = results_df[ranked_columns].copy()

            # "Cause" column - the same primary-cause category shown on
            # the Retention Action Center page, so someone scanning the
            # whole roster can spot a pattern (e.g. most of one
            # department's risk being compensation-driven) without
            # opening each employee individually. Skipped when there
            # aren't enough numeric patterns to base it on.
            CAUSE_DISPLAY = {
                cause: f'{meta["icon"]} {cause}'
                for cause, meta in CAUSE_META.items()
            }

            if not pattern_df.empty:
                raw_causes = results_df.apply(
                    lambda row: (
                        get_employee_primary_cause(row, pattern_df)
                        or "Unclear"
                    ),
                    axis=1
                )
                ranked_table["Cause"] = raw_causes.map(
                    lambda c: CAUSE_DISPLAY.get(c, CAUSE_DISPLAY["Unclear"])
                )
            else:
                ranked_table["Cause"] = "—"

            ranked_table = ranked_table.sort_values(
                "RiskScore", ascending=False
            )

            def color_risk_level(value):
                colors = {
                    "High risk": "color: #DC2626; font-weight: 600;",
                    "Medium risk": "color: #B45309; font-weight: 600;",
                    "Low risk": "color: #047857; font-weight: 600;"
                }
                return colors.get(value, "")

            def color_cause(value):
                for cause, meta in CAUSE_META.items():
                    if value == CAUSE_DISPLAY.get(cause):
                        return f'color: {meta["color"]}; font-weight: 600;'
                return ""

            try:
                styled_table = (
                    ranked_table.style
                    .format({"RiskScore": "{:.0f}%"})
                    .map(color_risk_level, subset=["RiskLevel"])
                    .map(color_cause, subset=["Cause"])
                )
            except AttributeError:
                # Older pandas versions use .applymap() instead of
                # .map() on a Styler object - fall back to that if
                # .map() isn't available in this environment.
                styled_table = (
                    ranked_table.style
                    .format({"RiskScore": "{:.0f}%"})
                    .applymap(color_risk_level, subset=["RiskLevel"])
                    .applymap(color_cause, subset=["Cause"])
                )

            st.dataframe(
                styled_table,
                use_container_width=True,
                hide_index=True
            )

            st.caption(
                "Cause is the single factor most associated with each "
                "employee's own risk score - useful for spotting a "
                "pattern across many people, not a diagnosis for any "
                "one of them."
            )


# ------------------------------------------------------------
# PAGE: RETENTION ACTION CENTER
# ------------------------------------------------------------
# Groups every high-risk employee by the single strongest factor
# behind their risk (career growth, workload/overtime, compensation,
# job satisfaction, or other) so a manager can see where the biggest
# pockets of risk are and act on a pattern rather than one employee
# at a time. Clicking a group's "View →" jumps straight into the
# Employee Lookup page for that person.
# ------------------------------------------------------------

elif page == "Retention Action Center":

    st.markdown('<div class="page-icon-badge">🎯</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-title">Retention action center</div>',
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

        st.caption(
            "High-risk employees, grouped by the single factor most "
            "likely driving their risk score - so you can spot and act "
            "on a pattern affecting several people, instead of "
            "reviewing everyone one at a time."
        )

        # Optional department filter - lets a manager check whether
        # one team's risk is mostly one cause (e.g. Sales skews
        # compensation, Engineering skews workload) instead of only
        # seeing the company-wide mix.
        selected_department = "All departments"

        if department_column and department_column in results_df.columns:

            department_options = ["All departments"] + sorted(
                results_df[department_column]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            selected_department = st.selectbox(
                "Filter by department",
                department_options,
                key="action_center_department_filter"
            )

        if pattern_df.empty:

            st.info(
                "There isn't enough numeric data in this dataset to "
                "group causes. See the Analyze page for details on "
                "what was found."
            )

        else:

            high_risk_df = results_df[
                results_df["RiskLevel"] == "High risk"
            ].copy()

            if selected_department != "All departments" and department_column:
                high_risk_df = high_risk_df[
                    high_risk_df[department_column].astype(str)
                    == selected_department
                ]

            department_phrase = (
                f" in {selected_department}"
                if selected_department != "All departments"
                else ""
            )

            if high_risk_df.empty:

                st.success(
                    f"No high-risk employees{department_phrase} right "
                    f"now - nothing urgent to act on."
                )

            else:

                high_risk_df["Cause"] = high_risk_df.apply(
                    lambda row: (
                        get_employee_primary_cause(row, pattern_df)
                        or "Unclear"
                    ),
                    axis=1
                )

                cause_counts = high_risk_df["Cause"].value_counts()

                st.markdown(
                    f'<div class="section-title">{len(high_risk_df)} '
                    f'high-risk employees{department_phrase}, grouped by '
                    f'likely cause</div>',
                    unsafe_allow_html=True
                )

                bar_fig = go.Figure(go.Bar(
                    x=cause_counts.values.tolist(),
                    y=cause_counts.index.tolist(),
                    orientation="h",
                    marker=dict(
                        color=[
                            CAUSE_META.get(cause, CAUSE_META["Unclear"])["color"]
                            for cause in cause_counts.index
                        ]
                    ),
                    text=cause_counts.values.tolist(),
                    textposition="outside"
                ))

                bar_fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#1A1D1C"),
                    height=max(220, 70 * len(cause_counts)),
                    margin=dict(t=10, b=10, l=10, r=50),
                    xaxis=dict(showgrid=False, zeroline=False),
                    yaxis=dict(autorange="reversed")
                )

                st.plotly_chart(bar_fig, use_container_width=True)

                st.markdown("<br>", unsafe_allow_html=True)

                st.markdown(
                    '<div class="section-title">Click a cause to see '
                    'who\'s in it</div>',
                    unsafe_allow_html=True
                )

                if "selected_cause" not in st.session_state:
                    st.session_state.selected_cause = cause_counts.index[0]

                cause_cols = st.columns(len(cause_counts))

                for col, (cause, count) in zip(cause_cols, cause_counts.items()):

                    meta = CAUSE_META.get(cause, CAUSE_META["Unclear"])
                    is_selected = st.session_state.selected_cause == cause

                    with col:
                        if st.button(
                            f"{meta['icon']} {cause} · {count}",
                            key=f"cause_btn_{cause}",
                            use_container_width=True,
                            type="primary" if is_selected else "secondary"
                        ):
                            st.session_state.selected_cause = cause
                            st.rerun()

                selected_cause = st.session_state.selected_cause

                if selected_cause not in cause_counts.index:
                    selected_cause = cause_counts.index[0]
                    st.session_state.selected_cause = selected_cause

                meta = CAUSE_META.get(selected_cause, CAUSE_META["Unclear"])

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(
                    f'<div class="section-title">{meta["icon"]} '
                    f'{selected_cause} - {cause_counts[selected_cause]} '
                    f'employees</div>',
                    unsafe_allow_html=True
                )

                cause_group_df = high_risk_df[
                    high_risk_df["Cause"] == selected_cause
                ].sort_values("RiskScore", ascending=False)

                for _, emp_row in cause_group_df.iterrows():

                    emp_id = emp_row[id_column]

                    dept_text = ""
                    if (
                        department_column
                        and department_column in emp_row.index
                    ):
                        dept_text = f' · {emp_row[department_column]}'

                    info_col, action_col = st.columns([5, 1])

                    with info_col:
                        st.markdown(
                            '<div class="feature-card" style="margin-bottom:8px; '
                            'display:flex; justify-content:space-between; '
                            'align-items:center;">'
                            f'<span><strong>{emp_id}</strong>{dept_text}</span>'
                            f'<span>{risk_badge_html(emp_row["RiskLevel"])} '
                            f'<span style="color:#6B726F; font-size:13px;">'
                            f'{emp_row["RiskScore"]:.0f}%</span></span>'
                            '</div>',
                            unsafe_allow_html=True
                        )

                    with action_col:
                        if st.button(
                            "View →",
                            key=f"view_{selected_cause}_{emp_id}",
                            use_container_width=True
                        ):
                            st.session_state.pending_employee_search = str(emp_id)
                            st.session_state.pending_nav = "Employee Lookup"
                            st.rerun()

                st.caption(
                    "Causes are inferred from the numeric factors that "
                    "most separate employees who stayed from employees "
                    "who left in this dataset, applied to each "
                    "employee's own values. Use this as a starting "
                    "point for where to look first, not a final "
                    "diagnosis."
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

        # ----------------------------------------------------
        # CAUSE TRENDS OVER TIME
        # ----------------------------------------------------
        # Re-derives each past analysis's cause breakdown (the same
        # categories used on the Retention Action Center) so it's
        # possible to see whether a specific problem - say, workload
        # - is growing or easing over successive uploads, rather than
        # only ever seeing today's snapshot.

        MAX_TREND_POINTS = 15

        trend_entries = list(reversed(history[:MAX_TREND_POINTS]))

        trend_rows = []

        for entry in trend_entries:

            cause_counts_for_entry = compute_cause_counts_for_history_entry(
                entry
            )

            if cause_counts_for_entry is None:
                continue

            date_label = str(entry.get("created_at", ""))[:10]
            trend_row = {"Date": date_label}
            trend_row.update(cause_counts_for_entry)
            trend_rows.append(trend_row)

        if len(trend_rows) == 1:

            st.info(
                "Analyze at least one more dataset to start seeing "
                "cause trends over time."
            )

        elif len(trend_rows) >= 2:

            trend_df = pd.DataFrame(trend_rows).fillna(0)

            st.markdown(
                '<div class="section-title">Cause trends over time'
                '</div>',
                unsafe_allow_html=True
            )

            st.caption(
                "How many high-risk employees fell into each cause "
                "category at the time of each past analysis - useful "
                "for spotting whether a specific problem is growing "
                "or easing, rather than only seeing today's snapshot."
            )

            trend_fig = go.Figure()

            for cause in CAUSE_CATEGORIES + ["Unclear"]:

                if cause in trend_df.columns:

                    meta = CAUSE_META.get(cause, CAUSE_META["Unclear"])

                    trend_fig.add_trace(go.Scatter(
                        x=trend_df["Date"],
                        y=trend_df[cause],
                        mode="lines+markers",
                        name=cause,
                        line=dict(color=meta["color"], width=2.5),
                        marker=dict(size=7)
                    ))

            trend_fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#1A1D1C"),
                height=340,
                margin=dict(t=20, b=10, l=10, r=10),
                legend=dict(orientation="h", y=-0.25),
                xaxis=dict(showgrid=False, title="Analysis date"),
                yaxis=dict(
                    showgrid=True,
                    gridcolor="#E2E8E5",
                    title="High-risk employees",
                    rangemode="tozero"
                )
            )

            st.plotly_chart(trend_fig, use_container_width=True)

            st.caption(
                "Based on your most recent "
                f"{len(trend_rows)} analyses. Runs saved before this "
                "feature was added may not appear."
            )

            st.markdown("<br>", unsafe_allow_html=True)

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
        "random forest model to distinguish employees who left from "
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
