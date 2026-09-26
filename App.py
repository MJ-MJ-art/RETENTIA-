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
    id_column
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
                    <p style="text-align:center; color:#8A928F; margin-top:14px;">
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
            textfont=dict(color="#F2F4F3", size=12)
        )])

        donut_fig.update_layout(
            showlegend=True,
            legend=dict(font=dict(color="#F2F4F3")),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            annotations=[dict(
                text=f"{(at_risk_count/len(results_df)*100):.1f}%<br>"
                     f"High Risk",
                x=0.5, y=0.5,
                font=dict(size=18, color="#F2F4F3"),
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
                f'font-size:13px; color:#F2F4F3; margin-bottom:4px;">'
                f'<span>{row["Feature"]}</span>'
                f'<span style="color:#34D399;">{pct:.0f}%</span>'
                f'</div>'
                f'<div style="background:#1F2422; border-radius:6px; height:8px;">'
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
        id_column=id_column
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

            search_query = st.text_input(
                "Search by employee ID, department, or other details",
                placeholder="Type to search…"
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
                    '</p></div>'
                    '</div>'
                )

                for column in detail_columns[:6]:
                    profile_html += (
                        f'<p style="margin:4px 0; font-size:13px;">'
                        f'<span style="color:#6B726F;">{column}:</span> '
                        f'<span style="color:#F2F4F3;">'
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
                        "font": {"color": "#F2F4F3", "size": 36}
                    },
                    gauge={
                        "axis": {
                            "range": [0, 100],
                            "tickcolor": "#8A928F",
                            "tickfont": {"color": "#8A928F"}
                        },
                        "bar": {"color": gauge_color},
                        "bgcolor": "#131615",
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
                    font=dict(color="#F2F4F3"),
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
                                f'</span> <span style="color:#F2F4F3;">'
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
                            'color:#34D399; display:flex; align-items:center; '
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
