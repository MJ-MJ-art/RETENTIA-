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
import plotly.graph_objects as go

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

            .liquid-label {
                position: absolute;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                font-family: 'Space Grotesk', sans-serif;
                font-size: 44px;
                font-weight: 700;
                color: #F2F4F3;
                z-index: 2;
                text-shadow: 0 2px 8px rgba(0, 0, 0, 0.4);
            }

            .splash-caption {
                margin-top: 22px;
                color: #8A928F;
                font-size: 14px;
                letter-spacing: 2px;
                text-transform: uppercase;
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
                <div class="liquid-label">R</div>
            </div>
            <div class="splash-caption">Retentia</div>
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

        bars_html = '<div class="feature-card">'

        for _, row in top_features.iterrows():

            pct = (
                (row["Importance"] / max_importance * 100)
                if max_importance > 0 else 0
            )

            bars_html += f'''
            <div style="margin-bottom:14px;">
                <div style="display:flex; justify-content:space-between;
                            font-size:13px; color:#F2F4F3; margin-bottom:4px;">
                    <span>{row["Feature"]}</span>
                    <span style="color:#34D399;">{pct:.0f}%</span>
                </div>
                <div style="background:#1F2422; border-radius:6px; height:8px;">
                    <div style="background:linear-gradient(90deg,#0D9488,#34D399);
                                width:{pct:.0f}%; height:8px; border-radius:6px;">
                    </div>
                </div>
            </div>
            '''

        bars_html += '</div>'

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

            selected_id = st.selectbox(
                "Select an employee (type to search)",
                options=results_df[id_column].astype(str).tolist()
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
