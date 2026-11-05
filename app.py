
import streamlit as st

from auth import register_user, login_user

st.set_page_config(
    page_title="LLM Chatbot Comparison",
    page_icon="🤖",
    layout="wide"
)

# -------------------- SESSION STATE --------------------

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "user" not in st.session_state:
    st.session_state.user = None


# -------------------- WELCOME PAGE --------------------

def show_welcome():
    st.title("🤖 LLM Chatbot Comparison")

    st.write(
        "Explore and compare a baseline chatbot with an "
        "LLM-preprocessed chatbot."
    )

    st.subheader("Your Workspace")

    col1, col2 = st.columns(2)

    with col1:
        st.info(
            "**Chatbot 1 — Baseline**\n\n"
            "Ask questions without the additional "
            "LLM-based preprocessing step."
        )

    with col2:
        st.info(
            "**Chatbot 2 — Preprocessed**\n\n"
            "Ask questions using your existing "
            "preprocessing pipeline."
        )

    st.divider()
    st.write("Register or log in to access your workspace.")


# -------------------- LOGIN AND REGISTRATION --------------------

def show_auth():
    login_tab, register_tab = st.tabs(["Login", "Register"])

    with login_tab:
        st.subheader("Welcome Back")

        with st.form("login_form"):
            identifier = st.text_input("Username or email")
            password = st.text_input(
                "Password",
                type="password"
            )

            login_submitted = st.form_submit_button(
                "Login",
                use_container_width=True
            )

        if login_submitted:
            if not identifier.strip() or not password:
                st.error("Enter your username/email and password.")
            else:
                success, user = login_user(identifier, password)

                if success:
                    st.session_state.logged_in = True
                    st.session_state.user = user
                    st.rerun()
                else:
                    st.error("Invalid username/email or password.")

    with register_tab:
        st.subheader("Create an Account")

        with st.form("register_form"):
            username = st.text_input("Choose a username")
            email = st.text_input("Email address")
            new_password = st.text_input(
                "Create a password",
                type="password"
            )
            confirm_password = st.text_input(
                "Confirm password",
                type="password"
            )

            register_submitted = st.form_submit_button(
                "Register",
                use_container_width=True
            )

        if register_submitted:
            if new_password != confirm_password:
                st.error("The passwords do not match.")
            else:
                success, message = register_user(
                    username,
                    email,
                    new_password
                )

                if success:
                    st.success(message)
                    st.info("Select the Login tab to sign in.")
                else:
                    st.error(message)


# -------------------- DASHBOARD --------------------

def dashboard():
    user = st.session_state.user

    st.title("🏠 Dashboard")
    st.write(f"Welcome, **{user['username']}**!")

    st.write(
        "Choose a chatbot or evaluation tool to continue."
    )

    st.divider()
    st.subheader("Chatbots")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 💬 Chatbot 1")
        st.write("Baseline chatbot without LLM preprocessing.")

        st.page_link(
            "chatbot_1_baseline/app.py",
            label="Open Baseline Chatbot",
            icon="💬",
            use_container_width=True
        )

    with col2:
        st.markdown("### 🧠 Chatbot 2")
        st.write("Chatbot with LLM-based preprocessing.")

        st.page_link(
            "chatbot_2_preprocessed/app.py",
            label="Open Preprocessed Chatbot",
            icon="🧠",
            use_container_width=True
        )

    st.divider()
    st.subheader("Evaluation and Comparison")

    col3, col4 = st.columns(2)

    with col3:
        st.page_link(
            "evaluation/dashboard.py",
            label="Evaluation Dashboard",
            icon="📊",
            use_container_width=True
        )

    with col4:
        st.page_link(
            "evaluation/compare_results.py",
            label="Compare Results",
            icon="📈",
            use_container_width=True
        )

    st.info(
        "The five-model comparison can be integrated with your "
        "existing shared LLM connections in a later step."
    )


# -------------------- LOGOUT --------------------

def logout():
    st.session_state.logged_in = False
    st.session_state.user = None
    st.rerun()


# -------------------- APPLICATION ENTRY POINT --------------------

if not st.session_state.logged_in:
    show_welcome()
    show_auth()

else:
    # Every page has a unique URL pathname.
    pages = [
        st.Page(
            dashboard,
            title="Dashboard",
            icon="🏠",
            default=True,
            url_path="dashboard"
        ),
        st.Page(
            "chatbot_1_baseline/app.py",
            title="Baseline Chatbot",
            icon="💬",
            url_path="baseline-chatbot"
        ),
        st.Page(
            "chatbot_2_preprocessed/app.py",
            title="Preprocessed Chatbot",
            icon="🧠",
            url_path="preprocessed-chatbot"
        ),
        st.Page(
            "evaluation/dashboard.py",
            title="Evaluation Dashboard",
            icon="📊",
            url_path="evaluation-dashboard"
        ),
        st.Page(
            "evaluation/compare_results.py",
            title="Compare Results",
            icon="📈",
            url_path="compare-results"
        )
    ]

    with st.sidebar:
        st.title("🤖 LLM Workspace")
        st.write(
            f"Logged in as **{st.session_state.user['username']}**"
        )

        st.divider()

        if st.button("Logout", use_container_width=True):
            logout()

    navigation = st.navigation(pages)
    navigation.run()
