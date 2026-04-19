import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

from data_handler import load_and_split_data
from preprocessor import fit_preprocessors, apply_preprocessors
from model import train_model
from evaluator import evaluate_model, get_feature_importance
from rag_store import seed_database
from agent import run_engagement_agent

# Configure the basic page settings. 
st.set_page_config(
    page_title="Player Churn Prediction",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom CSS injected into the Streamlit app.
# We use CSS variables like var(--text-color) and rgba() for backgrounds 
# to ensure the UI dynamically adapts to both Light and Dark themes.
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: var(--text-color);
        margin-bottom: 0.2rem;
        letter-spacing: -0.5px;
    }
    .sub-header {
        font-size: 1.1rem;
        color: var(--text-color);
        opacity: 0.7;
        margin-bottom: 2rem;
    }
    .result-card {
        border-radius: 8px;
        padding: 1.5rem;
        text-align: center;
        font-size: 1.25rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    /* Churn status card: Uses a subtle red background that works in both modes */
    .status-churn {
        background-color: rgba(220, 38, 38, 0.1);
        color: #DC2626;
        border: 1px solid #DC2626;
    }
    /* Retain status card: Uses a subtle green background that works in both modes */
    .status-retain {
        background-color: rgba(5, 150, 105, 0.1);
        color: #059669;
        border: 1px solid #059669;
    }
    .result-metric {
        font-size: 2.5rem;
        font-weight: 700;
        display: block;
        margin-top: 0.5rem;
        color: var(--text-color);
    }
</style>
""", unsafe_allow_html=True)

# Page Headers
st.markdown('<div class="main-header">Player Churn Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Identify at-risk players to proactively improve retention.</div>', unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def init_db():
    """Seeds the vector store once so the engagement agent can retrieve strategies."""
    seed_database()


init_db()

if "m2_strategy_output" not in st.session_state:
    st.session_state.m2_strategy_output = ""
if "m2_strategy_player_id" not in st.session_state:
    st.session_state.m2_strategy_player_id = None


def assign_risk_tier(risk_score: float) -> str:
    """Maps probability to a business-facing risk tier."""
    if risk_score >= 0.80:
        return "Critical"
    if risk_score >= 0.60:
        return "High"
    if risk_score >= 0.40:
        return "Moderate"
    return "Watchlist"


def derive_behavioral_signals(player_data: dict) -> list[str]:
    """Creates concise, rule-based diagnostics to complement LLM output."""
    sessions = float(player_data.get("SessionsPerWeek", 0))
    avg_session = float(player_data.get("AvgSessionDurationMinutes", 0))
    playtime = float(player_data.get("PlayTimeHours", 0))
    level = float(player_data.get("PlayerLevel", 1))
    achievements = float(player_data.get("AchievementsUnlocked", 0))
    purchases = float(player_data.get("InGamePurchases", 0))

    signals = []
    if sessions < 3:
        signals.append("Low weekly frequency suggests habit decay.")
    if avg_session < 15:
        signals.append("Very short sessions indicate reduced engagement depth.")
    if level > 0 and (achievements / level) < 0.6:
        signals.append("Low achievements-per-level may indicate progression friction.")
    if purchases == 0 and playtime >= 10:
        signals.append("High activity but no purchases indicates monetization conversion opportunity.")
    if sessions >= 7 and avg_session >= 120:
        signals.append("Potential binge pattern observed; burnout prevention should be considered.")

    if not signals:
        signals.append("No extreme behavioral anomaly detected; monitor trend over the next 7 days.")
    return signals


def intervention_priority(risk_score: float) -> tuple[str, str]:
    """Returns intervention urgency and target response SLA."""
    if risk_score >= 0.80:
        return "Immediate", "Within 24 hours"
    if risk_score >= 0.60:
        return "Urgent", "Within 72 hours"
    if risk_score >= 0.40:
        return "Planned", "Within 7 days"
    return "Observe", "Monitor only"


def build_strategy_brief(
    player_id,
    risk_score: float,
    risk_tier: str,
    priority: str,
    sla: str,
    behavioral_signals: list[str],
    strategy_text: str,
) -> str:
    """Builds an exportable markdown report for milestone 2 demos."""
    signal_block = "\n".join([f"- {item}" for item in behavioral_signals])
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return (
        f"# Retention Strategy Brief\n\n"
        f"Generated: {timestamp}\n\n"
        f"## Player Context\n"
        f"- Player Index: {player_id}\n"
        f"- Churn Risk Score: {risk_score:.2%}\n"
        f"- Risk Tier: {risk_tier}\n"
        f"- Intervention Priority: {priority}\n"
        f"- Response SLA: {sla}\n\n"
        f"## Behavioral Signals\n"
        f"{signal_block}\n\n"
        f"## AI Retention Plan\n"
        f"{strategy_text}\n"
    )

@st.cache_resource(show_spinner=False)
def load_and_train():
    """
    Loads the dataset, splits it, fits the preprocessing transformers, 
    and trains the predictive model.
    
    The @st.cache_resource decorator ensures this heavy computation 
    only runs once when the app starts, rather than every time the user 
    interacts with a widget.
    
    Returns:
        tuple: Containing the trained model, fitted scaler, fitted encoder, 
               feature lists, processed training/testing data, test labels,
               and raw test features for milestone 2 workflows.
    """
    data_splits, num_feats, cat_feats = load_and_split_data(
        "data/online_gaming_behavior_dataset.csv"
    )
    X_train, X_test, y_train, y_test = data_splits
    
    # Fit preprocessors only on the training data to prevent data leakage
    scaler, encoder = fit_preprocessors(X_train, num_feats, cat_feats)
    
    # Apply the fitted preprocessors to both train and test sets
    X_train_proc = apply_preprocessors(X_train, scaler, encoder, num_feats, cat_feats)
    X_test_proc  = apply_preprocessors(X_test,  scaler, encoder, num_feats, cat_feats)
    
    # Train the machine learning model
    model = train_model(X_train_proc, y_train)
    
    return model, scaler, encoder, num_feats, cat_feats, X_train_proc, X_test_proc, y_test, X_test

# Initialize the pipeline and show a spinner while the cache is building
with st.spinner("Initializing models..."):
    model, scaler, encoder, num_feats, cat_feats, X_train_proc, X_test_proc, y_test, X_test_raw = load_and_train()

# Define the tabs with Milestone 2 first for demo flow.
tab_m2, tab_eval, tab_predict = st.tabs([
    "Milestone 2: Engagement Agent",
    "Model Telemetry",
    "Predict Player Risk",
])

# --- TAB 1: MODEL TELEMETRY ---
with tab_eval:
    # Calculate performance metrics using the held-out test set
    metrics = evaluate_model(model, X_test_proc, y_test)

    st.markdown("### Performance Metrics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy",  f"{metrics['Accuracy']:.3f}")
    c2.metric("Precision", f"{metrics['Precision']:.3f}")
    c3.metric("Recall",    f"{metrics['Recall']:.3f}")
    c4.metric("AUC-ROC",   f"{metrics['AUC']:.3f}")

    st.divider()

    st.markdown("### Feature Importance")
    st.caption("Relative weight of factors driving churn decisions in the current model.")
    
    # Extract feature importance mapping from the trained model
    feature_names = list(X_train_proc.columns)
    importances = get_feature_importance(model, feature_names)

    # Build a horizontal bar chart for feature importances
    fig_imp = px.bar(
        x=importances.values,
        y=importances.index,
        orientation="h",
        labels={"x": "Importance Weight", "y": ""},
    )
    fig_imp.update_traces(
        marker_color="#3B82F6", # Flat blue color
        marker_line_width=0,
    )
    # Configure chart background to be transparent so it inherits the Streamlit theme
    fig_imp.update_layout(
        yaxis={"categoryorder": "total ascending"},
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=10, b=10),
        height=400,
    )
    st.plotly_chart(fig_imp, width="stretch", key="feature_importance_chart")


# --- TAB 2: PREDICT PLAYER RISK ---
with tab_predict:
    st.markdown("### Player Profile")
    st.caption("Enter the behavioral and demographic data of the player to assess their current churn risk.")

    # Form groups inputs together so the page doesn't refresh until 'Submit' is clicked
    with st.form("prediction_form", border=False):
        
        st.markdown("##### 1. Engagement Metrics")
        eng_1, eng_2, eng_3 = st.columns(3)
        with eng_1:
            play_time = st.number_input("Weekly Play Time (hrs)", min_value=0.0, max_value=168.0, value=10.0, step=0.5)
        with eng_2:
            sessions = st.number_input("Sessions per Week", min_value=1, max_value=50, value=5, step=1)
        with eng_3:
            avg_duration = st.number_input("Avg Session (mins)", min_value=1, max_value=600, value=90, step=5)

        st.markdown("##### 2. Game Progress & Monetization")
        prog_1, prog_2, prog_3 = st.columns(3)
        with prog_1:
            player_level = st.number_input("Player Level", min_value=1, max_value=99, value=20, step=1)
        with prog_2:
            achievements = st.number_input("Achievements Unlocked", min_value=0, max_value=500, value=20, step=1)
        with prog_3:
            in_game_purchases = st.selectbox("In-Game Purchases", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

        st.markdown("##### 3. Player Demographics & Preferences")
        dem_1, dem_2, dem_3, dem_4 = st.columns(4)
        with dem_1:
            age = st.number_input("Age", min_value=10, max_value=80, value=25, step=1)
        with dem_2:
            gender = st.selectbox("Gender", options=["Male", "Female"])
        with dem_3:
            location = st.selectbox("Location", options=["Asia", "Europe", "Other", "USA"])
        with dem_4:
            genre = st.selectbox("Preferred Genre", options=["Action", "RPG", "Simulation", "Sports", "Strategy"])
            difficulty = st.selectbox("Difficulty", options=["Easy", "Medium", "Hard"], index=1)

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("Run Risk Assessment", type="primary", width="stretch")

    # Execution block triggered only when the form is submitted
    if submitted:
        # 1. Construct a single-row DataFrame representing the new player
        input_df = pd.DataFrame([{
            "Age": age, "PlayTimeHours": play_time, "InGamePurchases": in_game_purchases,
            "SessionsPerWeek": sessions, "AvgSessionDurationMinutes": avg_duration,
            "PlayerLevel": player_level, "AchievementsUnlocked": achievements,
            "Gender": gender, "Location": location, "GameGenre": genre, "GameDifficulty": difficulty,
        }])

        # 2. Pass the raw input through the identical preprocessing pipeline used during training
        input_proc  = apply_preprocessors(input_df, scaler, encoder, num_feats, cat_feats)
        
        # 3. Generate predictions. 
        # predict() outputs the binary class (0 or 1)
        # predict_proba() outputs the probability array [prob_class_0, prob_class_1]
        prediction  = model.predict(input_proc)[0]
        churn_prob  = model.predict_proba(input_proc)[0][1]

        st.divider()
        res_col, gauge_col = st.columns([1, 1], gap="large")

        # Display text-based results
        with res_col:
            if prediction == 1:
                st.markdown(
                    f'<div class="result-card status-churn">'
                    f'High Risk of Churn'
                    f'<span class="result-metric">{churn_prob:.1%}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                st.error("Action Required: This player's engagement patterns suggest they are likely to abandon the game. Consider offering a targeted loyalty reward or adjusting the matchmaking difficulty.")
            else:
                st.markdown(
                    f'<div class="result-card status-retain">'
                    f'Healthy Engagement'
                    f'<span class="result-metric">{(1 - churn_prob):.1%}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                st.success("Stable: This player exhibits strong retention signals. No immediate intervention is required.")

            # Display the raw inputs. .astype(str) is used to prevent Apache Arrow serialization 
            # errors caused by mixing strings and numbers in a transposed dataframe column.
            with st.expander("View Submitted Profile", expanded=False):
                st.dataframe(input_df.astype(str).T.rename(columns={0: "Value"}), width="stretch")

        # Display the visual gauge chart
        with gauge_col:
            # Set the primary color of the gauge bar based on the binary prediction
            gauge_color = "#EF4444" if prediction == 1 else "#10B981"
            
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=churn_prob * 100,
                number={"suffix": "%", "font": {"size": 36}}, 
                title={"text": "Churn Probability", "font": {"size": 16}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1},
                    "bar":  {"color": gauge_color},
                    # A semi-transparent gray background for the track ensures it is 
                    # visible against both light and dark Streamlit themes.
                    "bgcolor": "rgba(128, 128, 128, 0.2)",
                    "borderwidth": 0,
                    "threshold": {
                        "line": {"color": gauge_color, "width": 2},
                        "thickness": 0.75,
                        "value": churn_prob * 100,
                    },
                },
            ))
            
            # Remove Plotly's default background colors to let Streamlit's theme show through
            fig_gauge.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                height=250,
                margin=dict(l=20, r=20, t=40, b=10),
            )
            st.plotly_chart(fig_gauge, width="stretch", key="gauge_chart")


# --- TAB 1: MILESTONE 2 ENGAGEMENT AGENT ---
with tab_m2:
    st.markdown("### Agentic Engagement Optimization")
    st.caption("Prioritize at-risk players and generate a personalized retention strategy using a LangGraph + RAG agent.")

    y_pred = model.predict(X_test_proc)
    y_prob = model.predict_proba(X_test_proc)[:, 1]

    results_df = X_test_raw.copy()
    results_df["Churn_Prediction"] = y_pred
    results_df["Risk_Score"] = y_prob
    results_df["Risk_Tier"] = results_df["Risk_Score"].apply(assign_risk_tier)
    at_risk_players = results_df[results_df["Churn_Prediction"] == 1]

    if not at_risk_players.empty:
        st.markdown("#### At-Risk Portfolio Snapshot")
        critical_count = int((at_risk_players["Risk_Tier"] == "Critical").sum())
        high_or_above = int((at_risk_players["Risk_Score"] >= 0.60).sum())
        avg_risk = float(at_risk_players["Risk_Score"].mean())

        m1, m2, m3 = st.columns(3)
        m1.metric("At-Risk Players", f"{len(at_risk_players)}")
        m2.metric("Critical Tier", f"{critical_count}")
        m3.metric("Avg Risk", f"{avg_risk:.2%}")

        st.caption(f"{high_or_above} players are High/Critical and should be prioritized this cycle.")

        preview_cols = [
            "Risk_Score",
            "Risk_Tier",
            "SessionsPerWeek",
            "AvgSessionDurationMinutes",
            "InGamePurchases",
            "GameGenre",
        ]
        st.dataframe(
            at_risk_players[preview_cols]
            .sort_values("Risk_Score", ascending=False)
            .head(12)
            .style.format({"Risk_Score": "{:.2%}"}),
            width="stretch",
        )

        player_id = st.selectbox(
            "Select an At-Risk Player (by Index) to generate a strategy:",
            at_risk_players.sort_values("Risk_Score", ascending=False).index,
        )

        selected_player_data = at_risk_players.loc[player_id].drop(["Churn_Prediction", "Risk_Score"]).to_dict()
        selected_risk_score = at_risk_players.loc[player_id, "Risk_Score"]
        selected_prediction = at_risk_players.loc[player_id, "Churn_Prediction"]
        selected_risk_tier = at_risk_players.loc[player_id, "Risk_Tier"]
        selected_priority, selected_sla = intervention_priority(float(selected_risk_score))
        behavioral_signals = derive_behavioral_signals(selected_player_data)

        with st.expander("View Player's Raw Metrics"):
            st.json(selected_player_data)
            r1, r2, r3 = st.columns(3)
            r1.metric("Risk Score", f"{selected_risk_score:.2%}")
            r2.metric("Risk Tier", selected_risk_tier)
            r3.metric("Priority", selected_priority)

        st.markdown("#### Rule-Based Diagnostic Signals")
        for signal in behavioral_signals:
            st.markdown(f"- {signal}")

        st.info(f"Intervention SLA: {selected_sla}")

        if st.button("Generate AI Retention Strategy", key="generate_strategy"):
            with st.spinner("Agent is retrieving strategies and generating plan..."):
                try:
                    strategy_output = run_engagement_agent(
                        player_data=selected_player_data,
                        risk_score=selected_risk_score,
                        prediction=int(selected_prediction),
                    )
                except Exception as exc:
                    strategy_output = (
                        "1. Summary: Strategy generation unavailable.\n"
                        "2. Analysis: Runtime exception occurred while invoking the engagement agent.\n"
                        "3. Plan: Trigger a manual campaign (daily login calendar + limited-time reward).\n"
                        "4. Refs: Fallback plan initiated by app runtime guard.\n"
                        "5. Disclaimer: Validate interventions with A/B testing before production rollout.\n\n"
                        f"Error details: {exc}"
                    )

                st.session_state.m2_strategy_output = strategy_output
                st.session_state.m2_strategy_player_id = player_id

        if (
            st.session_state.m2_strategy_output
            and st.session_state.m2_strategy_player_id == player_id
        ):
            strategy_output = st.session_state.m2_strategy_output

            st.success("Strategy Generated Successfully!")
            st.markdown(strategy_output)

            strategy_brief = build_strategy_brief(
                player_id=player_id,
                risk_score=float(selected_risk_score),
                risk_tier=selected_risk_tier,
                priority=selected_priority,
                sla=selected_sla,
                behavioral_signals=behavioral_signals,
                strategy_text=strategy_output,
            )
            st.download_button(
                "Download Strategy Brief (.md)",
                data=strategy_brief,
                file_name=f"retention_brief_player_{player_id}.md",
                mime="text/markdown",
                use_container_width=True,
            )
    else:
        st.success("No at-risk players found in the current test sample.")