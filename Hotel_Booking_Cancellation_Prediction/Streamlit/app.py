"""
Hotel Booking Cancellation Prediction — Streamlit App
------------------------------------------------------
Loads the artifacts saved at the end of the notebook:
    hotel_model.pkl     -> tuned XGBoost classifier
    preprocessor.pkl    -> fitted ColumnTransformer (scaling + one-hot)
    label_encoder.pkl   -> LabelEncoder for booking_status

Place these three files in the same folder as this script (or edit
ARTIFACTS_DIR below), then run:

    streamlit run app.py
"""

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from pathlib import Path
from datetime import date

# ----------------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Hotel Cancellation Predictor",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded",
)

ARTIFACTS_DIR = Path(__file__).parent

# ----------------------------------------------------------------------------
# Theme — Navy / Sky Blue / Purple / Pink
# ----------------------------------------------------------------------------
COLORS = {
    "bg_dark": "#071A35",
    "bg_darker": "#040D1F",
    "bg_card": "#0E2A52",
    "bg_card_alt": "#123568",
    "navy_border": "#2A4E85",
    "sky": "#4FC3F7",
    "sky_soft": "#9BE0FF",
    "purple": "#9B7FE8",
    "purple_soft": "#C6B6F5",
    "pink": "#F582BC",
    "pink_soft": "#FCC3E1",
    "text": "#F5F9FF",
    "text_dim": "#C4D3EC",
}

CUSTOM_CSS = f"""
<style>
.stApp {{
        background: radial-gradient(circle at 15% 0%, {COLORS['bg_card_alt']} 0%, {COLORS['bg_dark']} 45%, {COLORS['bg_darker']} 100%);
        color: {COLORS['text']};
    }}

    .stApp > header {{
        background-color: transparent !important;
    }}
    
    div[data-testid="stDecoration"] {{
        display: none;
    }}

    /* Sidebar - Increased Width */
    section[data-testid="stSidebar"] {{
        min-width: 400px !important;
        max-width: 450px !important;
        background: linear-gradient(180deg, {COLORS['bg_darker']} 0%, {COLORS['bg_card']} 100%);
        border-right: 1px solid {COLORS['navy_border']};
    }}
    section[data-testid="stSidebar"] * {{
        color: {COLORS['text']} !important;
    }}
    section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {{
        font-weight: 600;
        color: {COLORS['sky_soft']} !important;
        font-size: 0.92rem;
    }}

    h1, h2, h3, h4, p, span, label, .stMarkdown {{
        color: {COLORS['text']};
        font-family: 'Segoe UI', Arial, sans-serif;
    }}

    /* Header banner */
    .hero-banner {{
        background: linear-gradient(120deg, {COLORS['bg_card_alt']} 0%, {COLORS['bg_card']} 100%);
        border: 1px solid {COLORS['navy_border']};
        border-radius: 20px;
        padding: 28px 34px;
        margin-bottom: 22px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.4);
    }}
    .hero-title {{
        font-size: 3rem;
        font-weight: 900;
        line-height: 1.15;
        background: linear-gradient(90deg, {COLORS['sky']} 0%, {COLORS['purple']} 55%, {COLORS['pink']} 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }}
    .hero-subtitle {{
        color: {COLORS['text_dim']} !important;
        font-size: 1.1rem;
        margin-top: 8px;
    }}

    .gradient-divider {{
        height: 4px;
        border-radius: 10px;
        background: linear-gradient(90deg, {COLORS['sky']}, {COLORS['purple']}, {COLORS['pink']});
        margin: 6px 0 26px 0;
        border: none;
    }}

    .section-label {{
        color: {COLORS['sky_soft']} !important;
        font-weight: 800;
        font-size: 1rem;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        margin-bottom: 10px;
    }}

    /* Real Streamlit bordered containers, styled as glass cards */
    div[data-testid="stVerticalBlockBorderWrapper"] {{
        background: linear-gradient(160deg, {COLORS['bg_card']} 0%, {COLORS['bg_card_alt']} 100%);
        border: 1px solid {COLORS['navy_border']} !important;
        border-radius: 16px !important;
        box-shadow: 0 8px 24px rgba(0,0,0,0.35);
    }}

    .metric-box {{
        background: linear-gradient(160deg, {COLORS['bg_card_alt']} 0%, {COLORS['bg_card']} 100%);
        border: 1px solid {COLORS['navy_border']};
        border-radius: 14px;
        padding: 18px;
        text-align: center;
    }}
    .metric-value {{
        font-size: 2.1rem;
        font-weight: 800;
        color: {COLORS['sky']};
    }}
    .metric-label {{
        color: {COLORS['text_dim']} !important;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}

    .result-badge-cancel {{
        background: linear-gradient(90deg, {COLORS['pink']}, {COLORS['purple']});
        color: white !important;
        padding: 18px 24px;
        border-radius: 14px;
        font-size: 1.5rem;
        font-weight: 800;
        text-align: center;
        box-shadow: 0 6px 20px rgba(242,124,180,0.35);
    }}
    .result-badge-safe {{
        background: linear-gradient(90deg, {COLORS['sky']}, {COLORS['purple_soft']});
        color: {COLORS['bg_darker']} !important;
        padding: 18px 24px;
        border-radius: 14px;
        font-size: 1.5rem;
        font-weight: 800;
        text-align: center;
        box-shadow: 0 6px 20px rgba(79,195,247,0.35);
    }}

    .stButton>button {{
        background: linear-gradient(90deg, {COLORS['sky']} 0%, {COLORS['purple']} 55%, {COLORS['pink']} 100%);
        color: white !important;
        border: none;
        border-radius: 12px;
        padding: 12px 0;
        font-weight: 700;
        font-size: 1.05rem;
        width: 100%;
        transition: transform 0.15s ease;
    }}
    .stButton>button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(139,107,224,0.4);
    }}
    .stButton>button p {{
        color: white !important;
    }}

    div[data-baseweb="select"] > div {{
        background-color: {COLORS['bg_card']} !important;
        border-radius: 10px !important;
        border-color: {COLORS['navy_border']} !important;
    }}
    .stNumberInput input, .stDateInput input {{
        background-color: {COLORS['bg_card']} !important;
        color: {COLORS['text']} !important;
        border-radius: 10px !important;
        border-color: {COLORS['navy_border']} !important;
    }}

    .stAlert p {{
        color: {COLORS['bg_darker']} !important;
    }}

    footer, #MainMenu {{visibility: hidden;}}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Load artifacts (cached)
# ----------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load(ARTIFACTS_DIR / "hotel_model.pkl")
    preprocessor = joblib.load(ARTIFACTS_DIR / "preprocessor.pkl")
    label_encoder = joblib.load(ARTIFACTS_DIR / "label_encoder.pkl")
    return model, preprocessor, label_encoder


try:
    model, preprocessor, label_encoder = load_artifacts()
    artifacts_ok = True
except FileNotFoundError:
    artifacts_ok = False


SEASON_MAP = {
    12: "Winter", 1: "Winter", 2: "Winter",
    3: "Spring", 4: "Spring", 5: "Spring",
    6: "Summer", 7: "Summer", 8: "Summer",
    9: "Autumn", 10: "Autumn", 11: "Autumn",
}

MEAL_PLANS = ["Meal Plan 1", "Meal Plan 2", "Meal Plan 3", "Not Selected"]
ROOM_TYPES = [f"Room_Type {i}" for i in range(1, 8)]
MARKET_SEGMENTS = ["Online", "Offline", "Corporate", "Complementary", "Aviation"]


# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero-banner">
        <p class="hero-title">🏨 Hotel Booking Cancellation Predictor</p>
        <p class="hero-subtitle">Enter the booking details and get an instant cancellation-risk prediction, powered by a tuned XGBoost model.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if not artifacts_ok:
    st.error(
        "⚠️ Model files not found (hotel_model.pkl, preprocessor.pkl, label_encoder.pkl). "
        "Run the last cell of the notebook to save them, then place all three next to app.py."
    )
    st.stop()

# ----------------------------------------------------------------------------
# Sidebar — booking inputs
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<p class="section-label">👥 Guests & Stay</p>', unsafe_allow_html=True)
    no_of_adults = st.number_input("Number of adults", 0, 10, 2)
    no_of_children = st.number_input("Number of children", 0, 10, 0)
    no_of_weekend_nights = st.number_input("Weekend nights", 0, 20, 1)
    no_of_week_nights = st.number_input("Week nights", 0, 20, 2)

    st.markdown('<p class="section-label">🛏️ Room & Meal</p>', unsafe_allow_html=True)
    type_of_meal_plan = st.selectbox("Meal plan", MEAL_PLANS)
    room_type_reserved = st.selectbox("Room type", ROOM_TYPES)
    required_car_parking_space = st.selectbox("Requires parking space?", ["No", "Yes"])
    avg_price_per_room = st.slider("Average price per room ($)", 0.0, 550.0, 100.0, 1.0)

    st.markdown('<p class="section-label">📅 Arrival</p>', unsafe_allow_html=True)
    arrival_date_input = st.date_input(
        "Arrival date",
        value=date(2018, 7, 15),
        min_value=date(2017, 1, 1),
        max_value=date(2018, 12, 31),
    )
    arrival_year = arrival_date_input.year
    arrival_month = arrival_date_input.month
    arrival_day = arrival_date_input.day
    lead_time = st.slider("Lead time (days before arrival)", 0, 450, 50)

    st.markdown('<p class="section-label">📈 Guest History</p>', unsafe_allow_html=True)
    market_segment_type = st.selectbox("Market segment", MARKET_SEGMENTS)
    repeated_guest = st.selectbox("Repeated guest?", ["No", "Yes"])
    no_of_previous_cancellations = st.number_input("Previous cancellations", 0, 50, 0)
    no_of_previous_bookings_not_canceled = st.number_input("Previous completed bookings", 0, 100, 0)
    no_of_special_requests = st.slider("Number of special requests", 0, 5, 0)

    predict_clicked = st.button("🔮 Predict Cancellation")


# ----------------------------------------------------------------------------
# Build input row exactly like the notebook's feature engineering
# ----------------------------------------------------------------------------
def build_input_row():
    row = {
        "no_of_adults": no_of_adults,
        "no_of_children": no_of_children,
        "no_of_weekend_nights": no_of_weekend_nights,
        "no_of_week_nights": no_of_week_nights,
        "type_of_meal_plan": type_of_meal_plan,
        "required_car_parking_space": 1 if required_car_parking_space == "Yes" else 0,
        "room_type_reserved": room_type_reserved,
        "lead_time": lead_time,
        "arrival_year": arrival_year,
        "arrival_month": arrival_month,
        "arrival_day": arrival_day,
        "market_segment_type": market_segment_type,
        "repeated_guest": 1 if repeated_guest == "Yes" else 0,
        "no_of_previous_cancellations": no_of_previous_cancellations,
        "no_of_previous_bookings_not_canceled": no_of_previous_bookings_not_canceled,
        "avg_price_per_room": avg_price_per_room,
        "no_of_special_requests": no_of_special_requests,
    }
    row["arrival_season"] = SEASON_MAP[row["arrival_month"]]
    row["total_nights"] = row["no_of_weekend_nights"] + row["no_of_week_nights"]
    return pd.DataFrame([row])


# ----------------------------------------------------------------------------
# Main panel
# ----------------------------------------------------------------------------
left, right = st.columns([1.15, 1])

with left:
    with st.container(border=True):
        st.markdown('<p class="section-label">📋 Booking Summary</p>', unsafe_allow_html=True)
        summary_df = build_input_row().T.rename(columns={0: "Value"})
        st.dataframe(summary_df, use_container_width=True, height=440)

with right:
    if predict_clicked:
        input_df = build_input_row()
        X_input = preprocessor.transform(input_df)

        pred = model.predict(X_input)[0]
        proba = model.predict_proba(X_input)[0]
        pred_label = label_encoder.inverse_transform([pred])[0]
        cancel_idx = list(label_encoder.classes_).index("Canceled")
        cancel_prob = proba[cancel_idx]

        with st.container(border=True):
            st.markdown('<p class="section-label">🎯 Prediction Result</p>', unsafe_allow_html=True)

            if pred_label == "Canceled":
                st.markdown(
                    '<div class="result-badge-cancel">⚠️ Likely to be Canceled</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="result-badge-safe">✅ Likely Not Canceled</div>',
                    unsafe_allow_html=True,
                )

            st.write("")

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=cancel_prob * 100,
                number={"suffix": "%", "font": {"color": COLORS["text"], "size": 40}},
                title={"text": "Cancellation Probability", "font": {"color": COLORS["text_dim"], "size": 16}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": COLORS["text_dim"], "tickfont": {"color": COLORS["text_dim"]}},
                    "bar": {"color": COLORS["pink"]},
                    "bgcolor": COLORS["bg_card"],
                    "borderwidth": 1,
                    "bordercolor": COLORS["navy_border"],
                    "steps": [
                        {"range": [0, 33], "color": COLORS["sky"]},
                        {"range": [33, 66], "color": COLORS["purple_soft"]},
                        {"range": [66, 100], "color": COLORS["pink_soft"]},
                    ],
                    "threshold": {
                        "line": {"color": "white", "width": 3},
                        "thickness": 0.8,
                        "value": cancel_prob * 100,
                    },
                },
            ))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={"color": COLORS["text"]},
                height=280,
                margin=dict(t=40, b=10, l=20, r=20),
            )
            st.plotly_chart(fig, use_container_width=True)

            c1, c2 = st.columns(2)
            with c1:
                st.markdown(
                    f'<div class="metric-box"><div class="metric-value">{cancel_prob*100:.1f}%</div>'
                    f'<div class="metric-label">Cancel Probability</div></div>',
                    unsafe_allow_html=True,
                )
            with c2:
                st.markdown(
                    f'<div class="metric-box"><div class="metric-value">{(1-cancel_prob)*100:.1f}%</div>'
                    f'<div class="metric-label">Not Cancel Probability</div></div>',
                    unsafe_allow_html=True,
                )

    else:
        with st.container(border=True):
            st.markdown('<p class="section-label">👈 Get Started</p>', unsafe_allow_html=True)
            st.write("Fill in the booking details on the left sidebar, then click **Predict Cancellation** to see the result.")

# ----------------------------------------------------------------------------
# Dashboard Insights (Updated)
# ----------------------------------------------------------------------------
st.markdown('<hr class="gradient-divider">', unsafe_allow_html=True)
st.markdown('<p class="section-label">📊 Dashboard Insights</p>', unsafe_allow_html=True)

# Row 1 of Dashboard
dash_col1, dash_col2 = st.columns(2)
# Row 2 of Dashboard
dash_col3, dash_col4 = st.columns(2)

# التنسيق الموحد للخط ليظهر باللون البينك الفاتح ويكون أكبر وأوضح
chart_font = dict(color=COLORS["pink_soft"], size=15)

with dash_col1:
    with st.container(border=True):
        # 1. Bar chart for Stay Duration Breakdown
        fig_nights = go.Figure(go.Bar(
            x=["Weekend Nights", "Week Nights"],
            y=[no_of_weekend_nights, no_of_week_nights],
            marker_color=[COLORS["sky"], COLORS["purple"]],
            text=[no_of_weekend_nights, no_of_week_nights],
            textposition='auto',
            textfont=chart_font
        ))
        fig_nights.update_layout(
            title=dict(text="Stay Duration Breakdown", font=chart_font),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=chart_font,
            height=380
        )
        st.plotly_chart(fig_nights, use_container_width=True)

with dash_col2:
    with st.container(border=True):
        # 2. Donut chart for Guest Composition
        fig_guests = go.Figure(go.Pie(
            labels=["Adults", "Children"],
            values=[no_of_adults, no_of_children],
            hole=0.5,
            marker=dict(colors=[COLORS["pink"], COLORS["sky_soft"]]),
            textinfo='label+percent',
            textfont=chart_font
        ))
        fig_guests.update_layout(
            title=dict(text="Guest Composition", font=chart_font),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=chart_font,
            height=380
        )
        st.plotly_chart(fig_guests, use_container_width=True)

with dash_col3:
    with st.container(border=True):
        # 3. Bar chart for Previous Guest History
        fig_history = go.Figure(go.Bar(
            x=["Completed Bookings", "Previous Cancellations"],
            y=[no_of_previous_bookings_not_canceled, no_of_previous_cancellations],
            marker_color=[COLORS["sky_soft"], COLORS["pink"]],
            text=[no_of_previous_bookings_not_canceled, no_of_previous_cancellations],
            textposition='auto',
            textfont=chart_font
        ))
        fig_history.update_layout(
            title=dict(text="Guest History (Previous Bookings)", font=chart_font),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=chart_font,
            height=380
        )
        st.plotly_chart(fig_history, use_container_width=True)

with dash_col4:
    with st.container(border=True):
        # 4. Gauge indicator for Lead Time
        fig_lead = go.Figure(go.Indicator(
            mode="number+gauge",
            value=lead_time,
            title=dict(text="Lead Time (Days before arrival)", font=chart_font),
            number=dict(font=dict(color=COLORS["pink_soft"], size=45)),
            gauge=dict(
                axis=dict(range=[0, 450], tickcolor=COLORS["pink_soft"], tickfont=dict(color=COLORS["pink_soft"], size=12)),
                bar=dict(color=COLORS["purple"]),
                bgcolor=COLORS["bg_card"],
                borderwidth=1,
                bordercolor=COLORS["navy_border"],
            )
        ))
        fig_lead.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=chart_font,
            height=380
        )
        st.plotly_chart(fig_lead, use_container_width=True)

st.caption("Hotel Booking Cancellation Prediction · XGBoost · Built with Streamlit")