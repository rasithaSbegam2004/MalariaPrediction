import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import kneighbors_graph
from sklearn.exceptions import DataConversionWarning
import warnings
warnings.filterwarnings("ignore", category=DataConversionWarning)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, InputLayer
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics.pairwise import haversine_distances
from math import radians

st.set_page_config(page_title="Malaria Forecast + LSTM+GNN (Enhanced)", layout="wide")

# ---------------------------
# Config & Helpers
# ---------------------------
DISTRICT_COORDS = {
    "Chennai": [13.0827, 80.2707],
    "Coimbatore": [11.0168, 76.9558],
    "Salem": [11.6643, 78.1460],
    "Madurai": [9.9252, 78.1198],
    "Trichy": [10.7905, 78.7047],
    "Erode": [11.3424, 77.7275],
    "Vellore": [12.9165, 79.1325],
    "Tirunelveli": [8.7139, 77.7567],
    "Thanjavur": [10.7870, 79.1378],
    "Dindigul": [10.3624, 77.9695],
    "Kanchipuram": [12.8376, 79.7006],
    "Thoothukudi": [8.7642, 78.1348],
    "Nagapattinam": [10.7649, 79.8431],
    "Krishnagiri": [12.5307, 78.2130],
    "Dharmapuri": [12.1211, 78.1582],
}
DISTRIBUTOR_COORDS = {
    "North Zone": [13.0827, 80.2707],   # Chennai
    "West Zone": [11.0168, 76.9558],    # Coimbatore
    "South Zone": [9.9252, 78.1198],    # Madurai
    "East Zone": [10.7870, 79.1378],    # Thanjavur
}


SEQ_LEN = 12
EPOCHS = 80
BATCH = 16


def build_distributor_graph(districts):
    dist_coords = np.array([DISTRICT_COORDS[d] for d in districts])
    dist_coords = np.radians(dist_coords)

    dist_names = list(DISTRIBUTOR_COORDS.keys())
    hub_coords = np.array(list(DISTRIBUTOR_COORDS.values()))
    hub_coords = np.radians(hub_coords)

    # Distance matrix (district × distributor)
    dist_matrix = haversine_distances(dist_coords, hub_coords)

    # Convert distance → similarity (GNN edge weight)
    sim = 1 / (dist_matrix + 1e-6)
    sim = sim / sim.sum(axis=1, keepdims=True)  # normalize

    return sim, dist_names


def ensure_datetime(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ds"] = pd.to_datetime(df["ds"])
    return df.sort_values("ds")

def generate_enhanced_data(months=36):
    dates = pd.date_range("2022-01-01", periods=months, freq="MS")
    records = []
    rng = np.random.default_rng(42)
    for district in DISTRICT_COORDS.keys():
        baseline = rng.integers(40, 200)
        for d in dates:
            seasonal_cases = 60 * np.sin(2 * np.pi * (d.month - 4) / 12)
            noise_cases = rng.normal(0, 20)
            cases = max(0, int(baseline + seasonal_cases + noise_cases))
            rainfall = max(0, int(100*np.sin(2*np.pi*(d.month-6)/12) + rng.normal(50,20)))
            humidity = max(40, min(100, int(60+20*np.sin(2*np.pi*(d.month-5)/12) + rng.normal(0,5))))
            temperature = max(20, min(40, int(25+5*np.cos(2*np.pi*(d.month-3)/12) + rng.normal(0,2))))
            rdt = int(cases * 1.0)
            act = int(cases * 0.7)
            bed_nets = int(cases * 0.5)
            records.append({
                "ds": d,
                "district": district,
                "cases": cases,
                "RDT Kits": rdt,
                "ACT Doses": act,
                "Bed Nets": bed_nets,
                "Rainfall_mm": rainfall,
                "Humidity_%": humidity,
                "Temperature_C": temperature
            })
    return pd.DataFrame(records)

# ---------------------------
# Sidebar Navigation
# ---------------------------
st.sidebar.title("Malaria Forecasting (Enhanced LSTM + Spatial smoothing)")
page = st.sidebar.selectbox(
    "Choose page",
    [
        "Home",
        "Upload/Preview",
        "Train & Forecast",
        "Dashboard & Map",
        "Distributor & Resource Flow",
        "Seasonal Trend Analysis",
    ],
)

if "data" not in st.session_state:
    st.session_state["data"] = None
    st.session_state["forecast_table"] = None
    st.session_state["lstm_model"] = None
    st.session_state["adj_matrix"] = None

# ---------------------------
# HOME
# ---------------------------
if page == "Home":
    st.title("Malaria Demand Forecasting (Enhanced)")
    st.markdown("""
This enhanced prototype trains an LSTM on district-level malaria time series and applies a spatial smoothing step
(GNN-style) to reflect geographical correlation. It also tracks stock requirements and environmental features.

*Steps:*
1. Upload dataset or generate enhanced synthetic dataset.
2. Train LSTM model under Train & Forecast.
3. Explore forecasts, stocks, distributor zones, and seasonal patterns.
""")

# ---------------------------
# UPLOAD/PREVIEW
# ---------------------------
elif page == "Upload/Preview":
    st.header("Upload dataset (CSV) or generate enhanced synthetic dataset")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            must = {"ds", "cases"}
            if not must.issubset(set(df.columns)):
                st.error("CSV must include at least 'ds' and 'cases'. 'district' recommended.")
            else:
                df = ensure_datetime(df)
                df["district"] = df["district"].str.strip()
                st.session_state["data"] = df
                st.success("Dataset loaded successfully.")
                st.dataframe(df.head(50))
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
    else:
        st.info("No file uploaded.")
        if st.button("Use enhanced synthetic dataset"):
            df = generate_enhanced_data(36)
            df = ensure_datetime(df)
            df["district"] = df["district"].str.strip()
            st.session_state["data"] = df
            st.success("Enhanced synthetic dataset loaded.")
            st.dataframe(df.head(50))

# ---------------------------
# TRAIN & FORECAST
# ---------------------------
elif page == "Train & Forecast":
    st.header("Train LSTM and Forecast (cases only)")

    if st.session_state["data"] is None:
        st.warning("No data loaded. Please upload or generate first.")
    else:
        data = ensure_datetime(st.session_state["data"])
        districts = sorted(data["district"].unique())
        st.write(f"Detected {len(districts)} districts: {districts}")

        forecast_horizon = st.slider("Forecast horizon (months)", 1, 12, 6)
        btn = st.button("Train & Forecast")

        if btn:
            # Prepare sequences
            X_list, y_list = [], []
            scaler_dict = {}
            for d in districts:
                series = data.loc[data["district"] == d].sort_values("ds")
                vals = series["cases"].values.reshape(-1, 1)
                if len(vals) < SEQ_LEN + 1:
                    continue
                scaler = MinMaxScaler()
                scaled = scaler.fit_transform(vals)
                scaler_dict[d] = scaler
                for j in range(SEQ_LEN, len(scaled)):
                    X_list.append(scaled[j-SEQ_LEN:j, 0])
                    y_list.append(scaled[j, 0])

            X, y = np.array(X_list), np.array(y_list)
            X = X.reshape((X.shape[0], X.shape[1], 1))

            # Build and train LSTM
            model = Sequential([
                InputLayer(input_shape=(SEQ_LEN,1)),
                LSTM(64, activation="tanh"),
                Dense(32, activation="relu"),
                Dense(1, activation="linear")
            ])
            model.compile(optimizer="adam", loss="mse")
            es = EarlyStopping(monitor="loss", patience=8, restore_best_weights=True)
            model.fit(X, y, epochs=EPOCHS, batch_size=BATCH, verbose=1, callbacks=[es])
            st.success("LSTM trained successfully.")

            # Spatial adjacency
            coords = np.array([DISTRICT_COORDS[d] for d in districts])
            K = min(4, len(districts)-1)
            knn_graph = kneighbors_graph(coords, K, mode="distance", include_self=True)
            A = knn_graph.toarray()
            with np.errstate(divide='ignore'):
                A_sim = 1 / (A + np.eye(A.shape[0]))
            A_sim[np.isinf(A_sim)] = 0
            A_sim = (A_sim + A_sim.T)/2.0
            A_norm = A_sim / np.where(A_sim.sum(axis=1, keepdims=True)==0,1,A_sim.sum(axis=1,keepdims=True))

            # Forecasting
            future_months = pd.date_range(start=data["ds"].max() + pd.DateOffset(months=1),
                                          periods=forecast_horizon, freq="MS")
            fut_df = pd.DataFrame(index=future_months, columns=districts)
            for i, d in enumerate(districts):
                series = data.loc[data["district"]==d].sort_values("ds")
                vals = series["cases"].values.reshape(-1,1)
                scaler = scaler_dict[d]
                scaled_vals = scaler.transform(vals)
                curr_seq = scaled_vals[-SEQ_LEN:,0].reshape(1,SEQ_LEN,1)
                preds = []
                for _ in range(forecast_horizon):
                    p = model.predict(curr_seq, verbose=0)[0,0]
                    preds.append(p)
                    curr_seq = np.append(curr_seq[:,1:,:], [[[p]]], axis=1)
                preds_inv = scaler.inverse_transform(np.array(preds).reshape(-1,1)).flatten()
                fut_df[d] = preds_inv

            # Spatial smoothing
            alpha = st.slider("Spatial smoothing strength (α)", 0.0, 1.0, 0.6)
            fut_smoothed = fut_df.copy()
            for ts in fut_df.index:
                preds_vector = fut_df.loc[ts].values
                neighbor_avg = A_norm.dot(preds_vector)
                smoothed = alpha*preds_vector + (1-alpha)*neighbor_avg
                fut_smoothed.loc[ts] = smoothed

            # Convert to long format & add stocks & env features
            fut_smoothed_reset = fut_smoothed.reset_index().rename(columns={"index":"ds"})
            fut_long = fut_smoothed_reset.melt(
                id_vars="ds",
                var_name="district",
                value_name="cases"
            )
            fut_long["cases"] = fut_long["cases"].round().astype(int)
            fut_long["RDT Kits"] = (fut_long["cases"]*1.0).astype(int)
            fut_long["ACT Doses"] = (fut_long["cases"]*0.7).astype(int)
            fut_long["Bed Nets"] = (fut_long["cases"]*0.5).astype(int)
            fut_long["Rainfall_mm"] = fut_long["ds"].apply(lambda d: max(0, int(100*np.sin(2*np.pi*(d.month-6)/12) + np.random.normal(50,20))))
            fut_long["Humidity_%"] = fut_long["ds"].apply(lambda d: max(40, min(100, int(60 + 20*np.sin(2*np.pi*(d.month-5)/12) + np.random.normal(0,5)))))
            fut_long["Temperature_C"] = fut_long["ds"].apply(lambda d: max(20, min(40, int(25 + 5*np.cos(2*np.pi*(d.month-3)/12) + np.random.normal(0,2)))))

            st.session_state["forecast_table"] = fut_long
            st.session_state["lstm_model"] = model
            st.session_state["adj_matrix"] = A_norm

            st.success("Forecast and spatial smoothing completed.")
            st.dataframe(fut_long.head(50))

# ---------------------------
# DASHBOARD & MAP
# ---------------------------
elif page == "Dashboard & Map":
    st.header("Dashboard & Map")

    if st.session_state["forecast_table"] is None:
        st.warning("No forecast found. Please run Train & Forecast first.")
    else:
        fut_long = st.session_state["forecast_table"].copy()
        fut_long["ds"] = pd.to_datetime(fut_long["ds"])
        
        # Aggregate total cases per district
        totals = fut_long.groupby("district", as_index=False)["cases"].sum().rename(columns={"cases":"total_cases"})
        fig_bar = px.bar(totals, x="district", y="total_cases", color="total_cases", color_continuous_scale="Reds")
        st.plotly_chart(fig_bar, use_container_width=True)

        # Map: select month
        months = fut_long["ds"].dt.strftime("%Y-%m").unique().tolist()
        sel = st.selectbox("Select forecast month", months[::-1])
        view = fut_long[fut_long["ds"].dt.strftime("%Y-%m") == sel].copy()

        # Map coordinates
        view["lat"] = view["district"].map(lambda d: DISTRICT_COORDS.get(d, (None,None))[0])
        view["lon"] = view["district"].map(lambda d: DISTRICT_COORDS.get(d, (None,None))[1])

        fig_map = px.scatter_mapbox(
            view, lat="lat", lon="lon", size="cases",
            color="ACT Doses", hover_name="district",
            hover_data=["cases","RDT Kits","ACT Doses","Bed Nets",
                        "Rainfall_mm","Humidity_%","Temperature_C"],
            zoom=6, height=600, color_continuous_scale="Reds"
        )
        fig_map.update_layout(mapbox_style="open-street-map")
        st.plotly_chart(fig_map, use_container_width=True)

        st.dataframe(view[["district","cases","RDT Kits","ACT Doses","Bed Nets",
                           "Rainfall_mm","Humidity_%","Temperature_C"]])

# ---------------------------
# DISTRIBUTOR & RESOURCE FLOW
# ---------------------------
elif page == "Distributor & Resource Flow":
    st.header("GNN-based Distributor Resource Allocation")

    fut_long = st.session_state["forecast_table"].copy()
    latest_month = fut_long["ds"].max()
    latest = fut_long[fut_long["ds"] == latest_month]

    districts = latest["district"].tolist()

    # Build graph
    W, distributor_names = build_distributor_graph(districts)

    # Demand matrix
    demand = latest[["RDT Kits", "ACT Doses", "Bed Nets"]].values

    # GNN-style message passing
    allocation = W.T @ demand

    alloc_df = pd.DataFrame(
        allocation,
        columns=["RDT Kits", "ACT Doses", "Bed Nets"],
        index=distributor_names
    ).reset_index().rename(columns={"index": "Distributor Zone"})

    st.subheader(f"Resource Load per Distributor – {latest_month.strftime('%B %Y')}")
    st.dataframe(alloc_df)

    fig = px.bar(
        alloc_df,
        x="Distributor Zone",
        y=["RDT Kits", "ACT Doses", "Bed Nets"],
        barmode="group",
        title="GNN-based Resource Allocation"
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------
# SEASONAL TREND ANALYSIS
# ---------------------------
elif page == "Seasonal Trend Analysis":
    st.header("Seasonal Trend Analysis of Malaria Cases")

    if st.session_state["data"] is None:
        st.warning("No dataset loaded.")
    else:
        df = ensure_datetime(st.session_state["data"])
        df["month"] = df["ds"].dt.month_name()
        trend = df.groupby("month")["cases"].mean().reindex(
            ["January","February","March","April","May","June",
             "July","August","September","October","November","December"]
        )
        fig_line = px.line(x=trend.index, y=trend.values, markers=True,
                           title="Average Malaria Cases by Month",
                           labels={"x":"Month","y":"Average Cases"})
        st.plotly_chart(fig_line, use_container_width=True)
        st.markdown("""
*Insights:*
- Peaks appear during *monsoon months (June–August)*.
- Lows during *winter months (December–February)*.
""")