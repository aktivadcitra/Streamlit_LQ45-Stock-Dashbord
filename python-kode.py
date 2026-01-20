import streamlit as st
import yfinance as yf
import pandas as pd
import altair as alt
import numpy as np
import plotly.graph_objects as go
from prophet import Prophet

st.set_page_config(
    page_title="LQ45 Stock analysis dashboard",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
)
st.title('LQ45 Stock Dashboard')
cols = st.columns([1, 3])

STOCKS = [
    "AADI.JK",
    "ACES.JK",
    "ADMR.JK",
    "ADRO.JK",
    "AKRA.JK",
    "AMMN.JK",
    "AMRT.JK",
    "ANTM.JK",
    "ARTO.JK",
    "ASII.JK",
    "BBCA.JK",
    "BBNI.JK",
    "BBRI.JK",
    "BBTN.JK",
    "BMRI.JK",
    "BRIS.JK",
    "BRPT.JK",
    "CPIN.JK",
    "CTRA.JK",
    "EXCL.JK",
    "GOTO.JK",
    "ICBP.JK",
    "INCO.JK",
    "INDF.JK",
    "INKP.JK",
    "ISAT.JK",
    "ITMG.JK",
    "JPFA.JK",
    "JSMR.JK",
    "KLBF.JK",
    "MAPA.JK",
    "MAPI.JK",
    "MBMA.JK",
    "MDKA.JK",
    "MEDC.JK",
    "PGAS.JK",
    "PGEO.JK",
    "PTBA.JK",
    "SCMA.JK",
    "SMGR.JK",
    "SMRA.JK",
    "TLKM.JK",
    "TOWR.JK",
    "UNTR.JK",
    "UNVR.JK",
]

DEFAULT_STOCKS = ["BBCA.JK","BBRI.JK", "BMRI.JK", "ASII.JK", "TLKM.JK", "ANTM.JK", "ADRO.JK", "UNVR.JK"]

def stocks_to_str(stocks):
    return ",".join(stocks)

if "tickers_input" not in st.session_state:
    st.session_state.tickers_input = st.query_params.get(
        "stocks", stocks_to_str(DEFAULT_STOCKS)
    ).split(",")


# Callback to update query param when input changes
def update_query_param():
    if st.session_state.tickers_input:
        st.query_params["stocks"] = stocks_to_str(st.session_state.tickers_input)
    else:
        st.query_params.pop("stocks", None)


top_left_cell = cols[0].container(
    border=True, height="stretch", vertical_alignment="center"
)

with top_left_cell:
    # Selectbox for stock tickers
    tickers = st.multiselect(
        "Stock tickers",
        options=sorted(set(STOCKS) | set(st.session_state.tickers_input)),
        default=st.session_state.tickers_input,
        placeholder="Choose stocks to compare. Example: NVDA",
        accept_new_options=True,
    )

horizon_map = {
    "1 Months": "1mo",
    "3 Months": "3mo",
    "6 Months": "6mo",
    "1 Year": "1y",
    "5 Years": "5y",
    "10 Years": "10y",
    "20 Years": "20y",
}

with top_left_cell:
    # Buttons for picking time horizon
    horizon = st.pills(
        "Time horizon",
        options=list(horizon_map.keys()),
        default="6 Months",
    )

tickers = [t.upper() for t in tickers]

if tickers:
    st.query_params["stocks"] = stocks_to_str(tickers)
else:
    st.query_params.pop("stocks", None)

if not tickers:
    top_left_cell.info("Pick some stocks to compare", icon=":material/info:")
    st.stop()


right_cell = cols[1].container(
    border=True, height="stretch", vertical_alignment="center"
)


@st.cache_resource(show_spinner=False, ttl="6h")
def load_data(tickers, period):
    tickers_obj = yf.Tickers(tickers)
    data = tickers_obj.history(period=period)
    if data is None:
        raise RuntimeError("YFinance returned no data.")
    return data["Close"]


# Load the data
try:
    data = load_data(tickers, horizon_map[horizon])
except yf.exceptions.YFRateLimitError as e:
    st.warning("YFinance is rate-limiting us :(\nTry again later.")
    load_data.clear() 
    st.stop()

empty_columns = data.columns[data.isna().all()].tolist()

if empty_columns:
    st.error(f"Error loading data for the tickers: {', '.join(empty_columns)}.")
    st.stop()


normalized = data.div(data.iloc[0])

latest_norm_values = {normalized[ticker].iat[-1]: ticker for ticker in tickers}
max_norm_value = max(latest_norm_values.items())
min_norm_value = min(latest_norm_values.items())

bottom_left_cell = cols[0].container(
    border=True, height="stretch", vertical_alignment="center"
)

with bottom_left_cell:
    cols = st.columns(2)
    cols[0].metric(
        "Best stock",
        max_norm_value[1],
        delta=f"{round(max_norm_value[0] * 100)}%",
        width="content",
    )
    cols[1].metric(
        "Worst stock",
        min_norm_value[1],
        delta=f"{round(min_norm_value[0] * 100)}%",
        width="content",
    )


# Plot normalized prices
with right_cell:
    st.altair_chart(
        alt.Chart(
            normalized.reset_index().melt(
                id_vars=["Date"], var_name="Stock", value_name="Normalized price"
            )
        )
        .mark_line()
        .encode(
            alt.X("Date:T"),
            alt.Y("Normalized price:Q").scale(zero=False),
            alt.Color("Stock:N"),
        )
        .properties(height=400)
    )

if len(tickers) <= 1:
    st.warning("Pick 2 or more tickers to compare them")
    st.stop()

NUM_COLS = 4
cols = st.columns(NUM_COLS)

for i, ticker in enumerate(tickers):
    # Calculate peer average (excluding current stock)
    peers = normalized.drop(columns=[ticker])
    peer_avg = peers.mean(axis=1)

    # Create DataFrame with peer average.
    plot_data = pd.DataFrame(
        {
            "Date": normalized.index,
            ticker: normalized[ticker],
            "Peer average": peer_avg,
        }
    ).melt(id_vars=["Date"], var_name="Series", value_name="Price")

    chart = (
        alt.Chart(plot_data)
        .mark_line()
        .encode(
            alt.X("Date:T"),
            alt.Y("Price:Q").scale(zero=False),
            alt.Color(
                "Series:N",
                scale=alt.Scale(domain=[ticker, "Peer average"], range=["red", "gray"]),
                legend=alt.Legend(orient="bottom"),
            ),
            alt.Tooltip(["Date", "Series", "Price"]),
        )
        .properties(title=f"{ticker} vs peer average", height=300)
    )

    cell = cols[(i * 2) % NUM_COLS].container(border=True)
    cell.write("")
    cell.altair_chart(chart, use_container_width=True)

    # Create Delta chart
    plot_data = pd.DataFrame(
        {
            "Date": normalized.index,
            "Delta": normalized[ticker] - peer_avg,
        }
    )

    chart = (
        alt.Chart(plot_data)
        .mark_area()
        .encode(
            alt.X("Date:T"),
            alt.Y("Delta:Q").scale(zero=False),
        )
        .properties(title=f"{ticker} minus peer average", height=300)
    )

    cell = cols[(i * 2 + 1) % NUM_COLS].container(border=True)
    cell.write("")
    cell.altair_chart(chart, use_container_width=True)


#==================================================================================================
#==================================================================================================
#==== MENCOBA GOLDEN CROSS PADA SAHAM LQ45 ==============================================
#==================================================================================================
#==================================================================================================

def prepare_golden_cross_data(ticker, period):
    df = yf.download(ticker, period=period)
    df.reset_index(inplace=True)

    df["MA_50"] = df["Close"].rolling(window=50).mean()
    df["MA_200"] = df["Close"].rolling(window=200).mean()

    # Golden Cross: MA50 crosses FROM BELOW MA200
    df["Golden_Cross"] = (
        (df["MA_50"] > df["MA_200"]) &
        (df["MA_50"].shift(1) <= df["MA_200"].shift(1))
    ).astype(int)

    df["Kode Saham"] = ticker
    return df

st.divider()
st.subheader("📈 Stock Golden Cross")

gc_cols = st.columns(2)

with gc_cols[0]:
    gc_stock_1 = st.selectbox(
        "Select Stock (Chart 1)",
        options=STOCKS,
        index=0,
        key="gc_stock_1"
    )

with gc_cols[1]:
    gc_stock_2 = st.selectbox(
        "Select Stock (Chart 2)",
        options=[s for s in STOCKS if s != gc_stock_1],
        index=0,
        key="gc_stock_2"
    )

gc_data_1 = prepare_golden_cross_data(gc_stock_1, horizon_map[horizon])
gc_data_2 = prepare_golden_cross_data(gc_stock_2, horizon_map[horizon])

def golden_cross_chart(df, ticker):
    base = alt.Chart(df).encode(
        x=alt.X("Date:T", title="Date")
    )

    close_line = base.mark_line(color="gray").encode(
        y=alt.Y("Close:Q", title="Price"),
        tooltip=["Date:T", "Close:Q"]
    )

    ma50 = base.mark_line(color="blue").encode(
        y="MA_50:Q"
    )

    ma200 = base.mark_line(color="orange").encode(
        y="MA_200:Q"
    )

    golden_cross = base.transform_filter(
        alt.datum.Golden_Cross == 1
    ).mark_point(
        color="gold",
        size=120,
        shape="triangle-up"
    ).encode(
        y="Close:Q"
    )

    return (
        close_line + ma50 + ma200 + golden_cross
    ).properties(
        title=f"{ticker} Golden Cross",
        height=600
    )

chart_cols = st.columns(2)

with chart_cols[0]:
    st.altair_chart(
        golden_cross_chart(gc_data_1, gc_stock_1),
        use_container_width=True
    )

with chart_cols[1]:
    st.altair_chart(
        golden_cross_chart(gc_data_2, gc_stock_2),
        use_container_width=True
    )

#==================================================================================================
#==================================================================================================
#==================================================================================================
# Load raw data for a single stock display
#==================================================================================================
#==================================================================================================
#==================================================================================================


@st.cache_data(ttl="6h")
def load_raw_data(ticker):
    df = yf.download(ticker, period="max")
    df.reset_index(inplace=True)
    df["Kode Saham"] = ticker

    # Return calculations
    df["Return 1 Month"] = df["Close"].pct_change(21)
    df["Return 1 Year"] = df["Close"].pct_change(252)

    return df

st.divider()
st.subheader("📄 Raw Data")

selected_stock = st.selectbox(
    "Select dataset for raw data",
    options=STOCKS,
    index=0,
    key="raw_data_stock"
)
raw_data = load_raw_data(selected_stock)
raw_table = raw_data[
    [
        "Kode Saham",
        "Date",
        "Open",
        "High",
        "Low",
        "Close",
        "Return 1 Month",
        "Return 1 Year",
    ]
]

st.dataframe(
    raw_table.tail(50),
    use_container_width=True
)
