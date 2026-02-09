import streamlit as st
import polars as pl
import altair as alt

st.title("NYC Taxi Data Analysis")
st.write("This dashboard provides insights into the NYC taxi trips dataset. It includes summary statistics and visualizations to help you understand the data better.")
@st.cache_data # Cache the data loading function to improve performance
def load_data():
    return pl.read_parquet("cleaned_trips.parquet") #! Filename may change
data = load_data()

# Sidebar navigation
st.sidebar.header("Navigation")
page = st.sidebar.radio(
    "Go to:",
    [
        "Summary Statistics",
        "Top Pickup Zones",
        "Fare by Hour",
        "Trip Distance Distribution",
        "Payment Type Proportion",
        "Fare Heatmap"
    ]
)

# Sidebar filters
st.sidebar.header("Filters")
import pandas as pd
pickup_hour_pd = data["pickup_hour"].to_pandas()
min_hour = int(pd.to_numeric(pickup_hour_pd.min(), errors="coerce"))
max_hour = int(pd.to_numeric(pickup_hour_pd.max(), errors="coerce"))
hour_range = st.sidebar.slider(
    "Hour range",
    min_value=min_hour,
    max_value=max_hour,
    value=(min_hour, max_hour)
)
all_days = sorted(data["pickup_day_of_week"].unique().to_list())
selected_days = st.sidebar.multiselect(
    "Select days of week",
    options=all_days,
    default=all_days
)
payment_type_map = {
    1: "Credit Card",
    2: "Cash",
    3: "No Charge",
    4: "Dispute",
    5: "Unknown"
}
all_payment_types = list(payment_type_map.values())
selected_payment_types = st.sidebar.multiselect(
    "Select payment types",
    options=all_payment_types,
    default=all_payment_types
)
selected_payment_codes = [k for k, v in payment_type_map.items() if v in selected_payment_types]
filtered_data = data.filter(
    (data["pickup_hour"] >= hour_range[0]) &
    (data["pickup_hour"] <= hour_range[1]) &
    (data["pickup_day_of_week"].is_in(selected_days)) &
    (data["payment_type"].is_in(selected_payment_codes))
)

@st.cache_data(show_spinner=False)
def summary_statistics(filtered_data):
    st.header("Summary Statistics")
    st.metric("Total Trips", filtered_data.shape[0])
    trip_distance_mean = filtered_data["trip_distance"].to_pandas().mean() if filtered_data.shape[0] > 0 else 0
    fare_amount_mean = filtered_data["fare_amount"].to_pandas().mean() if filtered_data.shape[0] > 0 else 0
    trip_duration_mean = filtered_data["trip_duration_minutes"].to_pandas().mean() if filtered_data.shape[0] > 0 else 0
    st.metric("Average Trip Distance (miles)", round(trip_distance_mean, 3) if trip_distance_mean is not None else 0)
    st.metric("Average Fare Amount ($)", round(fare_amount_mean, 3) if fare_amount_mean is not None else 0)
    st.metric("Average Trip Duration (minutes)", round(trip_duration_mean, 3) if trip_duration_mean is not None else 0)

if page == "Summary Statistics":
    summary_statistics(filtered_data)

@st.cache_data(show_spinner=False)
def top_pickup_zones(filtered_data):
    st.header("Top 10 pickup zones by number of trips")
    import pandas as pd
    zone_lookup = pd.read_csv("taxi_zone_lookup.csv")
    pickup_zone_counts = (
        filtered_data.group_by("PULocationID")
        .agg(pl.count().alias("count"))
        .sort("count", descending=True)
        .head(10)
    )
    pickup_zone_counts = pickup_zone_counts.to_pandas()
    pickup_zone_counts = pickup_zone_counts.merge(
        zone_lookup[["LocationID", "Zone"]],
        left_on="PULocationID",
        right_on="LocationID",
        how="left"
    )
    st.bar_chart(pickup_zone_counts, x="Zone", y="count")

if page == "Top Pickup Zones":
    top_pickup_zones(filtered_data)

@st.cache_data(show_spinner=False)
def fare_by_hour_page(filtered_data):
    st.header("Average Fare Amount by Hour of Day")
    fare_by_hour = (
        filtered_data.to_pandas()
        .groupby("pickup_hour", dropna=False)
        .agg({"fare_amount": "mean"})
        .reset_index()
    )
    chart = (
        alt.Chart(fare_by_hour)
        .mark_line()
        .encode(
            x=alt.X("pickup_hour:O", title="Hour of Day"),
            y=alt.Y("fare_amount:Q", title="Average Fare Amount ($)")
        )
    )
    st.altair_chart(chart, use_container_width=True)
    st.text("We can see that 4-6 AM has the highest average fare amount. The fares at other times are stable.")

if page == "Fare by Hour":
    fare_by_hour_page(filtered_data)

@st.cache_data(show_spinner=False)
def trip_distance_distribution(filtered_data):
    st.header("Distribution of Trip Distances")
    import plotly.express as px
    df_hist = filtered_data.to_pandas()
    df_hist = df_hist[(df_hist["trip_distance"] >= -1) & (df_hist["trip_distance"] <= 20)]
    import numpy as np
    bin_edges = np.arange(0, 20, 0.5)
    histogram_fig = px.histogram(
        df_hist,
        x="trip_distance",
        category_orders={"trip_distance": bin_edges.tolist()},
        nbins=len(bin_edges)-1,
        title="Distribution of Trip Distances (1-20 miles)",
        labels={"trip_distance": "Trip Distance (miles)", "count": "Number of Trips"}
    )
    histogram_fig.update_xaxes(dtick=1, range=[0, 20])
    histogram_fig.update_layout(bargap=0.1)
    st.plotly_chart(histogram_fig, use_container_width=True)

if page == "Trip Distance Distribution":
    trip_distance_distribution(filtered_data)

@st.cache_data(show_spinner=False)
def payment_type_proportion(filtered_data):
    st.header("Proportion of Trips by Payment Type")
    payment_type_counts = (
        filtered_data.to_pandas()
        .groupby("payment_type", dropna=False)
        .size()
        .reset_index(name="count")
    )
    payment_type_counts["payment_type"] = payment_type_counts["payment_type"].map(payment_type_map)
    pie_chart = (
        alt.Chart(payment_type_counts)
        .mark_arc()
        .encode(
            theta=alt.Theta("count:Q", title="Number of Trips"),
            color=alt.Color("payment_type:N", title="Payment Type")
        )
    )
    st.altair_chart(pie_chart, use_container_width=True)
    st.text("The majority of trips are paid by credit card, followed by cash. A small percentage of trips have unknown payment types. This shows most passengers prefer cashless payments, but there is still a significant portion using cash.")

if page == "Payment Type Proportion":
    payment_type_proportion(filtered_data)

@st.cache_data(show_spinner=False)
def fare_heatmap_page(filtered_data):
    st.header("Average Fare Amount by Pickup Hour and Day of Week")
    fare_heatmap = (
        filtered_data.to_pandas()
        .groupby(["pickup_hour", "pickup_day_of_week"], dropna=False)
        .agg({"fare_amount": "mean"})
        .reset_index()
    )
    heatmap = (
        alt.Chart(fare_heatmap)
        .mark_rect()
        .encode(
            x=alt.X("pickup_hour:O", title="Hour of Day"),
            y=alt.Y("pickup_day_of_week:O", title="Day of Week"),
            color=alt.Color(
                "fare_amount:Q",
                title="Average Fare Amount ($)",
                scale=alt.Scale(scheme="darkblue", reverse=False)  # Higher values are darker
            )
        )
    )
    st.altair_chart(heatmap, use_container_width=True)
    st.text("The heatmap shows that Wednesday 4AM has a tremendously high average fare amount, which is likely due to a small number of trips with very high fares. Other than that, the average fare amounts are relatively stable across different hours and days, with slightly higher fares during early morning hours (4-6 AM) on weekdays.")

if page == "Fare Heatmap":
    fare_heatmap_page(filtered_data)