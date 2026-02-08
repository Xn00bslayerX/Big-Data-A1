import streamlit as st
import polars as pl
import altair as alt

st.title("NYC Taxi Data Analysis")
st.write("This dashboard provides insights into the NYC taxi trips dataset. It includes summary statistics and visualizations to help you understand the data better.")
@st.cache_data # Cache the data loading function to improve performance
def load_data():
    return pl.read_parquet("cleaned_trips.parquet") #! Filename may change
data = load_data()

# Prepare filter widgets
st.sidebar.header("Filters")
import pandas as pd

# Hour range slider
pickup_hour_pd = data["pickup_hour"].to_pandas()
min_hour = int(pd.to_numeric(pickup_hour_pd.min(), errors="coerce"))
max_hour = int(pd.to_numeric(pickup_hour_pd.max(), errors="coerce"))
hour_range = st.sidebar.slider(
    "Hour range",
    min_value=min_hour,
    max_value=max_hour,
    value=(min_hour, max_hour)
)

# Day of week multi-select
all_days = sorted(data["pickup_day_of_week"].unique().to_list())
selected_days = st.sidebar.multiselect(
    "Select days of week",
    options=all_days,
    default=all_days
)

# Payment type multi-select
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

# Apply filters
selected_payment_codes = [k for k, v in payment_type_map.items() if v in selected_payment_types]
filtered_data = data.filter(
    (data["pickup_hour"] >= hour_range[0]) &
    (data["pickup_hour"] <= hour_range[1]) &
    (data["pickup_day_of_week"].is_in(selected_days)) &
    (data["payment_type"].is_in(selected_payment_codes))
)

#Display summary statistics
st.header("Summary Statistics")
st.metric("Total Trips", filtered_data.shape[0])
trip_distance_mean = filtered_data["trip_distance"].to_pandas().mean() if filtered_data.shape[0] > 0 else 0
fare_amount_mean = filtered_data["fare_amount"].to_pandas().mean() if filtered_data.shape[0] > 0 else 0
trip_duration_mean = filtered_data["trip_duration_minutes"].to_pandas().mean() if filtered_data.shape[0] > 0 else 0
st.metric("Average Trip Distance (miles)", round(trip_distance_mean, 3) if trip_distance_mean is not None else 0)
st.metric("Average Fare Amount ($)", round(fare_amount_mean, 3) if fare_amount_mean is not None else 0)
st.metric("Average Trip Duration (minutes)", round(trip_duration_mean, 3) if trip_duration_mean is not None else 0)

#Visualize trip distance distribution
st.header("Top 10 pickup zones by number of trips")
# Use polars for grouping and aggregation
import pandas as pd

# Load the lookup file
zone_lookup = pd.read_csv("taxi_zone_lookup.csv")

# Group by PULocationID and count trips using polars
pickup_zone_counts = (
    filtered_data.group_by("PULocationID")
    .agg(pl.count().alias("count"))
    .sort("count", descending=True)
    .head(10)
)

# Convert to pandas for merging and plotting
pickup_zone_counts = pickup_zone_counts.to_pandas()

# Merge with zone lookup to get zone names
pickup_zone_counts = pickup_zone_counts.merge(
    zone_lookup[["LocationID", "Zone"]],
    left_on="PULocationID",
    right_on="LocationID",
    how="left"
)

# Display bar chart with zone names
st.bar_chart(pickup_zone_counts, x="Zone", y="count")

#Visualize fare amount distribution by hour of day
st.header("Average Fare Amount by Hour of Day")
# Group by hour of day and calculate average fare amount
fare_by_hour = (
    filtered_data.to_pandas()
    .groupby("pickup_hour", dropna=False)
    .agg({"fare_amount": "mean"})
    .reset_index()
)
# Create line chart using Altair
chart = (
    alt.Chart(fare_by_hour)
    .mark_line()
    .encode(
        x=alt.X("pickup_hour:O", title="Hour of Day"),
        y=alt.Y("fare_amount:Q", title="Average Fare Amount ($)")
    )
)
st.altair_chart(chart, use_container_width=True)

#Display histogram showing distribution of trip distances
st.header("Distribution of Trip Distances")
# Create histogram using Altair
histogram = (
    alt.Chart(filtered_data.to_pandas())
    .mark_bar()
    .encode(
        x=alt.X("trip_distance", bin=alt.Bin(maxbins=20), title="Trip Distance (miles)"),
        y=alt.Y("count()", title="Number of Trips")
    )
)
st.altair_chart(histogram, use_container_width=True)

#Display pie chart showing proportion of trips by payment type
st.header("Proportion of Trips by Payment Type")
# Group by payment type and count trips
payment_type_counts = (
    filtered_data.to_pandas()
    .groupby("payment_type", dropna=False)
    .size()
    .reset_index(name="count")
)
payment_type_counts["payment_type"] = payment_type_counts["payment_type"].map(payment_type_map)
# Create pie chart using Altair
pie_chart = (
    alt.Chart(payment_type_counts)
    .mark_arc()
    .encode(
        theta=alt.Theta("count:Q", title="Number of Trips"),
        color=alt.Color("payment_type:N", title="Payment Type")
    )
)
st.altair_chart(pie_chart, use_container_width=True)

#Create a heat map showing average fare amount by pickup hour and day of week
st.header("Average Fare Amount by Pickup Hour and Day of Week")
# Group by pickup hour and day of week, calculate average fare amount
fare_heatmap = (
    filtered_data.to_pandas()
    .groupby(["pickup_hour", "pickup_day_of_week"], dropna=False)
    .agg({"fare_amount": "mean"})
    .reset_index()
)
# Create heatmap using Altair
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