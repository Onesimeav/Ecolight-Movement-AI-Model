import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set(style="whitegrid")

# Load data
df = pd.read_csv("data/dataset.csv", header=None, names=['timestamp', 'location', 'status', 'device'])
df["timestamp"] = pd.to_datetime(df["timestamp"], format='mixed', errors='coerce')

# Filter only motion data since the model is now trained without light level
df_motion = df[df["device"].str.contains("Control4-Motion", case=False)].copy()

# Extract time and room features
df_motion["hour"] = df_motion["timestamp"].dt.hour
df_motion["weekday"] = df_motion["timestamp"].dt.day_name()
df_motion["room_type"] = df_motion["location"].apply(lambda loc: next((rtype for prefix, rtype in {
    "Bathroom": "bathroom", "Bedroom": "bedroom", "DiningRoom": "dining",
    "Entryway": "entryway", "Hallway": "hallway", "Kitchen": "kitchen",
    "LivingRoom": "living", "MainEntryway": "entryway", "Office": "office"
}.items() if loc.startswith(prefix)), "other"))

# Create binary status column
df_motion["etat_bin"] = df_motion["status"].str.upper().str.strip() == "ON"
df_motion["etat_bin"] = df_motion["etat_bin"].astype(int)

# Create output directory
os.makedirs("figures_updated", exist_ok=True)

# Graph 1: Motion detection by hour and room_type
plt.figure(figsize=(14, 7))
sns.lineplot(data=df_motion[df_motion["etat_bin"] == 1],
             x="hour", hue="room_type", estimator='mean', ci=None, lw=2)
plt.title("Motion Detection Patterns by Hour and Room Type")
plt.xlabel("Hour of Day")
plt.ylabel("Proportion of Motion Detections (ON)")
plt.tight_layout()
plt.savefig("figures_updated/motion_patterns_by_hour_room_type.png")

# Graph 2: Motion detection heatmap by hour and room_type
motion_hour_room = df_motion[df_motion["etat_bin"] == 1].groupby(["room_type", "hour"]).size().unstack(fill_value=0)
plt.figure(figsize=(12, 6))
sns.heatmap(motion_hour_room, cmap="YlGnBu", linewidths=.5)
plt.title("Heatmap of Motion Events by Room Type and Hour")
plt.xlabel("Hour of Day")
plt.ylabel("Room Type")
plt.tight_layout()
plt.savefig("figures_updated/motion_heatmap_hour_room_type.png")

# Graph 3: Weekly motion patterns by room type
plt.figure(figsize=(14, 6))
sns.countplot(x="weekday", hue="room_type", data=df_motion[df_motion["etat_bin"] == 1],
              order=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
plt.title("Weekly Motion Pattern by Room Type")
plt.xlabel("Day of the Week")
plt.ylabel("Count of Motion Events")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("figures_updated/weekly_motion_by_room_type.png")

# Summary statistics for presentation
summary_stats = {
    "motion_peak_hours": df_motion[df_motion["etat_bin"] == 1]["hour"].value_counts().nlargest(3).index.tolist(),
    "top_motion_room_types": df_motion["room_type"].value_counts().nlargest(5).index.tolist(),
    "total_motion_events": len(df_motion[df_motion["etat_bin"] == 1]),
    "days_covered": df_motion["timestamp"].dt.date.nunique()
}
pd.Series(summary_stats).to_json("data/updated_stats.json")

