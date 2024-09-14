import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Generate sample data
def generate_sample_data(n_samples=1000):
    np.random.seed(42)
    data = {
        'pickup_latitude': np.random.uniform(12.9, 13.1, n_samples),
        'pickup_longitude': np.random.uniform(77.5, 77.7, n_samples),
        'dropoff_latitude': np.random.uniform(12.9, 13.1, n_samples),
        'dropoff_longitude': np.random.uniform(77.5, 77.7, n_samples),
        'fare': np.random.uniform(50, 500, n_samples),
        'distance': np.random.uniform(1, 20, n_samples),
        'duration': np.random.uniform(10, 60, n_samples),
        'car_type': np.random.choice(['Mini', 'Sedan', 'SUV', 'Luxury'], n_samples),
        'rating': np.random.uniform(1, 5, n_samples),
        'time_of_day': np.random.choice(['Morning', 'Afternoon', 'Evening', 'Night'], n_samples)
    }
    return pd.DataFrame(data)

# Perform EDA
def perform_eda(data):
    st.subheader("Exploratory Data Analysis")
    
    # Display basic statistics
    st.write("Basic Statistics:")
    st.write(data.describe())
    
    # Fare distribution
    st.write("Fare Distribution:")
    fig = px.histogram(data, x="fare", nbins=30)
    st.plotly_chart(fig)
    
    # Distance vs Fare
    st.write("Distance vs Fare:")
    fig = px.scatter(data, x="distance", y="fare", color="car_type")
    st.plotly_chart(fig)
    
    # Average fare by car type
    st.write("Average Fare by Car Type:")
    avg_fare = data.groupby("car_type")["fare"].mean().sort_values(ascending=False)
    fig = px.bar(avg_fare)
    st.plotly_chart(fig)
    
    # Heatmap of pickup locations
    st.write("Heatmap of Pickup Locations:")
    fig = px.density_mapbox(data, lat="pickup_latitude", lon="pickup_longitude", zoom=10, height=500)
    fig.update_layout(mapbox_style="open-street-map")
    st.plotly_chart(fig)

# Perform clustering
def perform_clustering(data, n_clusters):
    features = ['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude', 'fare', 'distance']
    X = data[features]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    
    data['cluster'] = clusters
    
    # Calculate silhouette score
    silhouette_avg = silhouette_score(X_scaled, clusters)
    
    return data, silhouette_avg

# Visualize clusters
def visualize_clusters(data):
    st.subheader("Cluster Visualization")
    fig = px.scatter_mapbox(data, 
                            lat="pickup_latitude", 
                            lon="pickup_longitude", 
                            color="cluster",
                            zoom=10, 
                            height=600,
                            width=800)
    fig.update_layout(mapbox_style="open-street-map")
    st.plotly_chart(fig)
    
    # Display cluster statistics
    st.write("Cluster Statistics:")
    cluster_stats = data.groupby('cluster').agg({
        'fare': ['mean', 'min', 'max'],
        'distance': ['mean', 'min', 'max']
    }).round(2)
    st.write(cluster_stats)

# Generate recommendations
def generate_recommendations(data):
    st.subheader("Recommendations")
    
    # Most popular pickup areas
    popular_pickup = data.groupby(['pickup_latitude', 'pickup_longitude']).size().sort_values(ascending=False).head(5)
    st.write("Top 5 Popular Pickup Areas:")
    st.write(popular_pickup)
    
    # Best performing car types
    best_car_types = data.groupby('car_type')['rating'].mean().sort_values(ascending=False)
    st.write("Car Types by Average Rating:")
    st.write(best_car_types)
    
    # Busiest times of day
    busiest_times = data['time_of_day'].value_counts()
    st.write("Busiest Times of Day:")
    fig = px.pie(values=busiest_times.values, names=busiest_times.index)
    st.plotly_chart(fig)
    
    # Recommendations based on analysis
    st.write("Key Recommendations:")
    st.write("1. Focus on providing more cars in the top pickup areas to meet demand.")
    st.write(f"2. Promote the '{best_car_types.index[0]}' car type, as it has the highest average rating.")
    st.write(f"3. Increase availability during {busiest_times.index[0]} hours to capitalize on peak demand.")

# Main Streamlit app
def main():
    st.title("OLA Car Rental Analysis App")
    
    # Generate sample data
    data = generate_sample_data()
    
    # Sidebar for user input
    st.sidebar.header("Analysis Parameters")
    analysis_type = st.sidebar.selectbox("Select Analysis Type", ["EDA", "Clustering", "Recommendations"])
    
    if analysis_type == "EDA":
        perform_eda(data)
    elif analysis_type == "Clustering":
        n_clusters = st.sidebar.slider("Number of Clusters", min_value=2, max_value=10, value=5)
        clustered_data, silhouette_avg = perform_clustering(data, n_clusters)
        visualize_clusters(clustered_data)
        st.write(f"Silhouette Score: {silhouette_avg:.2f}")
    else:
        generate_recommendations(data)

if __name__ == "__main__":
    main()