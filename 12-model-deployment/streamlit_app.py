"""
🚀 ML Model Deployment with Streamlit
Interactive web application for machine learning predictions
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# Page configuration
st.set_page_config(
    page_title="🚀 ML Model Deployment",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .prediction-result {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #c3e6cb;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and prepare the Iris dataset"""
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = iris.target
    df['species_name'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
    return df, iris

@st.cache_resource
def train_model():
    """Train and cache the model"""
    iris = load_iris()
    X, y = iris.data, iris.target
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Calculate metrics
    train_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)
    
    return model, train_accuracy, test_accuracy, iris.feature_names, iris.target_names

def main():
    """Main application"""
    
    # Header
    st.markdown('<h1 class="main-header">🚀 ML Model Deployment Demo</h1>', unsafe_allow_html=True)
    st.markdown("### Interactive Iris Species Classification")
    
    # Load data and model
    df, iris = load_data()
    model, train_acc, test_acc, feature_names, target_names = train_model()
    
    # Sidebar
    st.sidebar.header("🎛️ Model Controls")
    
    # Model information
    with st.sidebar.expander("📊 Model Information", expanded=True):
        st.write(f"**Model Type:** Random Forest")
        st.write(f"**Training Accuracy:** {train_acc:.3f}")
        st.write(f"**Test Accuracy:** {test_acc:.3f}")
        st.write(f"**Features:** {len(feature_names)}")
        st.write(f"**Classes:** {len(target_names)}")
    
    # Feature inputs
    st.sidebar.header("🌸 Input Features")
    
    # Create input sliders
    feature_values = []
    for i, feature in enumerate(feature_names):
        min_val = float(df.iloc[:, i].min())
        max_val = float(df.iloc[:, i].max())
        mean_val = float(df.iloc[:, i].mean())
        
        value = st.sidebar.slider(
            f"{feature}",
            min_value=min_val,
            max_value=max_val,
            value=mean_val,
            step=0.1,
            help=f"Range: {min_val:.1f} - {max_val:.1f}"
        )
        feature_values.append(value)
    
    # Prediction button
    if st.sidebar.button("🔮 Make Prediction", type="primary"):
        # Make prediction
        features_array = np.array(feature_values).reshape(1, -1)
        prediction = model.predict(features_array)[0]
        prediction_proba = model.predict_proba(features_array)[0]
        
        # Store in session state
        st.session_state.prediction = prediction
        st.session_state.prediction_proba = prediction_proba
        st.session_state.feature_values = feature_values
    
    # Example buttons
    st.sidebar.header("📝 Quick Examples")
    col1, col2, col3 = st.sidebar.columns(3)
    
    with col1:
        if st.button("🌺 Setosa"):
            st.session_state.example_features = [5.1, 3.5, 1.4, 0.2]
    
    with col2:
        if st.button("🌿 Versicolor"):
            st.session_state.example_features = [5.7, 2.8, 4.1, 1.3]
    
    with col3:
        if st.button("🌸 Virginica"):
            st.session_state.example_features = [6.2, 2.8, 4.8, 1.8]
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Dataset overview
        st.header("📊 Dataset Overview")
        
        # Dataset statistics
        col_a, col_b, col_c, col_d = st.columns(4)
        with col_a:
            st.metric("Total Samples", len(df))
        with col_b:
            st.metric("Features", len(feature_names))
        with col_c:
            st.metric("Classes", len(target_names))
        with col_d:
            st.metric("Model Accuracy", f"{test_acc:.1%}")
        
        # Feature distribution plot
        st.subheader("🎨 Feature Distributions")
        
        selected_features = st.multiselect(
            "Select features to visualize:",
            feature_names,
            default=feature_names[:2]
        )
        
        if len(selected_features) >= 1:
            if len(selected_features) == 1:
                # Histogram
                fig = px.histogram(
                    df, 
                    x=selected_features[0], 
                    color='species_name',
                    title=f"Distribution of {selected_features[0]}",
                    marginal="box"
                )
            else:
                # Scatter plot
                fig = px.scatter(
                    df,
                    x=selected_features[0],
                    y=selected_features[1],
                    color='species_name',
                    title=f"{selected_features[0]} vs {selected_features[1]}",
                    hover_data=feature_names
                )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Correlation heatmap
        st.subheader("🔥 Feature Correlations")
        corr_matrix = df[feature_names].corr()
        
        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            title="Feature Correlation Matrix",
            color_continuous_scale="RdBu_r"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Prediction results
        st.header("🎯 Prediction Results")
        
        if hasattr(st.session_state, 'prediction'):
            prediction = st.session_state.prediction
            prediction_proba = st.session_state.prediction_proba
            
            # Display prediction
            st.markdown(f"""
            <div class="prediction-result">
                <h3>🌸 Predicted Species</h3>
                <h2 style="color: #1f77b4;">{target_names[prediction].title()}</h2>
                <p><strong>Confidence:</strong> {max(prediction_proba):.1%}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Probability chart
            prob_df = pd.DataFrame({
                'Species': target_names,
                'Probability': prediction_proba
            })
            
            fig = px.bar(
                prob_df,
                x='Species',
                y='Probability',
                title="Class Probabilities",
                color='Probability',
                color_continuous_scale="viridis"
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature importance
            st.subheader("📈 Feature Importance")
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=True)
            
            fig = px.bar(
                importance_df,
                x='Importance',
                y='Feature',
                orientation='h',
                title="Feature Importance in Model",
                color='Importance',
                color_continuous_scale="blues"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.info("👆 Adjust the feature values in the sidebar and click 'Make Prediction' to see results!")
        
        # Input summary
        st.subheader("📋 Current Input")
        input_df = pd.DataFrame({
            'Feature': feature_names,
            'Value': feature_values
        })
        st.dataframe(input_df, use_container_width=True)
    
    # Model performance section
    st.header("📊 Model Performance")
    
    # Load test data for detailed analysis
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42
    )
    
    y_pred = model.predict(X_test)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Confusion matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, y_pred)
        
        fig = px.imshow(
            cm,
            text_auto=True,
            aspect="auto",
            title="Confusion Matrix",
            labels=dict(x="Predicted", y="Actual"),
            x=target_names,
            y=target_names
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Classification report
        from sklearn.metrics import precision_recall_fscore_support
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average=None)
        
        metrics_df = pd.DataFrame({
            'Species': target_names,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1
        })
        
        fig = px.bar(
            metrics_df.melt(id_vars='Species', var_name='Metric', value_name='Score'),
            x='Species',
            y='Score',
            color='Metric',
            barmode='group',
            title="Classification Metrics by Species"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("### 🚀 About This App")
    st.markdown("""
    This Streamlit application demonstrates machine learning model deployment with:
    - **Interactive Predictions**: Real-time species classification
    - **Data Visualization**: Explore the Iris dataset
    - **Model Insights**: Feature importance and performance metrics
    - **User-Friendly Interface**: Easy-to-use controls and clear results
    
    Built with Streamlit, Scikit-learn, and Plotly. Perfect for showcasing ML models to stakeholders!
    """)

if __name__ == "__main__":
    main()
