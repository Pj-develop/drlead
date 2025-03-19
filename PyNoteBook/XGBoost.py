# Import required libraries
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import gradio as gr
import joblib

# Disable warnings
import warnings
warnings.filterwarnings('ignore')

print("Loading libraries complete")

# Define possible values for categorical features
possible_cities = [
    'Bangalore', 'New Delhi', 'Mumbai', 'Kolkata', 'Chennai', 'Hyderabad',
    'Pune', 'Ahmedabad', 'Jaipur', 'Lucknow', 'Surat', 'Kanpur', 'Nagpur',
    'Patna', 'Bhopal', 'Indore', 'Vadodara', 'Coimbatore', 'Visakhapatnam',
    'Guwahati', 'Thiruvananthapuram', 'Kochi', 'Mysore', 'Goa', 'Chandigarh',
    'Amritsar', 'Jodhpur', 'Udaipur', 'Agra', 'Varanasi', 'Dehradun',
    'Ranchi', 'Jamshedpur', 'Bhubaneswar', 'Raipur', 'Not specified'
]
possible_start_dates = ['Within 30 days', '31-90 days', 'More than 90 days', 'Not specified']
possible_durations = ['1-7 days', '8-30 days', 'More than 30 days', 'Not specified']
possible_budgets = ['High', 'Medium', 'Low', 'Not specified']
possible_incomes = ['High', 'Medium', 'Low', 'Not specified']
possible_lifestyles = ['Active', 'Relaxed', 'Luxury', 'Budget']
possible_distances = ['Long', 'Medium', 'Short', 'Not specified']
possible_safeties = ['High', 'Medium', 'Low', 'Not specified']
possible_phone = ['Yes', 'No']
possible_pages = ['home', 'about', 'services', 'pricing', 'contact', 'blog']
key_pages = ['services', 'pricing', 'contact']
possible_food = ['Vegetarian', 'Vegan', 'Gluten-free', 'None']
possible_transport = ['Car', 'Public Transit', 'Walking', 'Biking']
possible_accommodation = ['Hotel', 'Apartment', 'House', 'Hostel']

# Generate synthetic data
def generate_synthetic_data(n_samples=50000):
    np.random.seed(42)
    data = {
        'email': ['email@example.com'] * n_samples,
        'phone_provided': np.random.choice(possible_phone, n_samples),
        'currentCity': np.random.choice(possible_cities, n_samples),
        'targetCity': np.random.choice(possible_cities, n_samples),
        'startDate': np.random.choice(possible_start_dates, n_samples),
        'duration': np.random.choice(possible_durations, n_samples),
        'budget': np.random.choice(possible_budgets, n_samples),
        'income': np.random.choice(possible_incomes, n_samples),
        'lifestyle': np.random.choice(possible_lifestyles, n_samples),
        'distance': np.random.choice(possible_distances, n_samples),
        'safety': np.random.choice(possible_safeties, n_samples),
        'pagesVisited': [list(np.random.choice(possible_pages, np.random.randint(0, 7), replace=False)) for _ in range(n_samples)],
        'foodPreferences': [list(np.random.choice(possible_food, np.random.randint(0, 4), replace=False)) for _ in range(n_samples)],
        'transportType': [list(np.random.choice(possible_transport, np.random.randint(0, 5), replace=False)) for _ in range(n_samples)],
        'accommodationType': [list(np.random.choice(possible_accommodation, np.random.randint(0, 5), replace=False)) for _ in range(n_samples)],
    }
    
    df = pd.DataFrame(data)
    
    # Set phone based on phone_provided
    df['phone'] = df['phone_provided'].apply(lambda x: '1234567890' if x == 'Yes' else '')
    
    return df

# Function to preprocess data
def preprocess_data(df):
    # Function to check if key pages were visited
    def key_pages_visited(pages_list):
        return sum(1 for page in pages_list if page in key_pages)
    
    # Compute numerical features
    df['pages_visited'] = df['pagesVisited'].apply(len)
    df['key_pages_visited'] = df['pagesVisited'].apply(key_pages_visited)
    df['food_preferences'] = df['foodPreferences'].apply(len)
    df['transport_preferences'] = df['transportType'].apply(len)
    df['accommodation_preferences'] = df['accommodationType'].apply(len)
    df['preferences_specified'] = df['food_preferences'] + df['transport_preferences'] + df['accommodation_preferences']
    
    # Create interaction features
    df['key_pages_ratio'] = df['key_pages_visited'] / df['pages_visited'].clip(lower=1)
    df['budget_income_match'] = (df['budget'] == df['income']).astype(int)
    df['is_local_travel'] = ((df['currentCity'] != 'Not specified') &
                             (df['targetCity'] != 'Not specified') &
                             (df['currentCity'] == df['targetCity'])).astype(int)
    
    return df

# Define scoring functions for lead quality
def calculate_lead_score(df):
    # Scoring functions
    def target_city_score(x): return 15 if x != 'Not specified' else 0
    def start_date_score(x): return {'Within 30 days': 25, '31-90 days': 15, 'More than 90 days': 5}.get(x, 0)
    def duration_score(x): return {'1-7 days': 5, '8-30 days': 10, 'More than 30 days': 15}.get(x, 0)
    def budget_score(x): return {'High': 15, 'Medium': 10, 'Low': 5}.get(x, 0)
    
    def pages_score(visited, key_visited):
        base_score = min(visited * 0.8, 6)
        key_score = min(key_visited * 2, 6)
        return base_score + key_score
    
    def preferences_score(food, transport, accom):
        return min(food + transport + accom, 12)
    
    def contact_score(x): return 12 if x == 'Yes' else 0
    def distance_score(x): return {'Long': 10, 'Medium': 5, 'Short': 2}.get(x, 0)
    def safety_score(x): return {'High': 10, 'Medium': 5, 'Low': 1}.get(x, 0)
    def income_score(x): return {'High': 5, 'Medium': 3, 'Low': 1}.get(x, 0)
    def lifestyle_score(x): return {'Luxury': 5, 'Active': 3, 'Relaxed': 2, 'Budget': 1}.get(x, 0)
    
    # Apply scoring
    df['total_score'] = (
        df['targetCity'].apply(target_city_score) +
        df['startDate'].apply(start_date_score) +
        df['duration'].apply(duration_score) +
        df['budget'].apply(budget_score) +
        df.apply(lambda row: pages_score(row['pages_visited'], row['key_pages_visited']), axis=1) +
        df.apply(lambda row: preferences_score(row['food_preferences'],
                                               row['transport_preferences'],
                                               row['accommodation_preferences']), axis=1) +
        df['phone_provided'].apply(contact_score) +
        df['distance'].apply(distance_score) +
        df['safety'].apply(safety_score) +
        df['income'].apply(income_score) +
        df['lifestyle'].apply(lifestyle_score)
    )
    
    return df

# Exploratory data analysis function
def perform_eda(df):
    print("Dataset shape:", df.shape)
    print("\nSummary statistics for numerical features:")
    print(df[['pages_visited', 'key_pages_visited', 'preferences_specified', 'total_score']].describe())
    
    # Visualize distribution of target variable
    plt.figure(figsize=(10, 6))
    sns.histplot(df['total_score'], kde=True)
    plt.title('Distribution of Lead Scores')
    plt.savefig('lead_score_distribution.png')
    plt.close()
    
    # Correlation analysis
    numerical_cols = ['pages_visited', 'key_pages_visited', 'preferences_specified',
                      'food_preferences', 'transport_preferences', 'accommodation_preferences',
                      'key_pages_ratio', 'budget_income_match', 'is_local_travel', 'total_score']
    
    plt.figure(figsize=(12, 10))
    corr = df[numerical_cols].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png')
    plt.close()
    
    return corr

# Train models
def train_models(df):
    # Prepare features and target
    categorical_features = ['targetCity', 'currentCity', 'startDate', 'duration', 'budget',
                            'phone_provided', 'distance', 'safety', 'income', 'lifestyle']
    
    numerical_features = ['pages_visited', 'key_pages_visited', 'preferences_specified',
                          'food_preferences', 'transport_preferences', 'accommodation_preferences',
                          'key_pages_ratio', 'budget_income_match', 'is_local_travel']
    
    X = df[categorical_features + numerical_features]
    y = df['total_score']
    
    # Define preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
            ('num', StandardScaler(), numerical_features)
        ],
        verbose_feature_names_out=False
    )
    
    # Preprocess data
    X_processed = preprocessor.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, random_state=42
    )
    
    # Train Random Forest
    print("Training Random Forest model...")
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    
    # Train XGBoost
    print("Training XGBoost model...")
    xgb_model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    xgb_model.fit(X_train, y_train)
    
    # Get predictions
    y_pred_rf = rf_model.predict(X_test)
    y_pred_xgb = xgb_model.predict(X_test)
    
    # Create ensemble predictions (simple average)
    y_pred_ensemble = (y_pred_rf + y_pred_xgb) / 2
    
    # Evaluate models
    print("\nModel Evaluation:")
    print("-" * 50)
    print("Random Forest:")
    print(f"MSE: {mean_squared_error(y_test, y_pred_rf):.2f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_rf)):.2f}")
    print(f"MAE: {mean_absolute_error(y_test, y_pred_rf):.2f}")
    print(f"R²: {r2_score(y_test, y_pred_rf):.2f}")
    
    print("\nXGBoost:")
    print(f"MSE: {mean_squared_error(y_test, y_pred_xgb):.2f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_xgb)):.2f}")
    print(f"MAE: {mean_absolute_error(y_test, y_pred_xgb):.2f}")
    print(f"R²: {r2_score(y_test, y_pred_xgb):.2f}")
    
    print("\nEnsemble:")
    print(f"MSE: {mean_squared_error(y_test, y_pred_ensemble):.2f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_ensemble)):.2f}")
    print(f"MAE: {mean_absolute_error(y_test, y_pred_ensemble):.2f}")
    print(f"R²: {r2_score(y_test, y_pred_ensemble):.2f}")
    
    # Save models and preprocessor
    joblib.dump(preprocessor, 'lead_scoring_preprocessor.joblib')
    joblib.dump(rf_model, 'lead_scoring_rf.joblib')
    joblib.dump(xgb_model, 'lead_scoring_xgb.joblib')
    
    return preprocessor, rf_model, xgb_model

# Function to make predictions
def predict_lead_score(preprocessor, rf_model, xgb_model, input_data):
    # Preprocess the input data
    input_processed = preprocessor.transform(input_data)
    
    # Make predictions
    rf_pred = rf_model.predict(input_processed)[0]
    xgb_pred = xgb_model.predict(input_processed)[0]
    
    # Ensemble prediction
    ensemble_pred = (rf_pred + xgb_pred) / 2
    
    return ensemble_pred

# Function to interpret lead score
def interpret_lead_score(score):
    if score >= 90:
        return f"Score: {score:.1f} - Hot Lead (Very High Potential)"
    elif score >= 70:
        return f"Score: {score:.1f} - Warm Lead (High Potential)"
    elif score >= 50:
        return f"Score: {score:.1f} - Moderate Lead (Medium Potential)"
    elif score >= 30:
        return f"Score: {score:.1f} - Cool Lead (Low Potential)"
    else:
        return f"Score: {score:.1f} - Cold Lead (Very Low Potential)"

# Gradio interface function
def create_gradio_interface(preprocessor, rf_model, xgb_model):
    def predict(target_city, current_city, start_date, duration, budget, phone_provided,
                distance, safety, income, lifestyle, pages_visited, key_pages_visited,
                food_preferences, transport_preferences, accommodation_preferences):
        
        # Create a dataframe with the input data
        input_data = pd.DataFrame({
            'targetCity': [target_city],
            'currentCity': [current_city],
            'startDate': [start_date],
            'duration': [duration],
            'budget': [budget],
            'phone_provided': [phone_provided],
            'distance': [distance],
            'safety': [safety],
            'income': [income],
            'lifestyle': [lifestyle],
            'pages_visited': [pages_visited],
            'key_pages_visited': [key_pages_visited],
            'food_preferences': [food_preferences],
            'transport_preferences': [transport_preferences],
            'accommodation_preferences': [accommodation_preferences],
            'key_pages_ratio': [key_pages_visited / max(pages_visited, 1)],
            'budget_income_match': [1 if budget == income else 0],
            'is_local_travel': [1 if current_city == target_city and current_city != 'Not specified' else 0],
            'preferences_specified': [food_preferences + transport_preferences + accommodation_preferences]
        })
        
        # Make prediction
        score = predict_lead_score(preprocessor, rf_model, xgb_model, input_data)
        
        # Interpret the score
        interpretation = interpret_lead_score(score)
        
        # Generate some recommendations based on the score
        recommendations = generate_recommendations(score, input_data)
        
        return interpretation, recommendations
    
    # Function to generate recommendations
    def generate_recommendations(score, input_data):
        recommendations = []
        
        if score < 50:
            if input_data['phone_provided'].iloc[0] == 'No':
                recommendations.append("Encourage the lead to provide contact information.")
            
            if input_data['key_pages_visited'].iloc[0] < 2:
                recommendations.append("Nudge the lead to visit key pages like pricing and services.")
            
            if input_data['startDate'].iloc[0] == 'Not specified':
                recommendations.append("Try to get the lead to specify a start date.")
            
            if input_data['budget'].iloc[0] == 'Not specified':
                recommendations.append("Get the lead to specify their budget.")
                
            if input_data['preferences_specified'].iloc[0] < 3:
                recommendations.append("Encourage the lead to specify more preferences.")
        
        elif score >= 50 and score < 70:
            if input_data['phone_provided'].iloc[0] == 'No':
                recommendations.append("Follow up with an email asking for contact information.")
            
            recommendations.append("Send targeted content about their destination.")
            
            if input_data['budget'].iloc[0] != 'Not specified':
                recommendations.append(f"Provide {input_data['budget'].iloc[0].lower()}-budget options.")
        
        elif score >= 70:
            recommendations.append("Assign a dedicated sales representative for immediate follow-up.")
            recommendations.append("Prepare a personalized travel package.")
            
            if input_data['startDate'].iloc[0] == 'Within 30 days':
                recommendations.append("Offer an early booking discount.")
        
        if not recommendations:
            recommendations.append("No specific recommendations available.")
        
        return "\n".join(recommendations)
    
    # Create the Gradio interface
    with gr.Blocks(title="Lead Scoring System") as interface:
        gr.Markdown("# Travel Lead Scoring System")
        gr.Markdown("Enter lead information to predict the quality score.")
        
        with gr.Row():
            with gr.Column():
                target_city = gr.Dropdown(choices=possible_cities, label="Target City")
                current_city = gr.Dropdown(choices=possible_cities, label="Current City")
                start_date = gr.Dropdown(choices=possible_start_dates, label="Start Date")
                duration = gr.Dropdown(choices=possible_durations, label="Duration")
                budget = gr.Dropdown(choices=possible_budgets, label="Budget")
                phone_provided = gr.Dropdown(choices=possible_phone, label="Phone Provided")
                distance = gr.Dropdown(choices=possible_distances, label="Distance")
                safety = gr.Dropdown(choices=possible_safeties, label="Safety Priority")
            
            with gr.Column():
                income = gr.Dropdown(choices=possible_incomes, label="Income Level")
                lifestyle = gr.Dropdown(choices=possible_lifestyles, label="Lifestyle")
                pages_visited = gr.Slider(minimum=0, maximum=10, value=0, step=1, label="Pages Visited")
                key_pages_visited = gr.Slider(minimum=0, maximum=3, value=0, step=1, label="Key Pages Visited")
                food_preferences = gr.Slider(minimum=0, maximum=3, value=0, step=1, label="Food Preferences Count")
                transport_preferences = gr.Slider(minimum=0, maximum=4, value=0, step=1, label="Transport Preferences Count")
                accommodation_preferences = gr.Slider(minimum=0, maximum=4, value=0, step=1, label="Accommodation Preferences Count")
        
        submit_btn = gr.Button("Predict Lead Score")
        
        with gr.Row():
            score_output = gr.Textbox(label="Lead Score")
            recommendations = gr.Textbox(label="Recommendations", lines=5)
        
        submit_btn.click(
            fn=predict,
            inputs=[
                target_city, current_city, start_date, duration, budget, phone_provided,
                distance, safety, income, lifestyle, pages_visited, key_pages_visited,
                food_preferences, transport_preferences, accommodation_preferences
            ],
            outputs=[score_output, recommendations]
        )
    
    return interface

# Main function to run the entire pipeline
def main():
    print("Generating synthetic data...")
    df = generate_synthetic_data(n_samples=50000)
    
    print("Preprocessing data...")
    df = preprocess_data(df)
    
    print("Calculating lead scores...")
    df = calculate_lead_score(df)
    
    print("Performing exploratory data analysis...")
    perform_eda(df)
    
    print("Training models...")
    preprocessor, rf_model, xgb_model = train_models(df)
    
    print("Creating Gradio interface...")
    interface = create_gradio_interface(preprocessor, rf_model, xgb_model)
    
    print("Launching Gradio app...")
    interface.launch()


main()