# Import required libraries
import pandas as pd
import numpy as np
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


def load_models():
    """Load saved preprocessor and models from disk"""
    print("Loading preprocessor and models from disk...")
    model_dir = 'saved_models/'
    try:
        preprocessor = joblib.load(f'{model_dir}lead_scoring_preprocessor.joblib')
        rf_model = joblib.load(f'{model_dir}lead_scoring_rf_model.joblib')
        xgb_model = joblib.load(f'{model_dir}lead_scoring_xgb_model.joblib')
        print("Models loaded successfully!")
        return preprocessor, rf_model, xgb_model
    except Exception as e:
        print(f"Error loading models: {e}")
        return None, None, None

# Function to make predictions using loaded models
def predict_lead_score(preprocessor, rf_model, xgb_model, input_data):
    # Preprocess the input data
    input_processed = preprocessor.transform(input_data)
    
    # Make predictions with each model
    rf_pred = rf_model.predict(input_processed)[0]
    xgb_pred = xgb_model.predict(input_processed)[0]
    
    # Ensemble prediction (simple average)
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
        
        # Make prediction using the loaded models
        score = predict_lead_score(preprocessor, rf_model, xgb_model, input_data)
        
        # Interpret the score
        interpretation = interpret_lead_score(score)
        
        # Generate recommendations based on the score
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
    # Step 2: Load the saved models for prediction instead of using the trained models directly
    print("Loading saved models for prediction...")
    preprocessor, rf_model, xgb_model = load_models()
    
    if preprocessor is None or rf_model is None or xgb_model is None:
        print("Error: Failed to load models. Please check if models are saved correctly.")
        return
    
    print("Creating Gradio interface with loaded models...")
    interface = create_gradio_interface(preprocessor, rf_model, xgb_model)
    
    print("Launching Gradio app...")
    interface.launch()

# Run the script
if __name__ == "__main__":
    main()