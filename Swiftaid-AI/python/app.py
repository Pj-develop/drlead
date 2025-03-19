from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Initialize global variables for encoders and model
label_encoders = {}
scaler = None
model = None

def clean_value(val):
    """Clean value for JSON serialization"""
    if pd.isna(val) or pd.isnull(val):
        return None
    return val

def preprocess_data(data):
    """Preprocess the input data for model prediction."""
    # Convert list of dictionaries to DataFrame
    df = pd.DataFrame(data)
    
    # Clean and prepare data
    # Convert income ranges to numeric values
    income_mapping = {
        "0-30000": 15000,
        "30000-50000": 40000,
        "50000-100000": 75000,
        "100000+": 150000
    }
    
    # Convert budget levels to numeric values
    budget_mapping = {
        "low": 1,
        "medium": 2,
        "high": 3
    }
    
    # Convert distance ranges to numeric values
    distance_mapping = {
        "0-300": 150,
        "300-500": 400,
        "500-1000": 750,
        "1000+": 1200
    }
    
    # Convert duration to numeric values (months)
    duration_mapping = {
        "0-3 months": 1.5,
        "3-6 months": 4.5,
        "6-12 months": 9,
        "12+ months": 15,
        "permanent": 24
    }
    
    # Convert safety levels to numeric values
    safety_mapping = {
        "low": 1,
        "medium": 2,
        "high": 3
    }
    
    # Fill NaN values with defaults
    df["income"] = df["income"].fillna("0-30000")
    df["budget"] = df["budget"].fillna("medium")
    df["distance"] = df["distance"].fillna("0-300")
    df["duration"] = df["duration"].fillna("0-3 months")
    df["safety"] = df["safety"].fillna("medium")
    df["startDate"] = df["startDate"].fillna(pd.Timestamp.now().strftime("%Y-%m-%d"))
    df["foodPreferences"] = df["foodPreferences"].fillna("[]").apply(lambda x: [] if x == "[]" else x)
    df["transportType"] = df["transportType"].fillna("[]").apply(lambda x: [] if x == "[]" else x)
    df["accommodationType"] = df["accommodationType"].fillna("[]").apply(lambda x: [] if x == "[]" else x)
    
    # Apply numeric mappings
    df["income_score"] = df["income"].map(income_mapping).fillna(15000)
    df["budget_score"] = df["budget"].map(budget_mapping).fillna(1)
    df["distance_score"] = df["distance"].map(distance_mapping).fillna(150)
    df["duration_score"] = df["duration"].map(duration_mapping).fillna(1.5)
    df["safety_score"] = df["safety"].map(safety_mapping).fillna(1)
    
    # Calculate days until start
    df["days_until_start"] = (pd.to_datetime(df["startDate"]) - pd.Timestamp.now()).dt.days.fillna(0)
    
    # Process array fields
    df["food_score"] = df["foodPreferences"].apply(lambda x: len(x) if isinstance(x, list) else 0)
    df["transport_score"] = df["transportType"].apply(lambda x: len(x) if isinstance(x, list) else 0)
    df["accommodation_score"] = df["accommodationType"].apply(lambda x: len(x) if isinstance(x, list) else 0)
    
    # Create feature matrix
    feature_cols = [
        "income_score",
        "budget_score",
        "distance_score",
        "duration_score",
        "safety_score",
        "days_until_start",
        "food_score",
        "transport_score",
        "accommodation_score"
    ]
    
    X = df[feature_cols]
    
    # Scale features
    global scaler
    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)
    
    return X_scaled, df

def calculate_score(X_scaled):
    """Calculate exact scores for each customer based on their preferences and characteristics."""
    # Initialize weights for different factors
    weights = {
        "income": 20,      # 20 points max
        "budget": 15,      # 15 points max
        "duration": 20,    # 20 points max
        "safety": 10,      # 10 points max
        "distance": 10,    # 10 points max
        "days_until": 10,  # 10 points max
        "preferences": 15  # 15 points max
    }
    
    # Calculate composite score
    scores = np.zeros(X_scaled.shape[0])
    
    # Income score (0-20 points)
    # Higher income = higher score
    scores += weights["income"] * (X_scaled[:, 0] + 2) / 4  # Normalize to 0-20 range
    
    # Budget score (0-15 points)
    # Higher budget = higher score
    scores += weights["budget"] * (X_scaled[:, 1] + 1) / 2  # Normalize to 0-15 range
    
    # Distance score (0-10 points)
    # Lower distance = higher score
    scores += weights["distance"] * (1 - (X_scaled[:, 2] + 2) / 4)  # Invert and normalize
    
    # Duration score (0-20 points)
    # Longer duration = higher score
    scores += weights["duration"] * (X_scaled[:, 3] + 2) / 4  # Normalize to 0-20 range
    
    # Safety score (0-10 points)
    # Higher safety = higher score
    scores += weights["safety"] * (X_scaled[:, 4] + 1) / 2  # Normalize to 0-10 range
    
    # Days until start score (0-10 points)
    # Sooner start = higher score
    scores += weights["days_until"] * (1 - (X_scaled[:, 5] + 2) / 4)  # Invert and normalize
    
    # Preferences score (0-15 points)
    # More preferences = higher score
    pref_scores = np.mean(X_scaled[:, 6:], axis=1)  # Average of food, transport, accommodation scores
    scores += weights["preferences"] * (pref_scores + 2) / 4  # Normalize to 0-15 range
    
    # Ensure scores are between 0 and 100
    scores = np.clip(scores, 0, 100)
    
    return scores

@app.route('/rank_customers', methods=['POST'])
def rank_customers():
    """Endpoint to rank customers based on provided JSON data."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                "success": False,
                "error": "No data provided"
            }), 400

        # Preprocess the data
        X_scaled, df = preprocess_data(data)
        
        # Calculate scores
        scores = calculate_score(X_scaled)
        
        # Create rankings
        df["score"] = scores
        df = df.sort_values("score", ascending=False).reset_index(drop=True)
        df["rank"] = df.index + 1
        
        # Format results
        rankings = []
        for _, row in df.iterrows():
            rankings.append({
                "rank": int(row["rank"]),
                "email": clean_value(row["email"]),
                "phone": clean_value(row["phone"]),
                "score": float(row["score"]),
                "score_percentage": f"{row['score']:.1f}%",
                "current_city": clean_value(row["currentCity"]),
                "target_city": clean_value(row["targetCity"]),
                "budget": clean_value(row["budget"]),
                "duration": clean_value(row["duration"])
            })

        return jsonify({
            "success": True,
            "rankings": rankings,
            "model_metrics": {
                "total_customers": len(data),
                "features_used": [
                    "Income", "Budget", "Distance", "Duration",
                    "Safety", "Start Date", "Food Preferences",
                    "Transport Type", "Accommodation Type"
                ],
                "timestamp": datetime.now().isoformat()
            }
        })

    except Exception as e:
        print(f"Error processing request: {str(e)}")  # Add debug print
        import traceback
        traceback.print_exc()  # Print full traceback
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

if __name__ == '__main__':
    print("✅ Flask server started!")
    print("🔌 Endpoints available:")
    print("   POST /rank_customers - Rank customers based on provided data")
    app.run(debug=True, host='127.0.0.1', port=5000) 