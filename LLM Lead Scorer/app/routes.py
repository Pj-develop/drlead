from flask import Blueprint, request, jsonify, render_template
from app.models import mongo
from app.scoring import calculate_lead_score
import google.generativeai as genai
import os
import re

bp = Blueprint("routes", __name__)

# Configure Google AI Model
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")

# **Serve the HTML Form**
@bp.route("/", methods=["GET"])
def lead_form():
    return render_template("form.html")

@bp.route("/add_lead", methods=["POST"])
def add_lead():
    data = request.json
    if not data:
        return jsonify({"error": "Invalid data"}), 400

    # AI Prompt for Lead Scoring
    prompt = f"""
    You are an AI lead-scoring assistant for an urban relocation platform. 
    Evaluate leads based on the given **weighted criteria** and assign a **lead score (0-100).**

    ### **Lead Data:**
    - Full Name: {data.get("full_name", "N/A")}
    - Email: {data.get("email", "N/A")}
    - Phone: {data.get("phone", "N/A")}
    - Job Title: {data.get("job_title", "N/A")}
    - Company Name: {data.get("company_name", "N/A")}
    - Company Size: {data.get("company_size", "N/A")}
    - Industry: {data.get("industry", "N/A")}
    - Website Pages Visited: {data.get("website_pages_visited", 0)}
    - CTC in Lacs: {data.get("ctc", 0)}
    - Current Location: {data.get("current_location", "N/A")}
    - Monthly Budget for Accommodation: {data.get("monthly_budget", 0)}
    - Food Preference: {data.get("food_preference", "N/A")}
    - Relocation City: {data.get("relocation", "N/A")}
    - Distance (KM) from Current to Relocation: {data.get("distance", 0)}
    - Date of Relocation: {data.get("date_of_relocation", "N/A")}
    - Duration of Stay (Days): {data.get("duration_of_stay", 0)}
    - Transport Type: {data.get("transport_type", "N/A")}
    - Accommodation Type: {data.get("accommodation_type", "N/A")}

    ### **Scoring Rubric:**
    1. **High Impact (50%)**
       - **CTC**, **company size**, and **website visits** → Higher = better.
       - Preferred **job titles** and **industry** → Higher score.
       - Confirmed **relocation** → Bonus points.

    2. **Medium Impact (30%)**
       - Shorter **relocation distance** and **earlier relocation date** = Higher score.

    3. **Low Impact (20%)**
       - Monthly **budget, transport type, accommodation type** contribute slightly.

    **Task:**  
    - Calculate a **lead score (0-100)** based on the rubric.  
    - Respond in **valid JSON format**:  
      ```json
      {{"lead_score": SCORE}}
      ```
    """

    # Get AI-generated lead score
    response = model.generate_content(prompt)

    # Extract the lead score using regex
    match = re.search(r'"lead_score"\s*:\s*(\d+)', response.text)
    lead_score = int(match.group(1)) if match else 0

    # Ensure lead score is within valid range
    lead_score = max(0, min(100, lead_score))

    print(f"AI Response: {response.text}")
    print(f"Extracted Lead Score: {lead_score}")

    # Save lead with score to MongoDB
    data["lead_score"] = lead_score
    mongo.db.leads.insert_one(data)

    return jsonify({"lead_score": lead_score}), 201

@bp.route("/get_leads", methods=["GET"])
def get_leads():
    leads = list(mongo.db.leads.find({}, {"_id": 0}))
    return jsonify(leads), 200
