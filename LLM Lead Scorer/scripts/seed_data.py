from app.models import mongo

leads = [
    {
        "Full Name": "John Doe",
        "Email Address": "john@example.com",
        "Phone Number": "1234567890",
        "Job Title": "Software Engineer",
        "Company Name": "TechCorp",
        "Company Size": 500,
        "Industry": "Tech",
        "Website Pages Visited": 5,
        "CTC in lac": 15,
        "Current Location": "Delhi",
        "Monthly Budget for Accommodation": 60,
        "Food Preference": "Vegetarian",
        "Relocation": "Bangalore",
        "Distance In KM bw current location to relocation": 2000,
        "Date of relocation": "2025-06-01",
        "Duration of stay in days": 365,
        "Transport Type": "Flight",
        "Accomodation Type": "Apartment"
    }
]

mongo.db.leads.insert_many(leads)
print("Database seeded successfully!")
