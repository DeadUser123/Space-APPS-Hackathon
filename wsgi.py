# Import the Flask app
from app import app

# This is required for Vercel to recognize the application
# Vercel looks for an 'app' or 'application' object in the WSGI file
application = app
