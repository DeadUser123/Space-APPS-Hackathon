# Import the Flask app and ensure the trained model is loaded when
# the WSGI module is imported (Gunicorn imports this file; __main__ is not run).
from app import app, load_model_checkpoint, model_metadata

# Attempt to load the model at import time so production servers (Gunicorn)
# will have the model available immediately. This prints to the process logs.
try:
	if load_model_checkpoint():
		print("✅ Model loaded successfully at WSGI import time")
		if model_metadata:
			print(f"   Epoch: {model_metadata.get('epoch', 'Unknown')}")
			print(f"   Accuracy: {model_metadata.get('accuracy', 0):.4f}")
	else:
		print("⚠️ Warning: model file not found or could not be loaded at WSGI import time")
except Exception as e:
	print(f"Error loading model at WSGI import time: {e}")

# Expose the WSGI application
application = app
