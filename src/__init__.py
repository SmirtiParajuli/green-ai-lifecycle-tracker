"""
GreenAI Lifecycle Tracker source code.
This package contains the main implementation of the GreenAI lifecycle tracking system.
"""
# Create __init__.py files if missing
import os

for path in [
	"src/__init__.py",
	"src/models/__init__.py",
	"src/utils/__init__.py"
]:
	os.makedirs(os.path.dirname(path), exist_ok=True)
	if not os.path.exists(path):
		with open(path, "w"):
			pass