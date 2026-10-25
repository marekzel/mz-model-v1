# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install project dependencies
pip install -r requirements.txt

# Run inference with ChatGPT-like GUI
streamlit run inference_gui.py

# Disclaimer
This model is experimental and provided "as is" for research purposes only. The authors make no guarantees regarding its performance, safety, or accuracy. Use at your own risk.
