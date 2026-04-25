#!/bin/bash
echo "🚀 Starting Addis House Price Predictor..."
echo "=========================================="
echo ""py
echo "The app will open in your browser automatically"
echo "If it doesn't, open: http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the app"
echo ""

streamlit run app.py --server.port=8501 --server.address=localhost