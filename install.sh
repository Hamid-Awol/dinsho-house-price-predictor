#!/bin/bash
echo "📦 Installing dependencies..."
echo "=============================="

pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Installation complete!"
    echo ""
    echo "To run the application:"
    echo "  ./run.sh"
    echo ""
else
    echo "❌ Installation failed. Please try installing manually:"
    echo "  pip install streamlit pandas numpy scikit-learn plotly openpyxl"
    exit 1
fi