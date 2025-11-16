#!/bin/bash

# Setup script for AI Persona Usability Testing System

echo "======================================"
echo "AI Persona Testing - Setup Script"
echo "======================================"

# Activate bluepill conda environment
echo "Activating bluepill environment..."
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate bluepill

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Install Playwright browsers
echo "Installing Playwright browsers..."
playwright install 

echo ""
echo "======================================"
echo "Setup Complete!"
echo "======================================"
echo ""
echo "To run the system:"
echo "1. Activate bluepill environment: conda activate bluepill"
echo "2. Run all personas: python main.py"
echo "3. Run single persona: python main.py --single sarah_kim"
echo "4. Run with visible browser: python main.py (default)"
echo "5. Run headless: python main.py --headless"
echo ""
echo "Available personas:"
echo "  - sarah_kim: Subscription-Savvy Affluent New Parent"
echo "  - maya_rodriguez: Eco-Conscious Millennial Mom"
echo "  - lauren_peterson: Sleep-Deprived Premium Parent"
echo "  - jasmine_lee: Influencer-Following Social Mom"
echo "  - priya_desai: Convenience-First Urban Professional"
echo ""