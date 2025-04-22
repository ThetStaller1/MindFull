#!/bin/bash

# MindWatch iOS App Build Script

# Stop on first error
set -e

echo "=== MindWatch iOS App Build Script ==="
echo

# Configuration
APP_NAME="MindWatch"
PROJECT_DIR="$(pwd)"
BACKEND_DIR="../Backend"
IOS_DIR="$PROJECT_DIR/MindWatch"

# Check if Xcode is installed
if ! command -v xcodebuild &> /dev/null; then
    echo "Error: Xcode not found. Please install Xcode to build the application."
    exit 1
fi

# Check if CocoaPods is installed
if ! command -v pod &> /dev/null; then
    echo "CocoaPods not found. Installing CocoaPods..."
    sudo gem install cocoapods
fi

# Create Podfile if it doesn't exist
if [ ! -f "$IOS_DIR/Podfile" ]; then
    echo "Creating Podfile..."
    cat > "$IOS_DIR/Podfile" << EOF
platform :ios, '15.0'

target 'MindWatch' do
  use_frameworks!
  
  # Pods for MindWatch
  pod 'Alamofire'
end
EOF
fi

# Install dependencies
echo "Installing dependencies..."
cd "$IOS_DIR"
pod install || true

# Create Xcode project if it doesn't exist
if [ ! -d "$IOS_DIR/MindWatch.xcodeproj" ]; then
    echo "Creating Xcode project..."
    
    # Create project structure
    mkdir -p "$IOS_DIR/MindWatch/Assets.xcassets"
    
    # Create project file
    touch "$IOS_DIR/MindWatch.xcodeproj"
    
    echo "Xcode project structure created. Please open Xcode and create a new project in this directory."
fi

# Set environment variables for the backend
echo "Checking backend environment..."
if [ ! -f "$BACKEND_DIR/.env" ]; then
    echo "Creating sample .env file for backend..."
    cat > "$BACKEND_DIR/.env" << EOF
# Supabase Configuration
SUPABASE_URL=YOUR_SUPABASE_URL
SUPABASE_ANON_KEY=YOUR_SUPABASE_ANON_KEY

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=True
EOF
    echo "Please update the .env file with your Supabase credentials."
fi

# Build the app
echo
echo "To build the iOS app, please open the Xcode project and build manually."
echo "Command to open Xcode project:"
echo "open $IOS_DIR/MindWatch.xcworkspace"
echo

# Start the backend server
echo "Would you like to start the backend server? (y/n)"
read -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Starting backend server..."
    cd "$BACKEND_DIR"
    
    # Check if Python is installed
    if ! command -v python3 &> /dev/null; then
        echo "Error: Python 3 not found. Please install Python 3."
        exit 1
    fi
    
    # Check if virtual environment exists
    if [ ! -d "venv" ]; then
        echo "Creating virtual environment..."
        python3 -m venv venv
    fi
    
    # Activate virtual environment and install dependencies
    echo "Installing backend dependencies..."
    source venv/bin/activate
    pip install -r requirements.txt
    
    # Start the server
    echo "Starting FastAPI server..."
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload
fi

echo "Done!" 