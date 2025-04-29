#!/bin/bash

# Get the directory of the script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "📂 Script located at: $SCRIPT_DIR"

# Change to the directory containing the Xcode project
cd "$SCRIPT_DIR"
echo "📂 Changed to directory: $(pwd)"

# Check if Xcode is installed
if ! [ -d "/Applications/Xcode.app" ]; then
    echo "❌ Error: Xcode is not installed in /Applications"
    exit 1
fi

# Set Xcode as the active developer directory
# (Most ly working without this, then it don't require password. Uncomment if needed)
#sudo xcode-select --switch /Applications/Xcode.app/Contents/Developer

# Automatically detect the scheme
SCHEME=$(xcodebuild -list | sed -n '/Schemes:/,/Targets:/p' | sed -n '2p' | xargs)

if [ -z "$SCHEME" ]; then
    echo "❌ Error: No scheme found in the Xcode project."
    exit 1
fi

echo "🔍 Detected scheme: $SCHEME"

# List available simulators for debugging
echo "📱 Available simulators:"
xcrun simctl list devices | grep -E "iPhone|iPad"

# Find the first available iPhone simulator (either Shutdown or Booted)
echo "🔍 Searching for available iPhone simulators..."
SIMULATOR_INFO=$(xcrun simctl list devices available | grep -E "iPhone.*(Shutdown|Booted)" | grep -v "unavailable" | head -1)

echo "Debug - Full simulator info: $SIMULATOR_INFO"

if [ -z "$SIMULATOR_INFO" ]; then
    echo "❌ Error: No valid iPhone simulator found."
    exit 1
fi

# Extract the simulator ID (UUID) more carefully
SIMULATOR_ID=$(echo "$SIMULATOR_INFO" | grep -Eo '[A-Z0-9]{8}-[A-Z0-9]{4}-[A-Z0-9]{4}-[A-Z0-9]{4}-[A-Z0-9]{12}')

echo "Debug - Extracted simulator ID: $SIMULATOR_ID"

# Check if SIMULATOR_ID is valid
if [ -z "$SIMULATOR_ID" ]; then
    echo "❌ Error: Could not determine a valid simulator ID."
    exit 1
fi

# Check if simulator is already booted
if echo "$SIMULATOR_INFO" | grep -q "Shutdown"; then
    echo "🔄 Starting simulator: $SIMULATOR_ID"
    xcrun simctl boot "$SIMULATOR_ID"
else
    echo "✅ Simulator already booted: $SIMULATOR_ID"
fi

# Extract the full simulator name
SIMULATOR_NAME=$(echo "$SIMULATOR_INFO" | sed -E 's/^[[:space:]]*([^(]+).*$/\1/' | xargs)

# Get device info in JSON format
echo "Debug - Getting device info in JSON format:"
DEVICE_JSON=$(xcrun simctl list -j devices)

# Get the runtime for the device
SIMULATOR_RUNTIME=$(echo "$DEVICE_JSON" | python3 -c "
import sys, json
data = json.load(sys.stdin)
target_id = '$SIMULATOR_ID'
for runtime, devices in data['devices'].items():
    for device in devices:
        if device['udid'] == target_id:
            print(runtime)
            break
")

# Extract iOS version from runtime (e.g., com.apple.CoreSimulator.SimRuntime.iOS-18-2 -> 18.2)
SIMULATOR_OS=$(echo "$SIMULATOR_RUNTIME" | sed -E 's/.*iOS-([0-9]+)-([0-9]+).*/\1.\2/')

echo "Debug - Extracted name: '$SIMULATOR_NAME'"
echo "Debug - Extracted runtime: '$SIMULATOR_RUNTIME'"
echo "Debug - Extracted OS version: '$SIMULATOR_OS'"
echo "Debug - Full runtime list:"
xcrun simctl list runtimes
echo "Debug - Device JSON for simulator:"
echo "$DEVICE_JSON" | python3 -c "
import sys, json
data = json.load(sys.stdin)
target_id = '$SIMULATOR_ID'
for runtime, devices in data['devices'].items():
    for device in devices:
        if device['udid'] == target_id:
            print(f'Runtime: {runtime}')
            print(f'Device: {device}')
"

# Verify we have both name and OS version
if [ -z "$SIMULATOR_NAME" ] || [ -z "$SIMULATOR_OS" ]; then
    echo "❌ Error: Could not determine simulator name or OS version"
    echo "Name: $SIMULATOR_NAME"
    echo "OS: $SIMULATOR_OS"
    # Fallback to hard-coded values if needed
    SIMULATOR_NAME="iPhone 16 Pro"
    SIMULATOR_OS="18.2"
    echo "Using fallback values - Name: $SIMULATOR_NAME, OS: $SIMULATOR_OS"
fi

echo "📱 Using simulator: $SIMULATOR_NAME with iOS $SIMULATOR_OS"

# Function to perform build
perform_build() {
    local clean=$1
    local build_cmd="xcodebuild -scheme \"$SCHEME\" \
        -sdk iphonesimulator \
        -destination \"platform=iOS Simulator,OS=$SIMULATOR_OS,name=$SIMULATOR_NAME\""
    
    if [ "$clean" = true ]; then
        echo "🧹 Cleaning and rebuilding..."
        build_cmd="$build_cmd clean build"
    else
        echo "🏗️ Building project (using cache)..."
        build_cmd="$build_cmd build"
    fi
    
    echo "Running: $build_cmd"
    BUILD_OUTPUT=$(eval "$build_cmd" 2>&1)
    
    # Check for errors
    if echo "$BUILD_OUTPUT" | grep -q "error:"; then
        echo "❌ Build failed with errors:"
        echo "$BUILD_OUTPUT" | grep -A 5 "error:"
        return 1
    fi
    
    # Check for warnings
    if echo "$BUILD_OUTPUT" | grep -q "warning:"; then
        echo "⚠️ Build succeeded with warnings:"
        echo "$BUILD_OUTPUT" | grep -A 5 "warning:"
    fi
    
    echo "✅ Build succeeded!"
    return 0
}

# First try building without cleaning
perform_build false

# If the first build failed, try cleaning and rebuilding
if [ $? -ne 0 ]; then
    echo "🔄 Initial build failed, trying clean build..."
    perform_build true
    if [ $? -ne 0 ]; then
        exit 1
    fi
fi 