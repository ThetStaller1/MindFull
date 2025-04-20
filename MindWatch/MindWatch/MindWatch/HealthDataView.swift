import SwiftUI
import SwiftData
import HealthKit

struct HealthDataView: View {
    @Environment(\.modelContext) private var modelContext
    @ObservedObject var viewModel: HealthViewModel
    
    var body: some View {
        NavigationStack {
            if viewModel.isAuthorized {
                healthDataView
            } else {
                requestAccessView
            }
        }
        .onAppear {
            viewModel.checkAuthorizationStatus()
        }
    }
    
    private var healthDataView: some View {
        VStack {
            if viewModel.errorMessage != nil {
                ErrorView(errorMessage: viewModel.errorMessage ?? "Unknown error")
            } else if viewModel.isLoading {
                LoadingView(viewModel: viewModel)
            } else {
                HealthDataContentView(viewModel: viewModel)
            }
        }
        .navigationTitle("Health Data")
        .toolbar {
            ToolbarItem {
                Button(action: {
                    viewModel.fetchAllHealthData()
                }) {
                    Label("Refresh", systemImage: "arrow.clockwise")
                }
            }
        }
    }
    
    private var requestAccessView: some View {
        AuthorizationView(requestAuthorization: viewModel.requestAuthorization)
    }
}

struct HealthDataContentView: View {
    @ObservedObject var viewModel: HealthViewModel
    
    var body: some View {
        ScrollView {
            VStack(spacing: 20) {
                // Summary Cards
                LazyVGrid(columns: [
                    GridItem(.flexible()),
                    GridItem(.flexible())
                ], spacing: 16) {
                    DataCard(title: "Heart Rate", value: viewModel.latestHeartRate, systemImage: "heart.fill", color: .red)
                    DataCard(title: "Steps", value: viewModel.todaySteps, systemImage: "figure.walk", color: .green)
                    DataCard(title: "Calories", value: viewModel.todayActiveEnergy, systemImage: "flame.fill", color: .orange)
                    DataCard(title: "Distance", value: viewModel.todayDistance, systemImage: "figure.hiking", color: .blue)
                }
                .padding(.horizontal)
                
                // Daily Summaries
                if !viewModel.heartRateData.isEmpty {
                    HealthMetricSummaryView(
                        title: "Heart Rate",
                        systemImage: "heart.fill",
                        color: .red,
                        value: averageHeartRate(),
                        unit: "bpm",
                        description: "Average heart rate"
                    )
                }
                
                if !viewModel.stepCountData.isEmpty {
                    HealthMetricSummaryView(
                        title: "Steps",
                        systemImage: "figure.walk",
                        color: .green,
                        value: totalSteps(),
                        unit: "steps",
                        description: "Last 7 days"
                    )
                }
                
                if !viewModel.activeEnergyData.isEmpty {
                    HealthMetricSummaryView(
                        title: "Active Energy",
                        systemImage: "flame.fill",
                        color: .orange,
                        value: totalActiveEnergy(),
                        unit: "kcal",
                        description: "Last 7 days"
                    )
                }
                
                if !viewModel.sleepData.isEmpty {
                    HealthMetricSummaryView(
                        title: "Sleep",
                        systemImage: "bed.double.fill",
                        color: .indigo,
                        value: averageSleepHours(),
                        unit: "hours",
                        description: "Average per night"
                    )
                }
                
                // Data Count Summary
                DataCountSummaryView(viewModel: viewModel)
            }
            .padding(.vertical)
        }
    }
    
    // Calculate metrics for summary views
    private func averageHeartRate() -> String {
        let rates = viewModel.heartRateData.prefix(200).map { $0.value }
        guard !rates.isEmpty else { return "N/A" }
        let average = rates.reduce(0, +) / Double(rates.count)
        return String(format: "%.0f", average)
    }
    
    private func totalSteps() -> String {
        let calendar = Calendar.current
        let lastWeek = calendar.date(byAdding: .day, value: -7, to: Date())!
        
        let weekSteps = viewModel.stepCountData
            .filter { $0.endDate >= lastWeek }
            .reduce(0) { $0 + $1.value }
        
        return String(format: "%.0f", weekSteps)
    }
    
    private func totalActiveEnergy() -> String {
        let calendar = Calendar.current
        let lastWeek = calendar.date(byAdding: .day, value: -7, to: Date())!
        
        let weekEnergy = viewModel.activeEnergyData
            .filter { $0.endDate >= lastWeek }
            .reduce(0) { $0 + $1.value }
        
        return String(format: "%.0f", weekEnergy)
    }
    
    private func averageSleepHours() -> String {
        // Group sleep data by day
        let calendar = Calendar.current
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd"
        
        // Calculate duration in minutes for each sleep sample
        let sleepSamples = viewModel.sleepData.map { data -> (String, Double) in
            let dateString = formatter.string(from: data.startDate)
            let duration = data.endDate.timeIntervalSince(data.startDate) / 60.0 // minutes
            return (dateString, duration)
        }
        
        // Group by day and sum durations
        let sleepByDay = Dictionary(grouping: sleepSamples) { $0.0 }
            .mapValues { samples in
                samples.reduce(0) { $0 + $1.1 }
            }
        
        // Calculate average hours per night
        guard !sleepByDay.isEmpty else { return "N/A" }
        let totalMinutes = sleepByDay.values.reduce(0, +)
        let averageMinutes = totalMinutes / Double(sleepByDay.count)
        let averageHours = averageMinutes / 60.0
        
        return String(format: "%.1f", averageHours)
    }
}

struct HealthMetricSummaryView: View {
    let title: String
    let systemImage: String
    let color: Color
    let value: String
    let unit: String
    let description: String
    
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Image(systemName: systemImage)
                    .foregroundColor(color)
                Text(title)
                    .font(.headline)
                Spacer()
            }
            
            HStack {
                Text(value)
                    .font(.largeTitle)
                    .fontWeight(.bold)
                
                Text(unit)
                    .font(.headline)
                    .foregroundColor(.secondary)
                    .padding(.leading, -4)
                
                Spacer()
            }
            
            Text(description)
                .font(.subheadline)
                .foregroundColor(.secondary)
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(Color(UIColor.systemBackground))
                .shadow(color: Color.black.opacity(0.1), radius: 5, x: 0, y: 2)
        )
        .padding(.horizontal)
    }
}

struct DataCountSummaryView: View {
    @ObservedObject var viewModel: HealthViewModel
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Data Points Collected")
                .font(.headline)
                .padding(.bottom, 4)
            
            dataTypeRow(
                icon: "heart.fill", 
                color: .red, 
                type: "Heart rate", 
                count: viewModel.heartRateData.count
            )
            
            dataTypeRow(
                icon: "figure.walk", 
                color: .green, 
                type: "Steps", 
                count: viewModel.stepCountData.count
            )
            
            dataTypeRow(
                icon: "flame.fill", 
                color: .orange, 
                type: "Active Energy", 
                count: viewModel.activeEnergyData.count
            )
            
            dataTypeRow(
                icon: "bed.double.fill", 
                color: .indigo, 
                type: "Sleep", 
                count: viewModel.sleepData.count
            )
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(Color(UIColor.systemBackground))
                .shadow(color: Color.black.opacity(0.1), radius: 5, x: 0, y: 2)
        )
        .padding(.horizontal)
    }
    
    private func dataTypeRow(icon: String, color: Color, type: String, count: Int) -> some View {
        HStack(spacing: 10) {
            Image(systemName: icon)
                .foregroundColor(color)
                .frame(width: 24, height: 24)
            
            Text(type)
                .font(.subheadline)
            
            Spacer()
            
            Text("\(count) records")
                .font(.subheadline)
                .foregroundColor(.secondary)
        }
    }
}

struct DataCard: View {
    let title: String
    let value: String
    let systemImage: String
    let color: Color
    
    var body: some View {
        VStack {
            HStack {
                Image(systemName: systemImage)
                    .foregroundColor(color)
                Text(title)
                    .font(.headline)
                Spacer()
            }
            .padding(.bottom, 4)
            
            HStack {
                Text(value)
                    .font(.title2)
                    .bold()
                Spacer()
            }
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(Color(UIColor.systemBackground))
                .shadow(color: Color.black.opacity(0.1), radius: 5, x: 0, y: 2)
        )
    }
}

struct LoadingView: View {
    @ObservedObject var viewModel: HealthViewModel
    
    var body: some View {
        VStack(spacing: 20) {
            // Circular progress indicator
            ZStack {
                Circle()
                    .stroke(
                        Color.secondary.opacity(0.2),
                        lineWidth: 10
                    )
                    .frame(width: 120, height: 120)
                
                Circle()
                    .trim(from: 0, to: CGFloat(viewModel.dataFetchProgress))
                    .stroke(
                        Color.blue,
                        style: StrokeStyle(
                            lineWidth: 10,
                            lineCap: .round
                        )
                    )
                    .frame(width: 120, height: 120)
                    .rotationEffect(.degrees(-90))
                    .animation(.easeInOut, value: viewModel.dataFetchProgress)
                
                VStack {
                    Text("\(Int(viewModel.dataFetchProgress * 100))%")
                        .font(.system(size: 28, weight: .bold, design: .rounded))
                    
                    Text("Complete")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            }
            .padding(.bottom, 10)
            
            // Linear progress bar (keep this for more visual feedback)
            VStack(spacing: 6) {
                ProgressView(value: viewModel.dataFetchProgress)
                    .progressViewStyle(LinearProgressViewStyle())
                    .frame(height: 6)
                    .padding(.horizontal)
                
                Text("Loading \(viewModel.currentDataType)")
                    .font(.headline)
                    .fontWeight(.medium)
            }
            
            if viewModel.errorMessage != nil {
                Text(viewModel.errorMessage ?? "")
                    .font(.caption)
                    .foregroundColor(.red)
                    .multilineTextAlignment(.center)
                    .padding()
            }
            
            Text("This may take a few minutes as we process your health data.")
                .font(.subheadline)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal)
            
            Text("Please keep the app in the foreground.")
                .font(.caption)
                .foregroundColor(.secondary.opacity(0.8))
                .padding(.top, -8)
        }
        .padding()
        .background(Color.secondary.opacity(0.05))
        .cornerRadius(16)
        .padding()
    }
}

struct ErrorView: View {
    let errorMessage: String
    
    var body: some View {
        VStack(spacing: 20) {
            Image(systemName: "exclamationmark.triangle.fill")
                .font(.system(size: 50))
                .foregroundColor(.red)
            
            Text("Error")
                .font(.title)
                .bold()
            
            Text(errorMessage)
                .multilineTextAlignment(.center)
                .padding(.horizontal)
            
            Text("Please ensure HealthKit permissions are enabled in Settings.")
                .font(.caption)
                .multilineTextAlignment(.center)
                .padding(.horizontal)
        }
        .padding()
    }
}

struct AuthorizationView: View {
    let requestAuthorization: () -> Void
    
    var body: some View {
        VStack(spacing: 20) {
            Image(systemName: "heart.text.square.fill")
                .font(.system(size: 70))
                .foregroundColor(.red)
            
            Text("Health Data Access Required")
                .font(.title2)
                .bold()
                .multilineTextAlignment(.center)
            
            Text("This app needs access to your health data to show your metrics. No data will be shared with third parties.")
                .multilineTextAlignment(.center)
                .padding(.horizontal)
            
            Button(action: requestAuthorization) {
                Text("Authorize HealthKit Access")
                    .padding()
                    .frame(maxWidth: .infinity)
                    .background(Color.blue)
                    .foregroundColor(.white)
                    .cornerRadius(10)
                    .padding(.horizontal)
            }
        }
        .padding()
    }
} 