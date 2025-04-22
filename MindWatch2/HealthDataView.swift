import SwiftUI

struct HealthDataView: View {
    @EnvironmentObject private var authViewModel: AuthViewModel
    @EnvironmentObject private var healthViewModel: HealthViewModel
    @State private var isShowingHealthData = false
    
    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 20) {
                    // Health status card
                    VStack(spacing: 12) {
                        HStack {
                            Image(systemName: "heart.fill")
                                .foregroundColor(.red)
                                .font(.title)
                            
                            Text("Health Data")
                                .font(.title)
                                .fontWeight(.bold)
                            
                            Spacer()
                        }
                        
                        if let lastSync = healthViewModel.lastSyncDate {
                            Text("Last synced: \(formatDate(lastSync))")
                                .font(.subheadline)
                                .foregroundColor(.secondary)
                        } else {
                            Text("No data synced yet")
                                .font(.subheadline)
                                .foregroundColor(.secondary)
                        }
                        
                        Button(action: {
                            collectAndUploadData()
                        }) {
                            Text(healthViewModel.lastSyncDate == nil ? "Sync Health Data" : "Refresh Health Data")
                                .fontWeight(.bold)
                                .frame(maxWidth: .infinity)
                                .padding()
                                .background(Color.accentColor)
                                .foregroundColor(.white)
                                .cornerRadius(8)
                        }
                        .padding(.top, 8)
                        .disabled(healthViewModel.isLoading)
                        
                        if healthViewModel.isLoading {
                            ProgressView("Syncing data...")
                                .padding(.top, 8)
                        }
                        
                        if let errorMessage = healthViewModel.errorMessage {
                            Text(errorMessage)
                                .foregroundColor(.red)
                                .font(.caption)
                                .padding(.vertical, 8)
                        }
                    }
                    .padding()
                    .background(Color(.systemGray6))
                    .cornerRadius(12)
                    
                    // Health data summary
                    if isShowingHealthData {
                        healthDataSummary
                    }
                }
                .padding()
            }
            .navigationTitle("Health Data")
            .toolbar {
                Button(action: {
                    isShowingHealthData.toggle()
                }) {
                    Text(isShowingHealthData ? "Hide Summary" : "Show Summary")
                }
            }
        }
    }
    
    private var healthDataSummary: some View {
        VStack(spacing: 20) {
            // Heart rate data
            dataCard(
                title: "Heart Rate",
                systemImage: "heart.fill",
                color: .red,
                count: healthViewModel.heartRateData.count,
                sample: sampleValue(from: healthViewModel.heartRateData, unit: "bpm")
            )
            
            // Steps data
            dataCard(
                title: "Steps",
                systemImage: "figure.walk",
                color: .green,
                count: healthViewModel.stepData.count,
                sample: sampleValue(from: healthViewModel.stepData, unit: "steps")
            )
            
            // Active energy data
            dataCard(
                title: "Active Energy",
                systemImage: "flame.fill",
                color: .orange,
                count: healthViewModel.activeEnergyData.count,
                sample: sampleValue(from: healthViewModel.activeEnergyData, unit: "kcal")
            )
            
            // Sleep data
            dataCard(
                title: "Sleep",
                systemImage: "bed.double.fill",
                color: .blue,
                count: healthViewModel.sleepData.count,
                sample: sampleValue(from: healthViewModel.sleepData, unit: "min")
            )
            
            // Workout data
            dataCard(
                title: "Workouts",
                systemImage: "figure.run",
                color: .purple,
                count: healthViewModel.workoutData.count,
                sample: healthViewModel.workoutData.first != nil ? 
                       "\(formatDuration(Double(healthViewModel.workoutData.first!.duration) ?? 0))" : "No data"
            )
        }
    }
    
    private func dataCard(title: String, systemImage: String, color: Color, count: Int, sample: String) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Image(systemName: systemImage)
                    .foregroundColor(color)
                    .font(.title3)
                
                Text(title)
                    .font(.headline)
                
                Spacer()
                
                Text("\(count) records")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            
            Divider()
            
            Text("Sample: \(sample)")
                .font(.subheadline)
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(12)
    }
    
    private func collectAndUploadData() {
        healthViewModel.isLoading = true
        healthViewModel.errorMessage = nil
        
        // Step 1: Collect data from HealthKit
        healthViewModel.collectHealthData { success in
            if success {
                // Step 2: Upload data to backend
                healthViewModel.uploadHealthData(authToken: authViewModel.getAuthToken()) { uploadSuccess in
                    if uploadSuccess {
                        healthViewModel.errorMessage = nil
                    }
                }
            } else {
                healthViewModel.errorMessage = "No health data found to sync"
            }
        }
    }
    
    private func formatDate(_ date: Date) -> String {
        let formatter = DateFormatter()
        formatter.dateStyle = .medium
        formatter.timeStyle = .short
        return formatter.string(from: date)
    }
    
    private func sampleValue(from data: [HealthDataPoint], unit: String) -> String {
        if let first = data.first, let value = Double(first.value) {
            return "\(String(format: "%.1f", value)) \(unit)"
        }
        return "No data"
    }
    
    private func formatDuration(_ seconds: Double) -> String {
        let minutes = Int(seconds / 60)
        if minutes < 60 {
            return "\(minutes) min"
        } else {
            let hours = minutes / 60
            let remainingMinutes = minutes % 60
            return "\(hours)h \(remainingMinutes)m"
        }
    }
} 