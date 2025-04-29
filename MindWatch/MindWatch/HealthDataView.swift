import SwiftUI

struct HealthDataView: View {
    @EnvironmentObject private var authViewModel: AuthViewModel
    @EnvironmentObject private var healthViewModel: HealthViewModel
    @State private var isShowingHealthData = false
    @State private var showingDebugOptions = false
    
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
                            
                            // Add debug button (long press to activate)
                            Button(action: {
                                showingDebugOptions = true
                            }) {
                                Image(systemName: "gearshape")
                                    .foregroundColor(.gray)
                            }
                            .actionSheet(isPresented: $showingDebugOptions) {
                                ActionSheet(
                                    title: Text("Debug Options"),
                                    message: Text("Health data debugging tools"),
                                    buttons: [
                                        .default(Text("Test HealthKit Access")) {
                                            healthViewModel.testHealthKitAccess()
                                        },
                                        .cancel()
                                    ]
                                )
                            }
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
                            VStack(spacing: 8) {
                                // Progress bar
                                if healthViewModel.uploadProgress > 0 {
                                    VStack(alignment: .leading, spacing: 4) {
                                        Text(healthViewModel.uploadProgressMessage)
                                            .font(.caption)
                                            .foregroundColor(.gray)
                                        
                                        ProgressView(value: healthViewModel.uploadProgress, total: 1.0)
                                            .progressViewStyle(LinearProgressViewStyle())
                                            .padding(.vertical, 4)
                                        
                                        Text("\(Int(healthViewModel.uploadProgress * 100))% Complete")
                                            .font(.caption)
                                            .foregroundColor(.gray)
                                    }
                                    .padding(.top, 8)
                                } else {
                                    ProgressView("Collecting data...")
                                        .padding(.top, 8)
                                }
                            }
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
                       formatDuration(healthViewModel.workoutData.first!.value) : "No data"
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
        // Use the syncHealthData method directly
        healthViewModel.syncHealthData()
    }
    
    private func formatDate(_ date: Date) -> String {
        let formatter = DateFormatter()
        formatter.dateStyle = .medium
        formatter.timeStyle = .short
        return formatter.string(from: date)
    }
    
    private func sampleValue(from data: [HealthKitDataPoint], unit: String) -> String {
        if let first = data.first {
            return "\(String(format: "%.1f", first.value)) \(unit)"
        }
        return "No data"
    }
    
    private func formatDuration(_ minutes: Double) -> String {
        let totalMinutes = Int(minutes)
        if totalMinutes < 60 {
            return "\(totalMinutes) min"
        } else {
            let hours = totalMinutes / 60
            let remainingMinutes = totalMinutes % 60
            return "\(hours)h \(remainingMinutes)m"
        }
    }
} 