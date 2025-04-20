import SwiftUI
import SwiftData

struct MentalHealthView: View {
    @ObservedObject var viewModel: HealthViewModel
    @State private var isRefreshing = false
    
    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    headerSection
                    
                    if viewModel.isAnalyzing {
                        loadingSection
                    } else if let error = viewModel.errorMessage {
                        errorSection(message: error)
                    } else if viewModel.analysisResult != nil {
                        resultSection
                    } else {
                        noDataSection
                    }
                }
                .padding()
            }
            .navigationTitle("Mental Health Analysis")
            .toolbar {
                ToolbarItem(placement: .primaryAction) {
                    Button(action: performAnalysis) {
                        Label("Analyze", systemImage: "brain")
                    }
                    .disabled(viewModel.isAnalyzing)
                }
            }
            .refreshable {
                isRefreshing = true
                viewModel.fetchAllHealthData()
                isRefreshing = false
            }
        }
    }
    
    private var headerSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Mental Wellbeing Assessment")
                .font(.title2)
                .fontWeight(.bold)
            
            Text("This analysis uses your health data to assess potential mental health concerns.")
                .font(.subheadline)
                .foregroundColor(.secondary)
        }
    }
    
    private var loadingSection: some View {
        VStack(spacing: 20) {
            if viewModel.isLoading {
                VStack(spacing: 12) {
                    ProgressView(value: viewModel.dataFetchProgress)
                        .progressViewStyle(LinearProgressViewStyle())
                        .frame(height: 8)
                    
                    Text("Fetching \(viewModel.currentDataType)...")
                        .font(.headline)
                    
                    Text("\(Int(viewModel.dataFetchProgress * 100))% Complete")
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                }
            } else {
                ProgressView()
                    .scaleEffect(1.5)
            }
            
            Text(viewModel.isLoading ? "Loading your health data (this may take a few minutes)" : "Analyzing your health data...")
                .font(.headline)
                .multilineTextAlignment(.center)
                .padding(.top, 10)
            
            Text(viewModel.isLoading 
                 ? "Fetching 60 days of health data for accurate analysis"
                 : "This may take a moment as we process your health patterns.")
                .font(.subheadline)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal)
        }
        .frame(maxWidth: .infinity)
        .padding()
        .background(Color.secondary.opacity(0.1))
        .cornerRadius(12)
    }
    
    private func errorSection(message: String) -> some View {
        VStack(spacing: 16) {
            Image(systemName: "exclamationmark.triangle")
                .font(.system(size: 40))
                .foregroundColor(.red)
            
            Text("Analysis Error")
                .font(.headline)
            
            Text(message)
                .font(.subheadline)
                .multilineTextAlignment(.center)
                .foregroundColor(.secondary)
            
            Button("Try Again") {
                performAnalysis()
            }
            .padding(.horizontal, 24)
            .padding(.vertical, 12)
            .background(Color.blue)
            .foregroundColor(.white)
            .cornerRadius(8)
        }
        .frame(maxWidth: .infinity)
        .padding()
        .background(Color.secondary.opacity(0.1))
        .cornerRadius(12)
    }
    
    private var resultSection: some View {
        VStack(spacing: 24) {
            riskLevelView
            dataSummaryView
            riskFactorsView
            recommendationsView
        }
    }
    
    private var riskLevelView: some View {
        VStack(spacing: 16) {
            Text("Risk Assessment")
                .font(.headline)
                .frame(maxWidth: .infinity, alignment: .leading)
            
            HStack(spacing: 20) {
                ZStack {
                    Circle()
                        .stroke(
                            Color.secondary.opacity(0.2),
                            lineWidth: 15
                        )
                    
                    let riskColor = viewModel.riskStatus == "POSITIVE" ? Color.red : Color.green
                    let riskScore = viewModel.analysisResult?.riskScore ?? 0
                    
                    Circle()
                        .trim(from: 0, to: CGFloat(riskScore))
                        .stroke(
                            riskColor,
                            style: StrokeStyle(
                                lineWidth: 15,
                                lineCap: .round
                            )
                        )
                        .rotationEffect(.degrees(-90))
                    
                    VStack {
                        Text(viewModel.riskScorePercentage)
                            .font(.title)
                            .fontWeight(.bold)
                        
                        Text("Risk Score")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                }
                .frame(width: 130, height: 130)
                
                VStack(alignment: .leading, spacing: 8) {
                    Group {
                        if viewModel.riskStatus == "POSITIVE" {
                            Text("Elevated Risk Detected")
                                .foregroundColor(.red)
                        } else {
                            Text("Low Risk Detected")
                                .foregroundColor(.green)
                        }
                    }
                    .font(.headline)
                    
                    Group {
                        if viewModel.riskStatus == "POSITIVE" {
                            Text("Your health data shows patterns that may be associated with mental health concerns.")
                        } else {
                            Text("Your health data shows patterns that are generally not associated with mental health concerns.")
                        }
                    }
                    .font(.subheadline)
                    .foregroundColor(.secondary)
                }
            }
        }
        .padding()
        .background(Color.secondary.opacity(0.1))
        .cornerRadius(12)
    }
    
    private var dataSummaryView: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Data Summary")
                .font(.headline)
            
            Text("Analysis based on data from the past 60 days:")
                .font(.subheadline)
                .foregroundColor(.secondary)
            
            VStack(spacing: 12) {
                // Heart Rate Data
                dataRowView(
                    icon: "heart.fill", 
                    color: .red, 
                    label: "Heart Rate", 
                    value: "\(viewModel.heartRateData.count) records"
                )
                
                // Steps Data
                dataRowView(
                    icon: "figure.walk", 
                    color: .green, 
                    label: "Steps", 
                    value: "\(viewModel.stepCountData.count) records"
                )
                
                // Sleep Data
                dataRowView(
                    icon: "bed.double.fill", 
                    color: .indigo, 
                    label: "Sleep", 
                    value: "\(viewModel.sleepData.count) records"
                )
                
                // Workouts Data
                dataRowView(
                    icon: "figure.run", 
                    color: .orange, 
                    label: "Workouts", 
                    value: "\(viewModel.workoutData.count) records"
                )
            }
            .padding(.vertical, 8)
        }
        .padding()
        .background(Color.secondary.opacity(0.1))
        .cornerRadius(12)
    }
    
    private func dataRowView(icon: String, color: Color, label: String, value: String) -> some View {
        HStack {
            Image(systemName: icon)
                .foregroundColor(color)
                .frame(width: 28, height: 28)
                .background(color.opacity(0.1))
                .clipShape(Circle())
            
            Text(label)
                .font(.system(.body, design: .rounded))
            
            Spacer()
            
            Text(value)
                .font(.system(.body, design: .rounded))
                .fontWeight(.medium)
                .foregroundColor(.primary.opacity(0.7))
        }
        .padding(.vertical, 4)
    }
    
    private var riskFactorsView: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Contributing Factors")
                .font(.headline)
            
            if viewModel.topContributingFactors.isEmpty {
                Text("No specific contributing factors identified.")
                    .font(.subheadline)
                    .foregroundColor(.secondary)
            } else {
                ForEach(viewModel.topContributingFactors, id: \.name) { factor in
                    HStack {
                        Text(factor.name)
                            .font(.subheadline)
                        
                        Spacer()
                        
                        Text("\(Int(factor.value * 100))%")
                            .font(.subheadline)
                            .foregroundColor(.secondary)
                    }
                    
                    ProgressView(value: factor.value)
                        .tint(Color.blue)
                        .scaleEffect(x: 1, y: 1.5, anchor: .center)
                }
            }
        }
        .padding()
        .background(Color.secondary.opacity(0.1))
        .cornerRadius(12)
    }
    
    private var recommendationsView: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Recommendations")
                .font(.headline)
            
            ForEach(getRecommendations(), id: \.self) { recommendation in
                HStack(alignment: .top, spacing: 12) {
                    Image(systemName: "checkmark.circle.fill")
                        .foregroundColor(.green)
                    
                    Text(recommendation)
                        .font(.subheadline)
                }
            }
            
            Divider()
            
            Text("Note: This is not a medical diagnosis. If you're concerned about your mental health, please consult with a healthcare professional.")
                .font(.caption)
                .foregroundColor(.secondary)
                .padding(.top, 8)
        }
        .padding()
        .background(Color.secondary.opacity(0.1))
        .cornerRadius(12)
    }
    
    private var noDataSection: some View {
        VStack(spacing: 20) {
            Image(systemName: "brain")
                .font(.system(size: 60))
                .foregroundColor(.blue)
                .padding(.top, 30)
            
            Text("Ready to Analyze")
                .font(.title3)
                .fontWeight(.bold)
            
            Text("Press the 'Analyze' button to assess your mental health based on your recent health data.")
                .multilineTextAlignment(.center)
                .font(.body)
                .foregroundColor(.secondary)
                .padding(.horizontal)
            
            Button(action: performAnalysis) {
                Text("Begin Analysis")
                    .fontWeight(.semibold)
                    .padding(.horizontal, 32)
                    .padding(.vertical, 12)
                    .background(Color.blue)
                    .foregroundColor(.white)
                    .cornerRadius(8)
            }
            .padding(.top, 10)
            .padding(.bottom, 30)
        }
        .frame(maxWidth: .infinity)
        .padding()
        .background(Color.secondary.opacity(0.1))
        .cornerRadius(12)
    }
    
    private func performAnalysis() {
        viewModel.performMentalHealthAnalysis()
    }
    
    private func getRecommendations() -> [String] {
        if viewModel.riskStatus == "POSITIVE" {
            return [
                "Consider speaking with a mental health professional for a proper evaluation.",
                "Focus on maintaining a regular sleep schedule.",
                "Try to increase daily physical activity, even with short walks.",
                "Practice stress reduction techniques like meditation or deep breathing.",
                "Stay connected with friends and family for social support."
            ]
        } else {
            return [
                "Continue your current healthy lifestyle patterns.",
                "Maintain your regular physical activity routine.",
                "Keep up your healthy sleep habits.",
                "Remember to take breaks and manage stress proactively.",
                "Schedule regular check-ins with healthcare providers."
            ]
        }
    }
}

#Preview {
    let container = try! ModelContainer(for: HealthData.self)
    let context = ModelContext(container)
    return MentalHealthView(viewModel: HealthViewModel(modelContext: context))
} 