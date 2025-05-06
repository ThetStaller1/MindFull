import SwiftUI

struct AnalysisView: View {
    @EnvironmentObject private var authViewModel: AuthViewModel
    @EnvironmentObject private var healthViewModel: HealthViewModel
    @State private var isRequestingAnalysis = false
    
    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 20) {
                    // Analysis status card
                    VStack(spacing: 12) {
                        HStack {
                            Image(systemName: "brain.head.profile")
                                .foregroundColor(.purple)
                                .font(.title)
                            
                            Text("Mental Health Analysis")
                                .font(.title)
                                .fontWeight(.bold)
                            
                            Spacer()
                        }
                        
                        if let analysisResult = healthViewModel.analysisResult {
                            Text("Analysis date: \(formatDate(analysisResult.analysisDate))")
                                .font(.subheadline)
                                .foregroundColor(.secondary)
                        } else {
                            Text("No analysis available")
                                .font(.subheadline)
                                .foregroundColor(.secondary)
                        }
                        
                        Button(action: {
                            requestAnalysis()
                        }) {
                            Text(healthViewModel.analysisResult == nil ? "Sync Health Data" : "Update Health Data & Analysis")
                                .fontWeight(.bold)
                                .frame(maxWidth: .infinity)
                                .padding()
                                .background(Color.accentColor)
                                .foregroundColor(.white)
                                .cornerRadius(8)
                        }
                        .padding(.top, 8)
                        .disabled(isRequestingAnalysis)
                        
                        if isRequestingAnalysis {
                            ProgressView("Processing health data...")
                                .padding(.top, 8)
                        }
                        
                        if let errorMessage = healthViewModel.errorMessage {
                            Text(errorMessage)
                                .foregroundColor(.red)
                                .font(.caption)
                                .padding(.vertical, 8)
                        }
                        
                        // Show upload progress when syncing
                        if healthViewModel.isLoading && healthViewModel.uploadProgress > 0 {
                            VStack(spacing: 4) {
                                // Progress bar
                                GeometryReader { geometry in
                                    ZStack(alignment: .leading) {
                                        Rectangle()
                                            .frame(width: geometry.size.width, height: 8)
                                            .opacity(0.3)
                                            .foregroundColor(.gray)
                                        
                                        Rectangle()
                                            .frame(width: CGFloat(healthViewModel.uploadProgress) * geometry.size.width, height: 8)
                                            .foregroundColor(.blue)
                                    }
                                    .cornerRadius(4)
                                }
                                .frame(height: 8)
                                
                                // Progress message
                                Text(healthViewModel.uploadProgressMessage)
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                            }
                            .padding(.top, 8)
                        }
                    }
                    .padding()
                    .background(Color(.systemGray6))
                    .cornerRadius(12)
                    
                    // Analysis results
                    if let result = healthViewModel.analysisResult {
                        analysisResultCard(result: result)
                        
                        // Contributing factors
                        if let featureImportance = result.contributingFactors.feature_importance, !featureImportance.isEmpty {
                            contributingFactorsCard(factors: featureImportance)
                        }
                        
                        // Disclaimer
                        VStack(alignment: .leading, spacing: 8) {
                            Text("Important Notice")
                                .font(.headline)
                                .foregroundColor(.red)
                            
                            Text("This analysis is based on patterns in your health data and is not a medical diagnosis. If you are concerned about your mental health, please consult a healthcare professional.")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                        .padding()
                        .background(Color(.systemGray6))
                        .cornerRadius(12)
                    }
                }
                .padding()
            }
            .navigationTitle("Analysis")
            .onAppear {
                // Try to fetch latest analysis when view appears
                fetchLatestAnalysis()
            }
        }
    }
    
    private func analysisResultCard(result: AnalysisResult) -> some View {
        VStack(alignment: .leading, spacing: 16) {
            VStack(alignment: .leading, spacing: 8) {
                Text("Risk Assessment")
                    .font(.headline)
                
                HStack {
                    Text("Mental health condition risk:")
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                    
                    Spacer()
                    
                    Text(result.riskLevel)
                        .fontWeight(.bold)
                        .foregroundColor(result.riskLevel == "POSITIVE" ? .red : .green)
                }
                
                HStack {
                    Text("Risk score:")
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                    
                    Spacer()
                    
                    Text(String(format: "%.1f%%", result.riskScore * 100))
                        .fontWeight(.bold)
                }
            }
            
            // Progress bar for risk score
            VStack(alignment: .leading, spacing: 4) {
                GeometryReader { geometry in
                    ZStack(alignment: .leading) {
                        Rectangle()
                            .frame(width: geometry.size.width, height: 12)
                            .opacity(0.3)
                            .foregroundColor(.gray)
                        
                        Rectangle()
                            .frame(width: min(CGFloat(result.riskScore) * geometry.size.width, geometry.size.width), height: 12)
                            .foregroundColor(riskColor(score: result.riskScore))
                    }
                    .cornerRadius(6)
                }
                .frame(height: 12)
                
                HStack {
                    Text("Low Risk")
                        .font(.caption)
                    
                    Spacer()
                    
                    Text("High Risk")
                        .font(.caption)
                }
            }
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(12)
    }
    
    private func contributingFactorsCard(factors: [String: Double]) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Contributing Factors")
                .font(.headline)
            
            Divider()
            
            ForEach(factors.sorted(by: { $0.value > $1.value }).prefix(5), id: \.key) { factor, importance in
                HStack {
                    Text(formatFactorName(factor))
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                    
                    Spacer()
                    
                    Text(String(format: "%.1f%%", importance * 100))
                        .fontWeight(.bold)
                }
            }
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(12)
    }
    
    private func requestAnalysis() {
        isRequestingAnalysis = true
        healthViewModel.errorMessage = nil
        
        Task {
            await MainActor.run {
                healthViewModel.syncAndShowAnalysis()
                isRequestingAnalysis = false
            }
        }
    }
    
    private func fetchLatestAnalysis() {
        if healthViewModel.analysisResult == nil {
            isRequestingAnalysis = true
            
            Task {
                await MainActor.run {
                    healthViewModel.fetchLatestAnalysis()
                    isRequestingAnalysis = false
                }
            }
        }
    }
    
    private func formatDate(_ dateString: String) -> String {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd'T'HH:mm:ss.SSSZ"
        
        if let date = formatter.date(from: dateString) {
            formatter.dateStyle = .medium
            formatter.timeStyle = .short
            return formatter.string(from: date)
        }
        
        return dateString
    }
    
    private func riskColor(score: Double) -> Color {
        if score < 0.25 {
            return .green
        } else if score < 0.5 {
            return .yellow
        } else if score < 0.75 {
            return .orange
        } else {
            return .red
        }
    }
    
    private func formatFactorName(_ name: String) -> String {
        // Convert snake_case to readable text
        let words = name.split(separator: "_")
        
        // Capitalize first letter of each word
        let capitalizedWords = words.map { word in
            if let firstChar = word.first {
                return String(firstChar).uppercased() + word.dropFirst()
            }
            return String(word)
        }
        
        return capitalizedWords.joined(separator: " ")
    }
} 