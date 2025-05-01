import SwiftUI

struct AnalysisView: View {
    @EnvironmentObject var healthViewModel: HealthViewModel
    
    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 20) {
                    // Request analysis button
                    Button(action: {
                        healthViewModel.requestAnalysis()
                    }) {
                        HStack {
                            if healthViewModel.isAnalysisLoading {
                                ProgressView()
                                    .progressViewStyle(CircularProgressViewStyle(tint: .white))
                            } else {
                                Image(systemName: "waveform.path.ecg")
                            }
                            
                            Text("Generate Analysis")
                                .fontWeight(.semibold)
                        }
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(Color.blue)
                        .foregroundColor(.white)
                        .cornerRadius(10)
                    }
                    .padding(.horizontal)
                    .padding(.top)
                    .disabled(healthViewModel.isAnalysisLoading)
                    
                    // Error message
                    if let errorMessage = healthViewModel.analysisError {
                        Text(errorMessage)
                            .foregroundColor(.red)
                            .font(.caption)
                            .padding()
                    }
                    
                    // Results display
                    if let result = healthViewModel.analysisResult {
                        VStack(spacing: 15) {
                            // Mental health status card
                            VStack {
                                Text("Mental Health Status")
                                    .font(.headline)
                                    .padding(.top)
                                
                                HStack {
                                    Image(systemName: result.riskLevel == "POSITIVE" ? "exclamationmark.triangle.fill" : "checkmark.circle.fill")
                                        .foregroundColor(result.riskLevel == "POSITIVE" ? .orange : .green)
                                        .font(.system(size: 40))
                                    
                                    VStack(alignment: .leading) {
                                        Text(result.riskLevel == "POSITIVE" ? "At Risk" : "Healthy")
                                            .font(.system(size: 24, weight: .bold))
                                        
                                        Text(result.riskLevel == "POSITIVE" ? 
                                             "Your data shows signs of potential mental health concerns" : 
                                             "Your data indicates a healthy mental state")
                                            .font(.caption)
                                            .foregroundColor(.secondary)
                                    }
                                }
                                .padding()
                                
                                // Risk score
                                HStack {
                                    Text("Risk Score:")
                                    Text("\(Int(result.riskScore * 100))%")
                                        .fontWeight(.bold)
                                        .foregroundColor(result.riskLevel == "POSITIVE" ? .orange : .green)
                                }
                                .padding(.bottom)
                            }
                            .background(Color(.systemGray6))
                            .cornerRadius(12)
                            .padding(.horizontal)
                            
                            // Contributing factors
                            VStack(alignment: .leading) {
                                Text("Contributing Factors")
                                    .font(.headline)
                                    .padding(.horizontal)
                                
                                ScrollView(.horizontal, showsIndicators: false) {
                                    HStack(spacing: 15) {
                                        ForEach(Array(result.contributingFactors.keys.prefix(5)), id: \.self) { key in
                                            if let value = result.contributingFactors[key] {
                                                VStack {
                                                    ZStack {
                                                        Circle()
                                                            .stroke(Color.blue.opacity(0.3), lineWidth: 5)
                                                            .frame(width: 70, height: 70)
                                                        
                                                        Circle()
                                                            .trim(from: 0, to: CGFloat(value) * 2)
                                                            .stroke(Color.blue, lineWidth: 5)
                                                            .frame(width: 70, height: 70)
                                                            .rotationEffect(.degrees(-90))
                                                        
                                                        Text("\(Int(value * 100))%")
                                                            .font(.caption)
                                                            .bold()
                                                    }
                                                    
                                                    Text(formatFactorName(key))
                                                        .font(.caption)
                                                        .frame(width: 80)
                                                        .multilineTextAlignment(.center)
                                                }
                                                .padding(.bottom, 5)
                                            }
                                        }
                                    }
                                    .padding(.horizontal)
                                }
                            }
                            
                            // Analysis date
                            Text("Analysis: \(result.timestamp, formatter: dateFormatter)")
                                .font(.caption)
                                .foregroundColor(.secondary)
                                .padding(.top)
                        }
                    } else {
                        // No analysis yet
                        VStack(spacing: 20) {
                            Image(systemName: "doc.text.magnifyingglass")
                                .font(.system(size: 60))
                                .foregroundColor(.gray)
                            
                            Text("No Analysis Available")
                                .font(.headline)
                            
                            Text("Tap 'Generate Analysis' to analyze your health data for mental health insights.")
                                .multilineTextAlignment(.center)
                                .foregroundColor(.secondary)
                                .padding(.horizontal)
                        }
                        .padding(.vertical, 60)
                    }
                    
                    Spacer()
                }
            }
            .navigationTitle("Mental Health Analysis")
            .background(Color(.systemGroupedBackground).ignoresSafeArea())
        }
    }
    
    private var dateFormatter: DateFormatter {
        let formatter = DateFormatter()
        formatter.dateStyle = .medium
        formatter.timeStyle = .short
        return formatter
    }
    
    private func formatFactorName(_ name: String) -> String {
        // Split by underscore and capitalize each word
        let words = name.split(separator: "_")
        return words.map { word in
            let firstLetter = word.prefix(1).uppercased()
            let restOfWord = word.dropFirst()
            return firstLetter + restOfWord
        }.joined(separator: " ")
    }
}

struct AnalysisView_Previews: PreviewProvider {
    static var previews: some View {
        AnalysisView()
            .environmentObject(HealthViewModel())
    }
} 