import SwiftUI

struct AnalysisView: View {
    @EnvironmentObject private var authViewModel: AuthViewModel
    @EnvironmentObject private var healthViewModel: HealthViewModel
    @State private var isRequestingAnalysis = false
    @State private var showingAllHistory = false
    
    // Colors for the modernized UI
    private let primaryColor = Color.blue
    private let accentColor = Color.purple
    private let backgroundColor = Color(.systemBackground)
    private let cardBackgroundColor = Color(.secondarySystemBackground)
    
    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 24) {
                    // Header card with modernized UI
                    headerCard
                    
                    // Analysis results
                    if let result = healthViewModel.analysisResult {
                        // Analysis results card
                        analysisResultCard(result: result)
                        
                        // Risk trend chart
                        riskTrendCard
                        
                        // Disclaimer with improved styling
                        disclaimerCard
                    }
                }
                .padding()
            }
            .background(backgroundColor.edgesIgnoringSafeArea(.all))
            .navigationTitle("Analysis")
            .onAppear {
                // Try to fetch latest analysis and history when view appears
                fetchLatestAnalysis()
                healthViewModel.fetchAnalysisHistory()
            }
        }
    }
    
    // MARK: - UI Components
    
    private var headerCard: some View {
        VStack(spacing: 16) {
            HStack(spacing: 16) {
                Image(systemName: "brain.head.profile")
                    .foregroundColor(primaryColor)
                    .font(.system(size: 36))
                    .frame(width: 60, height: 60)
                    .background(primaryColor.opacity(0.1))
                    .clipShape(Circle())
                
                VStack(alignment: .leading, spacing: 4) {
                    Text("Mental Health Disorder Risk Analysis")
                        .font(.headline)
                        .fontWeight(.bold)
                    
                    if let analysisResult = healthViewModel.analysisResult {
                        Text("Latest analysis: \(formatDate(analysisResult.analysisDate))")
                            .font(.subheadline)
                            .foregroundColor(.secondary)
                    } else {
                        Text("No analysis available")
                            .font(.subheadline)
                            .foregroundColor(.secondary)
                    }
                }
                
                Spacer()
            }
            
            Button(action: {
                requestAnalysis()
            }) {
                HStack {
                    Spacer()
                    
                    if isRequestingAnalysis {
                        ProgressView()
                            .progressViewStyle(CircularProgressViewStyle(tint: .white))
                            .padding(.trailing, 5)
                    }
                    
                    Text(healthViewModel.analysisResult == nil ? "Sync Health Data" : "Update Analysis")
                        .fontWeight(.semibold)
                    
                    Spacer()
                }
                .padding()
                .background(primaryColor)
                .foregroundColor(.white)
                .cornerRadius(15)
                .shadow(color: primaryColor.opacity(0.3), radius: 5, x: 0, y: 3)
            }
            .disabled(isRequestingAnalysis)
            
            if isRequestingAnalysis {
                Text("Processing health data...")
                    .font(.caption)
                    .foregroundColor(.secondary)
                    .padding(.top, 4)
            }
            
            if let errorMessage = healthViewModel.errorMessage {
                Text(errorMessage)
                    .foregroundColor(.red)
                    .font(.caption)
                    .padding(.top, 4)
            }
            
            // Show upload progress when syncing
            if healthViewModel.isLoading && healthViewModel.uploadProgress > 0 {
                VStack(spacing: 8) {
                    // Progress bar
                    GeometryReader { geometry in
                        ZStack(alignment: .leading) {
                            RoundedRectangle(cornerRadius: 10)
                                .frame(width: geometry.size.width, height: 8)
                                .opacity(0.3)
                                .foregroundColor(.gray)
                            
                            RoundedRectangle(cornerRadius: 10)
                                .frame(width: CGFloat(healthViewModel.uploadProgress) * geometry.size.width, height: 8)
                                .foregroundColor(primaryColor)
                        }
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
        .padding(20)
        .background(cardBackgroundColor)
        .cornerRadius(20)
        .shadow(color: Color.black.opacity(0.05), radius: 10, x: 0, y: 5)
    }
    
    private func analysisResultCard(result: AnalysisResult) -> some View {
        VStack(alignment: .leading, spacing: 20) {
            HStack {
                Text("Risk Assessment")
                    .font(.headline)
                    .fontWeight(.bold)
                
                Spacer()
                
                Text(result.riskLevel)
                    .fontWeight(.semibold)
                    .foregroundColor(result.riskLevel == "POSITIVE" ? .red : .green)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 6)
                    .background(
                        (result.riskLevel == "POSITIVE" ? Color.red : Color.green)
                            .opacity(0.2)
                    )
                    .cornerRadius(8)
            }
            
            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    Text("Current risk score:")
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                    
                    Spacer()
                    
                    Text(String(format: "%.1f%%", result.riskScore * 100))
                        .font(.title3)
                        .fontWeight(.bold)
                        .foregroundColor(riskColor(score: result.riskScore))
                }
                
                // Modern progress bar for risk score
                ZStack(alignment: .leading) {
                    RoundedRectangle(cornerRadius: 10)
                        .fill(Color.gray.opacity(0.1))
                        .frame(height: 12)
                    
                    // Gradient bar
                    GeometryReader { geometry in
                        RoundedRectangle(cornerRadius: 10)
                            .fill(
                                LinearGradient(
                                    gradient: Gradient(colors: [.green, .yellow, .orange, .red]),
                                    startPoint: .leading,
                                    endPoint: .trailing
                                )
                            )
                            .frame(width: min(CGFloat(result.riskScore) * geometry.size.width, geometry.size.width), height: 12)
                    }
                    
                    // Score indicator
                    GeometryReader { geometry in
                        Circle()
                            .fill(.white)
                            .frame(width: 20, height: 20)
                            .shadow(color: Color.black.opacity(0.1), radius: 2, x: 0, y: 1)
                            .position(x: CGFloat(result.riskScore) * geometry.size.width, y: 6)
                            .overlay(
                                Circle()
                                    .fill(riskColor(score: result.riskScore))
                                    .frame(width: 12, height: 12)
                                    .position(x: CGFloat(result.riskScore) * geometry.size.width, y: 6)
                            )
                    }
                }
                .frame(height: 12)
                
                HStack {
                    Text("Low Risk")
                        .font(.caption)
                        .foregroundColor(.green)
                    
                    Spacer()
                    
                    Text("High Risk")
                        .font(.caption)
                        .foregroundColor(.red)
                }
            }
        }
        .padding(20)
        .background(cardBackgroundColor)
        .cornerRadius(20)
        .shadow(color: Color.black.opacity(0.05), radius: 10, x: 0, y: 5)
    }
    
    private var riskTrendCard: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                Text("Risk Trend")
                    .font(.headline)
                    .fontWeight(.bold)
                
                Spacer()
                
                Button(action: {
                    showingAllHistory.toggle()
                }) {
                    Text(showingAllHistory ? "Show Recent" : "Show All")
                        .font(.caption)
                        .fontWeight(.medium)
                        .padding(.horizontal, 12)
                        .padding(.vertical, 6)
                        .background(Color.blue.opacity(0.1))
                        .foregroundColor(.blue)
                        .cornerRadius(8)
                }
            }
            .padding(.bottom, 16)
            
            if healthViewModel.analysisHistory.isEmpty {
                HStack {
                    Spacer()
                    VStack(spacing: 8) {
                        if healthViewModel.isLoading {
                            ProgressView()
                                .padding()
                            Text("Loading risk history...")
                                .font(.subheadline)
                                .foregroundColor(.secondary)
                        } else {
                            Image(systemName: "chart.line.uptrend.xyaxis")
                                .font(.system(size: 32))
                                .foregroundColor(.secondary)
                                .padding()
                            Text("No risk history available yet")
                                .font(.subheadline)
                                .foregroundColor(.secondary)
                            Text("Continue analyzing your health data to build a trend")
                                .font(.caption)
                                .foregroundColor(.secondary)
                                .multilineTextAlignment(.center)
                                .padding(.horizontal)
                        }
                    }
                    Spacer()
                }
                .padding()
            } else {
                // Risk trend chart
                GeometryReader { geometry in
                    VStack(spacing: 8) {
                        // Chart
                        ZStack(alignment: .leading) {
                            // Y-axis labels
                            VStack(alignment: .trailing, spacing: 0) {
                                Text("100%").font(.caption2).foregroundColor(.secondary)
                                Spacer()
                                Text("75%").font(.caption2).foregroundColor(.secondary)
                                Spacer()
                                Text("50%").font(.caption2).foregroundColor(.secondary)
                                Spacer()
                                Text("25%").font(.caption2).foregroundColor(.secondary)
                                Spacer()
                                Text("0%").font(.caption2).foregroundColor(.secondary)
                            }
                            .frame(width: 35)
                            .padding(.trailing, 5)
                            
                            // Main chart area with grid lines
                            HStack {
                                Spacer(minLength: 40)
                                
                                ZStack(alignment: .bottomLeading) {
                                    // Grid lines
                                    VStack(spacing: 0) {
                                        ForEach(0..<5) { i in
                                            Divider()
                                                .background(Color.gray.opacity(0.2))
                                            if i < 4 {
                                                Spacer()
                                            }
                                        }
                                    }
                                    
                                    // Risk zones background
                                    VStack(spacing: 0) {
                                        Rectangle()
                                            .fill(Color.red.opacity(0.1))
                                            .frame(height: geometry.size.height * 0.5) // 50%-100% (high risk)
                                        
                                        Rectangle()
                                            .fill(Color.yellow.opacity(0.1))
                                            .frame(height: geometry.size.height * 0.25) // 25%-50% (medium risk)
                                        
                                        Rectangle()
                                            .fill(Color.green.opacity(0.1))
                                            .frame(height: geometry.size.height * 0.25) // 0%-25% (low risk)
                                    }
                                    
                                    // Line chart
                                    Path { path in
                                        let history = getChartData()
                                        if !history.isEmpty {
                                            let height = geometry.size.height
                                            let width = geometry.size.width - 40 // Account for Y-axis labels
                                            let stepX = width / CGFloat(max(1, history.count - 1))
                                            
                                            let points = history.enumerated().map { (index, result) in
                                                CGPoint(
                                                    x: CGFloat(index) * stepX,
                                                    y: height - CGFloat(result.riskScore) * height
                                                )
                                            }
                                            
                                            path.move(to: points[0])
                                            for i in 1..<points.count {
                                                path.addLine(to: points[i])
                                            }
                                        }
                                    }
                                    .stroke(primaryColor, style: StrokeStyle(lineWidth: 3, lineCap: .round, lineJoin: .round))
                                    
                                    // Data points
                                    ForEach(getChartData().indices, id: \.self) { index in
                                        let result = getChartData()[index]
                                        let height = geometry.size.height
                                        let width = geometry.size.width - 40 // Account for Y-axis labels
                                        let stepX = width / CGFloat(max(1, getChartData().count - 1))
                                        
                                        ZStack {
                                            Circle()
                                                .fill(Color.white)
                                                .frame(width: 12, height: 12)
                                            
                                            Circle()
                                                .fill(riskColor(score: result.riskScore))
                                                .frame(width: 8, height: 8)
                                        }
                                        .position(
                                            x: CGFloat(index) * stepX,
                                            y: height - CGFloat(result.riskScore) * height
                                        )
                                        
                                        // Point value labels
                                        Text("\(Int(result.riskScore * 100))%")
                                            .font(.system(size: 8))
                                            .foregroundColor(.secondary)
                                            .padding(4)
                                            .background(Color(.systemBackground).opacity(0.8))
                                            .cornerRadius(4)
                                            .position(
                                                x: CGFloat(index) * stepX,
                                                y: max(15, (height - CGFloat(result.riskScore) * height) - 15)
                                            )
                                    }
                                }
                            }
                        }
                        .frame(height: 150)
                        .padding(.top, 8)
                        
                        // X-axis labels (dates)
                        HStack(spacing: 0) {
                            Spacer(minLength: 40)
                            ForEach(getChartData().indices, id: \.self) { index in
                                let result = getChartData()[index]
                                Text(formatShortDate(result.analysisDate))
                                    .font(.caption2)
                                    .foregroundColor(.secondary)
                                    .frame(maxWidth: .infinity)
                                    .rotationEffect(Angle(degrees: -45))
                                    .offset(y: 5)
                            }
                        }
                        .frame(height: 35)
                        
                        // Legend
                        HStack(spacing: 10) {
                            HStack(spacing: 4) {
                                Circle()
                                    .fill(Color.green)
                                    .frame(width: 8, height: 8)
                                Text("Low")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                            }
                            
                            HStack(spacing: 4) {
                                Circle()
                                    .fill(Color.yellow)
                                    .frame(width: 8, height: 8)
                                Text("Medium")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                            }
                            
                            HStack(spacing: 4) {
                                Circle()
                                    .fill(Color.red)
                                    .frame(width: 8, height: 8)
                                Text("High")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                            }
                        }
                        .padding(.top, 8)
                    }
                }
                .frame(height: 220)
            }
        }
        .padding(20)
        .background(cardBackgroundColor)
        .cornerRadius(20)
        .shadow(color: Color.black.opacity(0.05), radius: 10, x: 0, y: 5)
    }
    
    private var disclaimerCard: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Image(systemName: "exclamationmark.triangle.fill")
                    .foregroundColor(.red)
                
                Text("Important Notice")
                    .font(.headline)
                    .fontWeight(.semibold)
                    .foregroundColor(.red)
            }
            
            Text("This analysis is based on patterns in your health data and is not a medical diagnosis. If you are concerned about your mental health, please consult a healthcare professional.")
                .font(.footnote)
                .foregroundColor(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding(20)
        .background(cardBackgroundColor)
        .cornerRadius(20)
        .shadow(color: Color.black.opacity(0.05), radius: 10, x: 0, y: 5)
    }
    
    // MARK: - Helper Methods
    
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
    
    private func getChartData() -> [AnalysisResult] {
        let history = healthViewModel.analysisHistory
        if history.isEmpty {
            return []
        }
        
        // Show all history or just recent ones based on toggle
        if showingAllHistory {
            return history
        } else {
            // Show last 5 results
            return Array(history.prefix(min(5, history.count)))
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
    
    private func formatShortDate(_ dateString: String) -> String {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd'T'HH:mm:ss.SSSZ"
        
        if let date = formatter.date(from: dateString) {
            formatter.dateFormat = "MM/dd"
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
} 