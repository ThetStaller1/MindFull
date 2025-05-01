import SwiftUI
import Charts

struct HealthDataView: View {
    @EnvironmentObject var healthViewModel: HealthViewModel
    
    var body: some View {
        NavigationView {
            List {
                if healthViewModel.isLoading {
                    Section {
                        HStack {
                            Spacer()
                            ProgressView("Loading data...")
                            Spacer()
                        }
                    }
                } else {
                    // Heart Rate Section
                    Section(header: Text("Heart Rate")) {
                        if healthViewModel.heartRateData.isEmpty {
                            Text("No heart rate data available")
                                .foregroundColor(.secondary)
                        } else {
                            HStack {
                                Text("Latest")
                                Spacer()
                                if let latest = healthViewModel.heartRateData.first {
                                    Text("\(Int(latest.value)) BPM")
                                        .foregroundColor(.red)
                                } else {
                                    Text("No data")
                                        .foregroundColor(.secondary)
                                }
                            }
                            
                            HStack {
                                Text("Readings")
                                Spacer()
                                Text("\(healthViewModel.heartRateData.count)")
                                    .foregroundColor(.secondary)
                            }
                        }
                    }
                    
                    // Activity Section
                    Section(header: Text("Activity")) {
                        if healthViewModel.activityData.isEmpty {
                            Text("No activity data available")
                                .foregroundColor(.secondary)
                        } else {
                            ForEach(healthViewModel.activityData.prefix(3)) { activity in
                                VStack(alignment: .leading, spacing: 4) {
                                    Text(activity.date, style: .date)
                                        .font(.headline)
                                    
                                    HStack {
                                        Image(systemName: "flame.fill")
                                            .foregroundColor(.orange)
                                        Text("\(Int(activity.activeCalories)) calories")
                                        
                                        Spacer()
                                        
                                        Image(systemName: "figure.walk")
                                            .foregroundColor(.green)
                                        Text("\(activity.stepCount) steps")
                                    }
                                    .font(.subheadline)
                                }
                                .padding(.vertical, 4)
                            }
                        }
                    }
                    
                    // Sleep Section
                    Section(header: Text("Sleep")) {
                        if healthViewModel.sleepData.isEmpty {
                            Text("No sleep data available")
                                .foregroundColor(.secondary)
                        } else {
                            ForEach(healthViewModel.sleepData.prefix(3)) { sleep in
                                VStack(alignment: .leading, spacing: 4) {
                                    Text(sleep.startDate, style: .date)
                                        .font(.headline)
                                    
                                    HStack {
                                        Image(systemName: "bed.double.fill")
                                            .foregroundColor(.blue)
                                        Text(String(format: "%.1f hours", sleep.duration))
                                        
                                        Spacer()
                                        
                                        Text(sleep.isAsleep ? "Asleep" : "In Bed")
                                            .foregroundColor(.secondary)
                                    }
                                    .font(.subheadline)
                                }
                                .padding(.vertical, 4)
                            }
                        }
                    }
                }
            }
            .navigationTitle("Health Data")
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button(action: {
                        healthViewModel.fetchHealthData()
                    }) {
                        Image(systemName: "arrow.clockwise")
                    }
                    .disabled(healthViewModel.isLoading)
                }
            }
            .refreshable {
                healthViewModel.fetchHealthData()
            }
        }
    }
}

struct HealthDataView_Previews: PreviewProvider {
    static var previews: some View {
        HealthDataView()
            .environmentObject(HealthViewModel())
    }
}

struct SleepDataView: View {
    @EnvironmentObject private var healthViewModel: HealthViewModel
    
    var body: some View {
        if healthViewModel.sleepData.isEmpty {
            ContentUnavailableView(
                "No Sleep Data",
                systemImage: "bed.double",
                description: Text("Sleep data will appear here after syncing with your Apple Watch.")
            )
        } else {
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    // Sleep Chart
                    VStack(alignment: .leading) {
                        Text("Sleep Duration (Last 7 Days)")
                            .font(.headline)
                            .padding(.horizontal)
                        
                        let last7DaysSleep = last7DaysSleepData()
                        
                        Chart(last7DaysSleep) { entry in
                            BarMark(
                                x: .value("Date", entry.day, unit: .day),
                                y: .value("Hours", entry.hours)
                            )
                            .foregroundStyle(Color.blue.gradient)
                        }
                        .chartXAxis {
                            AxisMarks(values: .stride(by: .day)) { _ in
                                AxisGridLine()
                                AxisTick()
                                AxisValueLabel(format: .dateTime.weekday())
                            }
                        }
                        .frame(height: 200)
                        .padding()
                    }
                    .background(Color.gray.opacity(0.1))
                    .cornerRadius(10)
                    .padding(.horizontal)
                    
                    // Sleep Quality Metrics
                    VStack(alignment: .leading, spacing: 15) {
                        Text("Sleep Quality Metrics")
                            .font(.headline)
                            .padding(.horizontal)
                        
                        HStack {
                            MetricView(
                                title: "Avg. Duration",
                                value: String(format: "%.1f hrs", averageSleepDuration()),
                                icon: "clock.fill",
                                color: .blue
                            )
                            
                            Divider()
                            
                            MetricView(
                                title: "Sleep Efficiency",
                                value: String(format: "%.1f%%", sleepEfficiency() * 100),
                                icon: "bed.double.fill",
                                color: .indigo
                            )
                        }
                        .padding()
                        .background(Color.gray.opacity(0.1))
                        .cornerRadius(10)
                        .padding(.horizontal)
                    }
                    
                    // Recent Sleep Data
                    VStack(alignment: .leading) {
                        Text("Recent Sleep Sessions")
                            .font(.headline)
                            .padding(.horizontal)
                        
                        ForEach(Array(healthViewModel.sleepData.sorted(by: { $0.startDate > $1.startDate }).prefix(5))) { entry in
                            HStack {
                                VStack(alignment: .leading) {
                                    Text(entry.startDate, style: .date)
                                        .font(.subheadline)
                                    
                                    HStack {
                                        Image(systemName: "clock")
                                            .foregroundColor(.gray)
                                        Text("\(entry.startDate.formatted(date: .omitted, time: .shortened)) - \(entry.endDate.formatted(date: .omitted, time: .shortened))")
                                            .font(.caption)
                                    }
                                }
                                
                                Spacer()
                                
                                VStack(alignment: .trailing) {
                                    Text(String(format: "%.1f hrs", entry.duration))
                                        .font(.headline)
                                    
                                    Text(entry.isAsleep ? "Asleep" : "In Bed")
                                        .font(.caption)
                                        .foregroundColor(entry.isAsleep ? .green : .orange)
                                }
                            }
                            .padding()
                            .background(Color.gray.opacity(0.05))
                            .cornerRadius(8)
                            .padding(.horizontal)
                        }
                    }
                }
                .padding(.vertical)
            }
        }
    }
    
    // Helper methods for sleep data
    private func last7DaysSleepData() -> [DailySleepEntry] {
        let calendar = Calendar.current
        let endDate = Date()
        let startDate = calendar.date(byAdding: .day, value: -6, to: endDate)!
        
        // Initialize with zero hours for all 7 days
        var dailyData: [Date: Double] = [:]
        for dayOffset in 0...6 {
            if let date = calendar.date(byAdding: .day, value: -dayOffset, to: endDate) {
                let dayStart = calendar.startOfDay(for: date)
                dailyData[dayStart] = 0
            }
        }
        
        // Aggregate sleep data by day
        for entry in healthViewModel.sleepData {
            let day = calendar.startOfDay(for: entry.startDate)
            if day >= startDate && day <= endDate {
                dailyData[day, default: 0] += entry.duration
            }
        }
        
        // Convert to array for Chart
        return dailyData.map { DailySleepEntry(day: $0.key, hours: $0.value) }
            .sorted { $0.day < $1.day }
    }
    
    private func averageSleepDuration() -> Double {
        guard !healthViewModel.sleepData.isEmpty else { return 0 }
        
        // Group by day to avoid counting multiple segments separately
        let calendar = Calendar.current
        var dailyDurations: [Date: Double] = [:]
        
        for entry in healthViewModel.sleepData {
            let day = calendar.startOfDay(for: entry.startDate)
            dailyDurations[day, default: 0] += entry.duration
        }
        
        let totalDuration = dailyDurations.values.reduce(0, +)
        return totalDuration / Double(dailyDurations.count)
    }
    
    private func sleepEfficiency() -> Double {
        let asleepEntries = healthViewModel.sleepData.filter { $0.isAsleep }
        let totalAsleepDuration = asleepEntries.reduce(0) { $0 + $1.duration }
        
        let totalDuration = healthViewModel.sleepData.reduce(0) { $0 + $1.duration }
        
        return totalDuration > 0 ? totalAsleepDuration / totalDuration : 0
    }
}

struct HeartRateDataView: View {
    @EnvironmentObject private var healthViewModel: HealthViewModel
    
    var body: some View {
        if healthViewModel.heartRateData.isEmpty {
            ContentUnavailableView(
                "No Heart Rate Data",
                systemImage: "heart",
                description: Text("Heart rate data will appear here after syncing with your Apple Watch.")
            )
        } else {
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    // Heart Rate Chart
                    VStack(alignment: .leading) {
                        Text("Heart Rate (Last 24 Hours)")
                            .font(.headline)
                            .padding(.horizontal)
                        
                        let recentHRData = recentHeartRateData()
                        
                        Chart(recentHRData) { entry in
                            LineMark(
                                x: .value("Time", entry.timestamp),
                                y: .value("BPM", entry.value)
                            )
                            .foregroundStyle(Color.red.gradient)
                            .interpolationMethod(.catmullRom)
                        }
                        .chartYScale(domain: [min(60, minHeartRate() - 5), maxHeartRate() + 5])
                        .frame(height: 200)
                        .padding()
                    }
                    .background(Color.gray.opacity(0.1))
                    .cornerRadius(10)
                    .padding(.horizontal)
                    
                    // Heart Rate Metrics
                    VStack(alignment: .leading, spacing: 15) {
                        Text("Heart Rate Metrics")
                            .font(.headline)
                            .padding(.horizontal)
                        
                        HStack {
                            MetricView(
                                title: "Average",
                                value: "\(Int(averageHeartRate())) BPM",
                                icon: "heart.fill",
                                color: .red
                            )
                            
                            Divider()
                            
                            MetricView(
                                title: "Max",
                                value: "\(Int(maxHeartRate())) BPM",
                                icon: "waveform.path.ecg.rectangle.fill",
                                color: .pink
                            )
                            
                            Divider()
                            
                            MetricView(
                                title: "Min",
                                value: "\(Int(minHeartRate())) BPM",
                                icon: "heart.circle.fill",
                                color: .purple
                            )
                        }
                        .padding()
                        .background(Color.gray.opacity(0.1))
                        .cornerRadius(10)
                        .padding(.horizontal)
                    }
                }
                .padding(.vertical)
            }
        }
    }
    
    // Helper methods for heart rate data
    private func recentHeartRateData() -> [HeartRateEntry] {
        let calendar = Calendar.current
        let endDate = Date()
        let startDate = calendar.date(byAdding: .hour, value: -24, to: endDate)!
        
        return healthViewModel.heartRateData
            .filter { $0.timestamp >= startDate && $0.timestamp <= endDate }
            .sorted { $0.timestamp < $1.timestamp }
    }
    
    private func averageHeartRate() -> Double {
        guard !healthViewModel.heartRateData.isEmpty else { return 0 }
        let total = healthViewModel.heartRateData.reduce(0) { $0 + $1.value }
        return total / Double(healthViewModel.heartRateData.count)
    }
    
    private func maxHeartRate() -> Double {
        healthViewModel.heartRateData.map { $0.value }.max() ?? 0
    }
    
    private func minHeartRate() -> Double {
        healthViewModel.heartRateData.map { $0.value }.min() ?? 0
    }
}

struct ActivityDataView: View {
    @EnvironmentObject private var healthViewModel: HealthViewModel
    
    var body: some View {
        if healthViewModel.activityData.isEmpty {
            ContentUnavailableView(
                "No Activity Data",
                systemImage: "figure.walk",
                description: Text("Activity data will appear here after syncing with your Apple Watch.")
            )
        } else {
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    // Steps Chart
                    VStack(alignment: .leading) {
                        Text("Daily Steps (Last 7 Days)")
                            .font(.headline)
                            .padding(.horizontal)
                        
                        let last7DaysActivity = last7DaysActivity()
                        
                        Chart(last7DaysActivity) { entry in
                            BarMark(
                                x: .value("Date", entry.date, unit: .day),
                                y: .value("Steps", entry.stepCount)
                            )
                            .foregroundStyle(Color.green.gradient)
                        }
                        .chartXAxis {
                            AxisMarks(values: .stride(by: .day)) { _ in
                                AxisGridLine()
                                AxisTick()
                                AxisValueLabel(format: .dateTime.weekday())
                            }
                        }
                        .frame(height: 200)
                        .padding()
                    }
                    .background(Color.gray.opacity(0.1))
                    .cornerRadius(10)
                    .padding(.horizontal)
                    
                    // Activity Metrics
                    VStack(alignment: .leading, spacing: 15) {
                        Text("Activity Metrics")
                            .font(.headline)
                            .padding(.horizontal)
                        
                        HStack {
                            MetricView(
                                title: "Avg. Steps",
                                value: "\(Int(averageSteps()))",
                                icon: "figure.walk",
                                color: .green
                            )
                            
                            Divider()
                            
                            MetricView(
                                title: "Avg. Calories",
                                value: "\(Int(averageCalories())) cal",
                                icon: "flame.fill",
                                color: .orange
                            )
                        }
                        .padding()
                        .background(Color.gray.opacity(0.1))
                        .cornerRadius(10)
                        .padding(.horizontal)
                    }
                    
                    // Recent Activity List
                    VStack(alignment: .leading) {
                        Text("Recent Activity")
                            .font(.headline)
                            .padding(.horizontal)
                        
                        ForEach(Array(healthViewModel.activityData.sorted(by: { $0.date > $1.date }).prefix(5))) { entry in
                            HStack {
                                VStack(alignment: .leading) {
                                    Text(entry.date, style: .date)
                                        .font(.subheadline)
                                    
                                    HStack {
                                        Image(systemName: "figure.walk")
                                            .foregroundColor(.gray)
                                        Text("\(entry.stepCount) steps")
                                            .font(.caption)
                                    }
                                }
                                
                                Spacer()
                                
                                VStack(alignment: .trailing) {
                                    Text("\(Int(entry.activeCalories)) cal")
                                        .font(.headline)
                                    
                                    Text("Active Calories")
                                        .font(.caption)
                                        .foregroundColor(.orange)
                                }
                            }
                            .padding()
                            .background(Color.gray.opacity(0.05))
                            .cornerRadius(8)
                            .padding(.horizontal)
                        }
                    }
                }
                .padding(.vertical)
            }
        }
    }
    
    // Helper methods for activity data
    private func last7DaysActivity() -> [ActivityEntry] {
        let calendar = Calendar.current
        let endDate = Date()
        let startDate = calendar.date(byAdding: .day, value: -6, to: endDate)!
        
        return healthViewModel.activityData
            .filter { $0.date >= startDate && $0.date <= endDate }
            .sorted { $0.date < $1.date }
    }
    
    private func averageSteps() -> Double {
        guard !healthViewModel.activityData.isEmpty else { return 0 }
        let total = healthViewModel.activityData.reduce(0) { $0 + $1.stepCount }
        return Double(total) / Double(healthViewModel.activityData.count)
    }
    
    private func averageCalories() -> Double {
        guard !healthViewModel.activityData.isEmpty else { return 0 }
        let total = healthViewModel.activityData.reduce(0.0) { $0 + $1.activeCalories }
        return total / Double(healthViewModel.activityData.count)
    }
}

// Helper Views
struct MetricView: View {
    let title: String
    let value: String
    let icon: String
    let color: Color
    
    var body: some View {
        VStack {
            Image(systemName: icon)
                .font(.title2)
                .foregroundColor(color)
            
            Text(value)
                .font(.headline)
            
            Text(title)
                .font(.caption)
                .foregroundColor(.gray)
        }
        .frame(maxWidth: .infinity)
    }
}

// Data models for charts
struct DailySleepEntry: Identifiable {
    var id = UUID()
    let day: Date
    let hours: Double
} 