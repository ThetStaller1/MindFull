import Foundation
import HealthKit
import Combine

class HealthViewModel: ObservableObject {
    private let healthStore = HKHealthStore()
    private let apiService = APIService.shared
    
    @Published var isLoading = false
    @Published var errorMessage: String?
    @Published var lastSyncDate: Date?
    @Published var analysisResult: AnalysisResult?
    @Published var analysisHistory: [AnalysisResult] = []
    
    // For HealthDataView
    @Published var uploadProgress: Double = 0.0
    @Published var uploadProgressMessage: String = ""
    
    // Sample health data for display
    @Published var heartRateData: [HealthKitDataPoint] = []
    @Published var stepData: [HealthKitDataPoint] = []
    @Published var activeEnergyData: [HealthKitDataPoint] = []
    @Published var sleepData: [HealthKitDataPoint] = []
    @Published var workoutData: [HealthKitDataPoint] = []
    
    // Types of health data we want to collect
    private let healthTypes: Set<HKSampleType> = [
        HKQuantityType(.heartRate),
        HKQuantityType(.stepCount),
        HKQuantityType(.activeEnergyBurned),
        HKQuantityType(.basalEnergyBurned),
        HKQuantityType(.oxygenSaturation),
        HKQuantityType(.flightsClimbed),
        HKCategoryType(.sleepAnalysis)
    ]
    
    // For workouts, we need to use HKWorkoutType
    private let workoutType = HKObjectType.workoutType()
    
    // Debug mode
    private let debugMode = false
    
    init() {
        requestHealthKitAuthorization()
    }
    
    func requestHealthKitAuthorization() {
        // Request authorization to access HealthKit data including workouts
        var typesToRead = healthTypes
        typesToRead.insert(workoutType)
        
        healthStore.requestAuthorization(toShare: nil, read: typesToRead) { success, error in
            DispatchQueue.main.async {
                if !success {
                    self.errorMessage = "Failed to get HealthKit authorization: \(error?.localizedDescription ?? "Unknown error")"
                }
            }
        }
    }
    
    func fetchLastSyncDate() {
        isLoading = true
        errorMessage = nil
        
        Task {
            do {
                let date = try await apiService.getLastSyncDate()
                DispatchQueue.main.async {
                    self.lastSyncDate = date
                    self.isLoading = false
                }
            } catch {
                DispatchQueue.main.async {
                    self.errorMessage = "Failed to fetch last sync date: \(error.localizedDescription)"
                    self.isLoading = false
                }
            }
        }
    }
    
    func syncHealthData() {
        isLoading = true
        errorMessage = nil
        uploadProgress = 0.0
        uploadProgressMessage = "Checking permissions..."
        
        // First verify HealthKit is available on this device
        guard HKHealthStore.isHealthDataAvailable() else {
            DispatchQueue.main.async {
                self.errorMessage = "HealthKit is not available on this device"
                self.isLoading = false
            }
            return
        }
        
        Task {
            do {
                // First confirm we have permissions
                let authorizationStatus = await checkHealthKitAuthorization()
                if !authorizationStatus {
                    throw NSError(domain: "MindWatch", code: 1001, userInfo: [NSLocalizedDescriptionKey: "Health data access not authorized"])
                }
                
                // Update progress
                DispatchQueue.main.async {
                    self.uploadProgressMessage = "Collecting health data..."
                    self.uploadProgress = 0.2
                }
                
                // Get the start date - either the last sync date or 60 days ago if we don't have a last sync
                let startDate = lastSyncDate ?? Calendar.current.date(byAdding: .day, value: -60, to: Date())!
                
                // Collect all health data since the last sync
                let healthData = try await collectHealthData(from: startDate)
                
                if healthData.isEmpty {
                    DispatchQueue.main.async {
                        self.uploadProgressMessage = "No new health data to sync. Fetching latest analysis..."
                        self.uploadProgress = 0.7
                    }
                    
                    // If no new data, fetch the latest analysis results
                    try await fetchLatestAnalysisAsync()
                    
                    DispatchQueue.main.async {
                        self.uploadProgress = 1.0
                        self.uploadProgressMessage = "Analysis retrieved!"
                        self.isLoading = false
                    }
                    return
                }
                
                // Update the sample data for display
                updateSampleHealthData(from: healthData)
                
                // Update progress
                DispatchQueue.main.async {
                    self.uploadProgress = 0.5
                    self.uploadProgressMessage = "Sending data to server..."
                }
                
                // Add a progress update for potentially long server processing
                let progressUpdateTask = Task {
                    var percentage = 0.5
                    var messageIndex = 0
                    let processingMessages = [
                        "Processing your health data...",
                        "Analyzing patterns in your data...",
                        "Almost there, finalizing analysis...",
                        "Still working, please wait..."
                    ]
                    
                    // Update progress periodically while the server processes data
                    while !Task.isCancelled {
                        // Wait a bit before updating
                        try await Task.sleep(nanoseconds: 5_000_000_000) // 5 seconds
                        
                        // Update progress and message
                        DispatchQueue.main.async {
                            // Increment progress slowly up to 95%
                            if percentage < 0.95 {
                                percentage += 0.05
                                self.uploadProgress = percentage
                            }
                            
                            // Cycle through messages
                            messageIndex = (messageIndex + 1) % processingMessages.count
                            self.uploadProgressMessage = processingMessages[messageIndex]
                        }
                    }
                }
                
                // Send health data to the backend
                do {
                    try await apiService.sendHealthData(healthData)
                    
                    // Cancel the progress update task as we're done
                    progressUpdateTask.cancel()
                    
                    // Update the last sync date to now
                    DispatchQueue.main.async {
                        self.lastSyncDate = Date()
                        self.uploadProgress = 0.9
                        self.uploadProgressMessage = "Data synced! Fetching latest analysis..."
                    }
                    
                    // Fetch the latest analysis results after syncing
                    try await fetchLatestAnalysisAsync()
                    
                    DispatchQueue.main.async {
                        self.uploadProgress = 1.0
                        self.uploadProgressMessage = "Sync complete!"
                        self.isLoading = false
                    }
                } catch {
                    // Cancel the progress update task if there's an error
                    progressUpdateTask.cancel()
                    throw error
                }
            } catch let error as APIError {
                DispatchQueue.main.async {
                    // Check if it's a timeout error
                    if case .serverError(let message) = error, message.contains("timed out") {
                        // Create a user-friendly timeout message
                        self.errorMessage = "The request took too long to complete. Your data is still being processed on the server. Please try again in a few minutes to check your results."
                    } else {
                        self.errorMessage = "Failed to sync health data. Error: \(error.description) (code: \(error.code))"
                    }
                    self.isLoading = false
                }
            } catch {
                DispatchQueue.main.async {
                    self.errorMessage = "Failed to sync health data: \(error.localizedDescription)"
                    self.isLoading = false
                }
            }
        }
    }
    
    private func checkHealthKitAuthorization() async -> Bool {
        return await withCheckedContinuation { continuation in
            // Check if we have all the permissions we need
            var typesToCheck = healthTypes
            typesToCheck.insert(workoutType)
            
            // Request authorization if needed
            healthStore.requestAuthorization(toShare: nil, read: typesToCheck) { success, error in
                if let error = error {
                    print("HealthKit authorization error: \(error.localizedDescription)")
                }
                continuation.resume(returning: success)
            }
        }
    }
    
    private func updateSampleHealthData(from healthData: [HealthKitDataPoint]) {
        // Group the health data by type
        var heartRate: [HealthKitDataPoint] = []
        var steps: [HealthKitDataPoint] = []
        var activeEnergy: [HealthKitDataPoint] = []
        var sleep: [HealthKitDataPoint] = []
        var workout: [HealthKitDataPoint] = []
        var flightsClimbed: [HealthKitDataPoint] = []
        
        for point in healthData {
            switch point.type {
            case .heartRate:
                heartRate.append(point)
            case .stepCount:
                steps.append(point)
            case .activeEnergy:
                activeEnergy.append(point)
            case .sleepAnalysis:
                sleep.append(point)
            case .workout:
                workout.append(point)
            case .flightsClimbed:
                flightsClimbed.append(point)
            default:
                break
            }
        }
        
        // Update on main thread
        DispatchQueue.main.async {
            self.heartRateData = heartRate
            self.stepData = steps
            self.activeEnergyData = activeEnergy
            self.sleepData = sleep
            self.workoutData = workout
        }
    }
    
    // Simulated methods for compatibility with HealthDataView
    func checkMissingDataTypes(authToken: String?, completion: @escaping (Bool) -> Void) {
        // Since we've updated our architecture, we'll assume we always need to sync
        completion(true)
    }
    
    func collectSelectedHealthData(completion: @escaping (Bool) -> Void) {
        // This is now handled by syncHealthData
        syncHealthData()
        completion(true)
    }
    
    func uploadHealthData(authToken: String?, completion: @escaping (Bool) -> Void) {
        // This is now part of syncHealthData
        completion(true)
    }
    
    func requestAnalysis() {
        isLoading = true
        errorMessage = nil
        
        Task {
            do {
                try await apiService.requestAnalysis()
                
                // Fetch the latest analysis after requesting a new one
                let analysis = try await apiService.getLatestAnalysis()
                
                DispatchQueue.main.async {
                    self.analysisResult = analysis
                    self.isLoading = false
                }
            } catch let error as APIError {
                DispatchQueue.main.async {
                    // Check if it's a timeout error
                    if case .serverError(let message) = error, message.contains("timed out") {
                        // Create a user-friendly timeout message
                        self.errorMessage = "The analysis request took too long to complete. Your analysis is still being processed on the server. Please try again in a few minutes to check your results."
                    } else {
                        self.errorMessage = "Failed to request analysis: \(error.description)"
                    }
                    self.isLoading = false
                }
            } catch {
                DispatchQueue.main.async {
                    self.errorMessage = "Failed to request analysis: \(error.localizedDescription)"
                    self.isLoading = false
                }
            }
        }
    }
    
    func fetchLatestAnalysis() {
        isLoading = true
        errorMessage = nil
        
        Task {
            do {
                let analysis = try await apiService.getLatestAnalysis()
                
                DispatchQueue.main.async {
                    self.analysisResult = analysis
                    self.isLoading = false
                }
            } catch let error as APIError {
                DispatchQueue.main.async {
                    // Check if it's a timeout error
                    if case .serverError(let message) = error, message.contains("timed out") {
                        // Create a user-friendly timeout message
                        self.errorMessage = "The request took too long to complete. Please try again in a few minutes."
                    } else {
                        self.errorMessage = "Failed to fetch latest analysis: \(error.description)"
                    }
                    self.isLoading = false
                }
            } catch {
                DispatchQueue.main.async {
                    self.errorMessage = "Failed to fetch latest analysis: \(error.localizedDescription)"
                    self.isLoading = false
                }
            }
        }
    }
    
    // Collect health data from HealthKit
    private func collectHealthData(from startDate: Date) async throws -> [HealthKitDataPoint] {
        var allHealthData: [HealthKitDataPoint] = []
        
        // Collect standard health data types
        for type in healthTypes {
            if let quantityType = type as? HKQuantityType {
                let healthData = try await fetchQuantityData(for: quantityType, from: startDate)
                allHealthData.append(contentsOf: healthData)
                print("DEBUG: Collected \(healthData.count) data points for \(quantityType.identifier)")
            } else if let categoryType = type as? HKCategoryType, categoryType.identifier == HKCategoryTypeIdentifier.sleepAnalysis.rawValue {
                let sleepData = try await fetchSleepData(from: startDate)
                allHealthData.append(contentsOf: sleepData)
                print("DEBUG: Added \(sleepData.count) sleep data points to collection")
            }
        }
        
        // Also collect workout data
        let workoutData = try await fetchWorkoutData(from: startDate)
        allHealthData.append(contentsOf: workoutData)
        print("DEBUG: Added \(workoutData.count) workout data points to collection")
        
        // Log summary of collected data
        var typeCounts: [String: Int] = [:]
        for point in allHealthData {
            typeCounts[point.type.rawValue] = (typeCounts[point.type.rawValue] ?? 0) + 1
        }
        print("DEBUG: HEALTH DATA SUMMARY - Total: \(allHealthData.count) points - \(typeCounts)")
        
        return allHealthData
    }
    
    // Fetch quantity data for heart rate, steps, energy, etc.
    private func fetchQuantityData(for quantityType: HKQuantityType, from startDate: Date) async throws -> [HealthKitDataPoint] {
        return try await withCheckedThrowingContinuation { continuation in
            let predicate = HKQuery.predicateForSamples(withStart: startDate, end: Date(), options: .strictStartDate)
            
            let query = HKSampleQuery(sampleType: quantityType, predicate: predicate, limit: HKObjectQueryNoLimit, sortDescriptors: nil) { _, samples, error in
                
                if let error = error {
                    continuation.resume(throwing: error)
                    return
                }
                
                guard let samples = samples else {
                    continuation.resume(returning: [])
                    return
                }
                
                let healthData = samples.compactMap { sample -> HealthKitDataPoint? in
                    guard let sample = sample as? HKQuantitySample else {
                        return nil
                    }
                    
                    let dataType = self.getDataType(for: quantityType)
                    let value = self.getValue(from: sample, for: quantityType)
                    let unit = self.getUnit(for: quantityType)
                    
                    return HealthKitDataPoint(
                        type: dataType,
                        timestamp: sample.startDate,
                        value: value,
                        unit: unit,
                        source: sample.sourceRevision.source.name
                    )
                }
                
                continuation.resume(returning: healthData)
            }
            
            healthStore.execute(query)
        }
    }
    
    // Fetch sleep data specifically
    private func fetchSleepData(from startDate: Date) async throws -> [HealthKitDataPoint] {
        print("DEBUG: Starting to fetch sleep data from \(startDate)")
        
        return try await withCheckedThrowingContinuation { continuation in
            let sleepType = HKCategoryType(.sleepAnalysis)
            let predicate = HKQuery.predicateForSamples(withStart: startDate, end: Date(), options: .strictStartDate)
            
            // Sort by start date to help identify sleep sessions
            let sortDescriptor = NSSortDescriptor(key: HKSampleSortIdentifierStartDate, ascending: true)
            
            let query = HKSampleQuery(sampleType: sleepType, predicate: predicate, limit: HKObjectQueryNoLimit, sortDescriptors: [sortDescriptor]) { _, samples, error in
                
                if let error = error {
                    print("DEBUG: Error fetching sleep data: \(error.localizedDescription)")
                    continuation.resume(throwing: error)
                    return
                }
                
                guard let samples = samples, !samples.isEmpty else {
                    print("DEBUG: No sleep samples found")
                    continuation.resume(returning: [])
                    return
                }
                
                print("DEBUG: Found \(samples.count) sleep samples")
                
                // Group samples into sleep sessions based on proximity in time
                var sleepSessions: [[HKCategorySample]] = []
                var currentSession: [HKCategorySample] = []
                var lastEndDate: Date?
                
                // First, sort samples by start date to ensure proper sequencing
                let sortedSamples = samples.compactMap { $0 as? HKCategorySample }.sorted { $0.startDate < $1.startDate }
                
                print("DEBUG: Processed \(sortedSamples.count) category samples")
                
                // Group samples into sessions (sleep entries within 30 minutes of each other are considered the same session)
                for sample in sortedSamples {
                    if let lastEnd = lastEndDate, sample.startDate.timeIntervalSince(lastEnd) > 30 * 60 {
                        // Gap of more than 30 minutes - this is a new session
                        if !currentSession.isEmpty {
                            sleepSessions.append(currentSession)
                            currentSession = []
                        }
                    }
                    
                    currentSession.append(sample)
                    lastEndDate = sample.endDate
                }
                
                // Add the last session if not empty
                if !currentSession.isEmpty {
                    sleepSessions.append(currentSession)
                }
                
                print("DEBUG: Identified \(sleepSessions.count) sleep sessions")
                
                // Process each session and create HealthKitDataPoint objects
                var sleepData: [HealthKitDataPoint] = []
                
                for (sessionIndex, session) in sleepSessions.enumerated() {
                    // Create a unique session ID based on the first entry's start time
                    let sessionStartTime = session.first!.startDate
                    let sessionID = "sleep_session_\(sessionIndex)_\(Int(sessionStartTime.timeIntervalSince1970))"
                    
                    print("DEBUG: Processing sleep session \(sessionIndex) with \(session.count) entries")
                    
                    for categorySample in session {
                        // Duration in minutes
                        let durationMinutes = categorySample.endDate.timeIntervalSince(categorySample.startDate) / 60
                        
                        // Get sleep stage from the value
                        let sleepStageValue = String(categorySample.value)
                        let sleepStage = HealthKitDataPoint.SleepStage.fromHealthKitValue(sleepStageValue)
                        
                        let dataPoint = HealthKitDataPoint(
                            type: .sleepAnalysis,
                            timestamp: categorySample.startDate,
                            endTimestamp: categorySample.endDate,
                            value: durationMinutes,
                            unit: "minutes",
                            sleepStage: sleepStage,
                            source: categorySample.sourceRevision.source.name,
                            sessionID: sessionID
                        )
                        
                        print("DEBUG: Sleep entry - Stage: \(sleepStage.rawValue), Duration: \(durationMinutes) minutes")
                        
                        sleepData.append(dataPoint)
                    }
                }
                
                print("DEBUG: Created \(sleepData.count) sleep data points")
                continuation.resume(returning: sleepData)
            }
            
            healthStore.execute(query)
        }
    }
    
    // New method to fetch workout data
    private func fetchWorkoutData(from startDate: Date) async throws -> [HealthKitDataPoint] {
        return try await withCheckedThrowingContinuation { continuation in
            let predicate = HKQuery.predicateForSamples(withStart: startDate, end: Date(), options: .strictStartDate)
            
            let query = HKSampleQuery(sampleType: workoutType, predicate: predicate, limit: HKObjectQueryNoLimit, sortDescriptors: nil) { _, samples, error in
                
                if let error = error {
                    continuation.resume(throwing: error)
                    return
                }
                
                guard let workouts = samples as? [HKWorkout] else {
                    continuation.resume(returning: [])
                    return
                }
                
                let healthData = workouts.map { workout -> HealthKitDataPoint in
                    let durationMinutes = workout.duration / 60 // Convert seconds to minutes
                    
                    return HealthKitDataPoint(
                        type: .workout,
                        timestamp: workout.startDate,
                        value: durationMinutes,
                        unit: "minutes",
                        source: workout.sourceRevision.source.name,
                        workoutActivityType: Int(workout.workoutActivityType.rawValue),
                        totalEnergyBurned: workout.totalEnergyBurned?.doubleValue(for: .kilocalorie()),
                        totalDistance: workout.totalDistance?.doubleValue(for: .meter())
                    )
                }
                
                continuation.resume(returning: healthData)
            }
            
            healthStore.execute(query)
        }
    }
    
    // Helper function to map HKQuantityType to our DataType enum
    private func getDataType(for quantityType: HKQuantityType) -> HealthKitDataPoint.DataType {
        switch quantityType.identifier {
        case HKQuantityTypeIdentifier.heartRate.rawValue:
            return .heartRate
        case HKQuantityTypeIdentifier.stepCount.rawValue:
            return .stepCount
        case HKQuantityTypeIdentifier.activeEnergyBurned.rawValue:
            return .activeEnergy
        case HKQuantityTypeIdentifier.basalEnergyBurned.rawValue:
            return .basalEnergy
        case HKQuantityTypeIdentifier.oxygenSaturation.rawValue:
            return .oxygenSaturation
        case HKQuantityTypeIdentifier.flightsClimbed.rawValue:
            return .flightsClimbed
        default:
            fatalError("Unexpected quantity type: \(quantityType.identifier)")
        }
    }
    
    // Helper function to get the appropriate value from health samples
    private func getValue(from sample: HKQuantitySample, for quantityType: HKQuantityType) -> Double {
        let unit = getHKUnit(for: quantityType)
        return sample.quantity.doubleValue(for: unit)
    }
    
    // Helper function to get the appropriate unit for a health type
    private func getHKUnit(for quantityType: HKQuantityType) -> HKUnit {
        switch quantityType.identifier {
        case HKQuantityTypeIdentifier.heartRate.rawValue:
            return HKUnit.count().unitDivided(by: HKUnit.minute())
            
        case HKQuantityTypeIdentifier.stepCount.rawValue:
            return HKUnit.count()
            
        case HKQuantityTypeIdentifier.activeEnergyBurned.rawValue,
             HKQuantityTypeIdentifier.basalEnergyBurned.rawValue:
            return HKUnit.kilocalorie()
            
        case HKQuantityTypeIdentifier.oxygenSaturation.rawValue:
            return HKUnit.percent()
            
        case HKQuantityTypeIdentifier.flightsClimbed.rawValue:
            return HKUnit.count()
            
        default:
            return HKUnit.count()
        }
    }
    
    // Helper function to get string representation of the unit
    private func getUnit(for quantityType: HKQuantityType) -> String {
        switch quantityType.identifier {
        case HKQuantityTypeIdentifier.heartRate.rawValue:
            return "bpm"
            
        case HKQuantityTypeIdentifier.stepCount.rawValue:
            return "count"
            
        case HKQuantityTypeIdentifier.activeEnergyBurned.rawValue,
             HKQuantityTypeIdentifier.basalEnergyBurned.rawValue:
            return "kcal"
            
        case HKQuantityTypeIdentifier.oxygenSaturation.rawValue:
            return "percent"
            
        case HKQuantityTypeIdentifier.flightsClimbed.rawValue:
            return "count"
            
        default:
            return "unknown"
        }
    }
    
    // MARK: - Debug Methods
    
    func testHealthKitAccess() {
        isLoading = true
        errorMessage = nil
        uploadProgress = 0.0
        uploadProgressMessage = "Testing HealthKit Access..."
        
        if !HKHealthStore.isHealthDataAvailable() {
            DispatchQueue.main.async {
                self.errorMessage = "HealthKit is not available on this device"
                self.isLoading = false
            }
            return
        }
        
        Task {
            do {
                let authorized = await checkHealthKitAuthorization()
                
                if authorized {
                    // Test reading a small amount of data
                    let startDate = Calendar.current.date(byAdding: .day, value: -7, to: Date())!
                    let testData = try await collectHealthData(from: startDate)
                    
                    let summary = createDataSummary(from: testData)
                    
                    DispatchQueue.main.async {
                        self.errorMessage = nil
                        self.isLoading = false
                        self.uploadProgress = 1.0
                        self.uploadProgressMessage = "HealthKit access test successful.\n\(summary)"
                    }
                } else {
                    DispatchQueue.main.async {
                        self.errorMessage = "HealthKit access not authorized"
                        self.isLoading = false
                    }
                }
            } catch {
                DispatchQueue.main.async {
                    self.errorMessage = "Error testing HealthKit access: \(error.localizedDescription)"
                    self.isLoading = false
                }
            }
        }
    }
    
    private func createDataSummary(from data: [HealthKitDataPoint]) -> String {
        var heartRate = 0
        var steps = 0
        var activeEnergy = 0
        var sleepAnalysis = 0
        var workout = 0
        var flightsClimbed = 0
        var other = 0
        
        for point in data {
            switch point.type {
            case .heartRate: heartRate += 1
            case .stepCount: steps += 1
            case .activeEnergy: activeEnergy += 1
            case .sleepAnalysis: sleepAnalysis += 1
            case .workout: workout += 1
            case .flightsClimbed: flightsClimbed += 1
            default: other += 1
            }
        }
        
        return """
        Found \(data.count) records in the last 7 days:
        - Heart Rate: \(heartRate)
        - Steps: \(steps)
        - Active Energy: \(activeEnergy)
        - Sleep: \(sleepAnalysis)
        - Workouts: \(workout)
        - Flights Climbed: \(flightsClimbed)
        - Other: \(other)
        """
    }
    
    // Helper function to fetch the latest analysis asynchronously
    private func fetchLatestAnalysisAsync() async throws {
        do {
            let analysis = try await apiService.getLatestAnalysis()
            
            // Use Task with MainActor to safely update the property
            await MainActor.run {
                self.analysisResult = analysis
            }
        } catch let error as APIError {
            // If it's a timeout error, add some context for this specific case
            if case .serverError(let message) = error, message.contains("timed out") {
                await MainActor.run {
                    self.errorMessage = "The analysis request took too long to complete. Please check your results later."
                }
                throw error  // Simply re-throw the original error instead of .timeoutError
            } else {
                throw error
            }
        }
    }
    
    // Combined function for Analysis tab to sync data and show analysis
    func syncAndShowAnalysis() {
        syncHealthData()
    }
    
    func fetchAnalysisHistory() {
        isLoading = true
        errorMessage = nil
        
        Task {
            do {
                let history = try await apiService.getAnalysisHistory()
                DispatchQueue.main.async {
                    self.analysisHistory = history.sorted { $0.analysisDate > $1.analysisDate }
                    self.isLoading = false
                }
            } catch let error as APIError {
                DispatchQueue.main.async {
                    if case .serverError(let message) = error, message.contains("404") {
                        // The endpoint might not exist yet, so we'll just set an empty array
                        self.analysisHistory = []
                        self.errorMessage = nil // Don't show error to user for missing endpoint
                    } else {
                        self.errorMessage = "Failed to fetch analysis history: \(error.description)"
                    }
                    self.isLoading = false
                }
            } catch {
                DispatchQueue.main.async {
                    self.errorMessage = "Failed to fetch analysis history: \(error.localizedDescription)"
                    self.isLoading = false
                }
            }
        }
    }
} 