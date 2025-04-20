import Foundation
import HealthKit
import SwiftData
import Combine

class HealthViewModel: ObservableObject {
    private let healthKitManager = HealthKitManager()
    private let apiService = APIService()
    
    @Published var heartRateData: [HealthData] = []
    @Published var stepCountData: [HealthData] = []
    @Published var activeEnergyData: [HealthData] = []
    @Published var distanceData: [HealthData] = []
    @Published var sleepData: [HealthData] = []
    @Published var basalEnergyData: [HealthData] = []
    @Published var flightsClimbedData: [HealthData] = []
    @Published var workoutData: [HealthData] = []
    
    @Published var isAuthorized = false
    @Published var errorMessage: String?
    @Published var isLoading = false
    @Published var dataFetchProgress: Double = 0.0
    @Published var currentDataType: String = ""
    @Published var isAnalyzing = false
    @Published var analysisResult: AnalysisResult?
    
    private let modelContext: ModelContext
    private var cancellables = Set<AnyCancellable>()
    private var progressUpdateTimer: Timer?
    
    init(modelContext: ModelContext) {
        self.modelContext = modelContext
        checkAuthorizationStatus()
    }
    
    deinit {
        progressUpdateTimer?.invalidate()
    }
    
    func checkAuthorizationStatus() {
        isAuthorized = healthKitManager.isAuthorized
        errorMessage = healthKitManager.errorMessage
    }
    
    func requestAuthorization() {
        isLoading = true
        healthKitManager.requestAuthorization { [weak self] success, error in
            guard let self = self else { return }
            
            self.isAuthorized = success
            self.errorMessage = error?.localizedDescription
            
            if success {
                self.startProgressUpdateTimer()
                self.fetchAllHealthData()
            } else {
                self.isLoading = false
            }
        }
    }
    
    private func startProgressUpdateTimer() {
        progressUpdateTimer?.invalidate()
        
        // Reset progress
        dataFetchProgress = 0.0
        healthKitManager.resetProgress()
        
        // Start a timer to update progress
        progressUpdateTimer = Timer.scheduledTimer(withTimeInterval: 0.5, repeats: true) { [weak self] _ in
            guard let self = self else { return }
            
            // Update progress from health kit manager
            DispatchQueue.main.async {
                self.dataFetchProgress = self.healthKitManager.overallDataFetchProgress
                
                // Detect when fetching is complete
                if self.dataFetchProgress >= 0.99 {
                    self.progressUpdateTimer?.invalidate()
                    self.progressUpdateTimer = nil
                    self.dataFetchProgress = 1.0
                }
            }
        }
    }
    
    func fetchAllHealthData() {
        isLoading = true
        currentDataType = "Preparing"
        errorMessage = nil
        
        // Start progress updates
        startProgressUpdateTimer()
        
        // Clear any existing data to free up memory
        clearExistingData()
        
        // Use a sequential approach instead of loading everything in parallel
        fetchDataSequentially()
    }
    
    private func clearExistingData() {
        // Clear existing arrays to free up memory
        heartRateData = []
        stepCountData = []
        activeEnergyData = []
        distanceData = []
        sleepData = []
        basalEnergyData = []
        flightsClimbedData = []
        workoutData = []
        
        // Try to clean up SwiftData context
        try? modelContext.save()
    }
    
    private func fetchDataSequentially() {
        // Create an array of data fetch operations
        let fetchOperations: [(String, (@escaping ([HealthData]) -> Void) -> Void)] = [
            ("Heart Rate", healthKitManager.fetchHeartRateData),
            ("Steps", healthKitManager.fetchStepCountData),
            ("Active Energy", healthKitManager.fetchActiveEnergyData),
            ("Distance", healthKitManager.fetchDistanceData),
            ("Basal Energy", healthKitManager.fetchBasalEnergyData),
            ("Flights Climbed", healthKitManager.fetchFlightsClimbedData),
            ("Workouts", healthKitManager.fetchWorkoutData),
            ("Sleep", fetchSleepDataWithStreamHandling)
        ]
        
        // Run the operations sequentially
        fetchNextDataType(operations: fetchOperations, index: 0)
    }
    
    // Custom handler for sleep data to process it as a stream
    private func fetchSleepDataWithStreamHandling(completion: @escaping ([HealthData]) -> Void) {
        // Create a temporary array to collect accumulated sleep data
        var allSleepData: [HealthData] = []
        var isComplete = false
        
        // Handle sleep data in batches to avoid memory buildup
        healthKitManager.fetchSleepData { [weak self] newBatch in
            guard let self = self else { return }
            
            // Check if this is the completion marker
            if newBatch.count == 1 && newBatch[0].type == "sleep_complete_marker" {
                isComplete = true
            } else {
                // Add this batch to our running total and save to SwiftData in small chunks
                self.saveToSwiftData(newBatch)
                
                // Only keep a reasonable number of samples in memory for UI display
                if allSleepData.count < 100 {
                    allSleepData.append(contentsOf: newBatch)
                }
            }
            
            // If complete, update the published property and call completion
            if isComplete {
                DispatchQueue.main.async {
                    self.sleepData = allSleepData
                    completion(allSleepData) // This signals to move to the next data type
                }
            }
        }
    }
    
    private func fetchNextDataType(operations: [(String, (@escaping ([HealthData]) -> Void) -> Void)], index: Int) {
        // Check if we've completed all operations
        if index >= operations.count {
            // All done
            DispatchQueue.main.async {
                self.progressUpdateTimer?.invalidate()
                self.progressUpdateTimer = nil
                self.dataFetchProgress = 1.0
                self.currentDataType = "Complete"
                self.isLoading = false
            }
            return
        }
        
        // Get the current operation
        let (dataTypeName, fetchFunction) = operations[index]
        
        // Update UI
        DispatchQueue.main.async {
            self.currentDataType = dataTypeName
        }
        
        // Execute the fetch operation
        fetchFunction { [weak self] healthData in
            guard let self = self else { return }
            
            // Process the results based on the data type
            DispatchQueue.main.async {
                switch index {
                case 0: self.heartRateData = healthData
                case 1: self.stepCountData = healthData
                case 2: self.activeEnergyData = healthData
                case 3: self.distanceData = healthData
                case 4: self.basalEnergyData = healthData
                case 5: self.flightsClimbedData = healthData
                case 6: self.workoutData = healthData
                case 7: break // Sleep data is already handled in fetchSleepDataWithStreamHandling
                default: break
                }
                
                // Save the data to SwiftData unless it's sleep data (already saved in stream)
                if index != 7 {
                    self.saveToSwiftData(healthData)
                }
                
                // Continue with the next operation
                self.fetchNextDataType(operations: operations, index: index + 1)
            }
        }
    }
    
    private func saveToSwiftData(_ healthData: [HealthData]) {
        // Save data in smaller batches to avoid high memory usage
        let batchSize = 50 // Reduced from 100 to 50
        
        for i in stride(from: 0, to: healthData.count, by: batchSize) {
            let endIndex = min(i + batchSize, healthData.count)
            let batch = healthData[i..<endIndex]
            
            autoreleasepool {
                for data in batch {
                    modelContext.insert(data)
                }
                
                do {
                    try modelContext.save()
                } catch {
                    errorMessage = "Failed to save data: \(error.localizedDescription)"
                }
            }
        }
    }
    
    // MARK: - Mental Health Analysis
    
    func performMentalHealthAnalysis() {
        isAnalyzing = true
        errorMessage = nil
        
        // Check server health first
        apiService.checkServerHealth()
            .sink(receiveCompletion: { [weak self] completion in
                if case .failure(let error) = completion {
                    DispatchQueue.main.async {
                        self?.errorMessage = "Server connection error: \(error.description)"
                        self?.isAnalyzing = false
                    }
                }
            }, receiveValue: { [weak self] isHealthy in
                if isHealthy {
                    self?.sendHealthDataForAnalysis()
                } else {
                    DispatchQueue.main.async {
                        self?.errorMessage = "Server reported unhealthy status"
                        self?.isAnalyzing = false
                    }
                }
            })
            .store(in: &cancellables)
    }
    
    private func sendHealthDataForAnalysis() {
        // This operation might be heavy - do the data preparation on a background queue
        // but ensure ModelContext operations stay on the main thread
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self = self else { return }
            
            // Prepare the data payload on the background queue
            let healthDataPayload = self.prepareHealthDataForAPI()
            
            // Send the prepared data to the API
            self.apiService.analyzeHealthData(healthData: healthDataPayload)
                .receive(on: DispatchQueue.main) // Ensure we receive on main thread
                .sink(receiveCompletion: { [weak self] completion in
                    DispatchQueue.main.async {
                        self?.isAnalyzing = false
                        
                        if case .failure(let error) = completion {
                            self?.errorMessage = "Analysis failed: \(error.description)"
                            print("API error: \(error)")
                        }
                    }
                }, receiveValue: { [weak self] result in
                    DispatchQueue.main.async {
                        self?.analysisResult = result
                        print("Analysis complete: \(result)")
                    }
                })
                .store(in: &self.cancellables)
        }
    }
    
    // Separate data preparation from ModelContext access
    private func prepareHealthDataForAPI() -> [String: Any] {
        // Execute on the main thread to avoid ModelContext threading issues
        let result = DispatchQueue.main.sync { [weak self] () -> [String: Any] in
            guard let self = self else { return [:] }
            
            // Get data from SwiftData (must be on main thread)
            let heartRateAll = self.getAllDataOfType(HKQuantityTypeIdentifier.heartRate.rawValue)
            let stepsAll = self.getAllDataOfType(HKQuantityTypeIdentifier.stepCount.rawValue)
            let activeEnergyAll = self.getAllDataOfType(HKQuantityTypeIdentifier.activeEnergyBurned.rawValue)
            let sleepAll = self.getAllDataOfType("sleep")
            let workoutAll = self.getAllDataOfType("HKWorkoutTypeIdentifier")
            let distanceAll = self.getAllDataOfType(HKQuantityTypeIdentifier.distanceWalkingRunning.rawValue)
            let basalEnergyAll = self.getAllDataOfType(HKQuantityTypeIdentifier.basalEnergyBurned.rawValue)
            let flightsClimbedAll = self.getAllDataOfType(HKQuantityTypeIdentifier.flightsClimbed.rawValue)
            
            // Log the counts for debugging
            print("Sending to API - Records count by type:")
            print("Heart Rate: \(heartRateAll.count)")
            print("Steps: \(stepsAll.count)")
            print("Active Energy: \(activeEnergyAll.count)")
            print("Sleep: \(sleepAll.count)")
            print("Workouts: \(workoutAll.count)")
            print("Distance: \(distanceAll.count)")
            print("Basal Energy: \(basalEnergyAll.count)")
            print("Flights Climbed: \(flightsClimbedAll.count)")
            
            // Convert data to dictionaries
            let heartRateDicts = self.convertToArray(heartRateAll)
            let stepsDicts = self.convertToArray(stepsAll)
            let activeEnergyDicts = self.convertToArray(activeEnergyAll)
            let sleepDicts = self.convertToArray(sleepAll)
            let workoutDicts = self.convertToArray(workoutAll)
            let distanceDicts = self.convertToArray(distanceAll)
            let basalEnergyDicts = self.convertToArray(basalEnergyAll)
            let flightsClimbedDicts = self.convertToArray(flightsClimbedAll)
            
            // Create user info
            let userInfo: [String: Any] = [
                "personId": "1001",  // Default ID
                "age": 33,           // Default age
                "genderBinary": 1    // Default female (1)
            ]
            
            // Format payload with all data types
            return [
                "heartRate": heartRateDicts,
                "steps": stepsDicts,
                "activeEnergy": activeEnergyDicts, 
                "sleep": sleepDicts,
                "workout": workoutDicts,
                "distance": distanceDicts,
                "basalEnergy": basalEnergyDicts,
                "flightsClimbed": flightsClimbedDicts,
                "userInfo": userInfo
            ]
        }
        
        return result
    }
    
    // Convert HealthData to dictionary array - safe to call from any thread
    private func convertToArray(_ data: [HealthData]) -> [[String: Any]] {
        return data.map { item in
            let formatter = ISO8601DateFormatter()
            formatter.formatOptions = [.withInternetDateTime]
            
            return [
                "type": item.type,
                "startDate": formatter.string(from: item.startDate),
                "endDate": formatter.string(from: item.endDate),
                "value": item.value,
                "unit": item.unit
            ]
        }
    }
    
    // MARK: - Safe SwiftData Access
    
    // This should only be called from the main thread
    private func getAllDataOfType(_ type: String, limit: Int = 1000000) -> [HealthData] {
        assert(Thread.isMainThread, "getAllDataOfType must be called from the main thread")
        
        do {
            // Create a predicate to filter by type
            let predicate = #Predicate<HealthData> { $0.type == type }
            
            // Create a descriptor to sort by date
            let sortDescriptor = SortDescriptor<HealthData>(\.startDate, order: .forward)
            
            // Fetch from SwiftData with pagination to handle large datasets
            var allResults: [HealthData] = []
            var offset = 0
            let batchSize = 500
            
            while true {
                // Create the descriptor with only the supported parameters
                var fetchDescriptor = FetchDescriptor<HealthData>(
                    predicate: predicate,
                    sortBy: [sortDescriptor]
                )
                
                // Set the limit and offset properties after initialization
                fetchDescriptor.fetchLimit = batchSize
                fetchDescriptor.fetchOffset = offset
                
                let batch = try modelContext.fetch(fetchDescriptor)
                allResults.append(contentsOf: batch)
                
                if batch.count < batchSize || allResults.count >= limit {
                    break
                }
                
                offset += batchSize
            }
            
            return allResults
        } catch {
            print("Error fetching data from SwiftData: \(error)")
            return []
        }
    }
    
    // MARK: - Model Context Update
    
    // Add a method to update the model context
    func updateModelContext(_ newContext: ModelContext) {
        assert(Thread.isMainThread, "updateModelContext must be called on the main thread")
        
        // First, try to save any pending changes in the current context
        do {
            try modelContext.save()
        } catch {
            errorMessage = "Error saving model context: \(error.localizedDescription)"
        }
        
        // Log that the context would be updated (actual implementation would need to handle transferring data)
        print("Model context would be updated here")
    }
    
    // MARK: - Data Accessors
    
    // Get latest heart rate value
    var latestHeartRate: String {
        guard let first = heartRateData.first else { return "No data" }
        return "\(Int(first.value)) bpm"
    }
    
    // Get total steps for today
    var todaySteps: String {
        let calendar = Calendar.current
        
        let todaySteps = stepCountData
            .filter { calendar.isDate($0.endDate, inSameDayAs: Date()) }
            .reduce(0) { $0 + $1.value }
        
        return "\(Int(todaySteps)) steps"
    }
    
    // Get total active energy for today
    var todayActiveEnergy: String {
        let calendar = Calendar.current
        
        let todayEnergy = activeEnergyData
            .filter { calendar.isDate($0.endDate, inSameDayAs: Date()) }
            .reduce(0) { $0 + $1.value }
        
        return "\(Int(todayEnergy)) kcal"
    }
    
    // Get total distance for today
    var todayDistance: String {
        let calendar = Calendar.current
        
        let todayDistance = distanceData
            .filter { calendar.isDate($0.endDate, inSameDayAs: Date()) }
            .reduce(0) { $0 + $1.value }
        
        // Convert to kilometers
        let kilometers = todayDistance / 1000
        return String(format: "%.2f km", kilometers)
    }
    
    // Data count summaries
    var dataCountSummary: String {
        return """
        Heart rate: \(heartRateData.count) records
        Steps: \(stepCountData.count) records
        Sleep: \(sleepData.count) records
        Workouts: \(workoutData.count) records
        """
    }
    
    // Risk level status
    var riskStatus: String {
        guard let result = analysisResult else { return "Not analyzed" }
        return result.riskLevel
    }
    
    // Risk score percentage
    var riskScorePercentage: String {
        guard let result = analysisResult else { return "N/A" }
        return String(format: "%.1f%%", result.riskScore * 100)
    }
    
    // Top contributing factors
    var topContributingFactors: [(name: String, value: Double)] {
        guard let result = analysisResult else { return [] }
        
        return result.contributingFactors
            .sorted(by: { $0.value > $1.value })
            .prefix(5)
            .map { (formatFeatureName($0.key), $0.value) }
    }
    
    // Format feature names for display
    private func formatFeatureName(_ name: String) -> String {
        // Convert snake_case to readable text
        return name
            .split(separator: "_")
            .map { $0.capitalized }
            .joined(separator: " ")
    }
} 