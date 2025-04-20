import Foundation
import HealthKit
import SwiftData
import Combine

// Define the HealthData class type properly
// Swift should be able to find this class within the project
// But sometimes it needs some help with the type system
//typealias HealthKitData = HealthData

// Use our central models export instead
// This approach is more explicit and helps Swift's type resolution
import Foundation

class HealthKitManager: ObservableObject {
    private let healthStore = HKHealthStore()
    
    @Published var isAuthorized = false
    @Published var errorMessage: String?
    @Published var dataFetchProgress: [String: Double] = [:]
    
    // Types we want to read from HealthKit
    let typesToRead: Set<HKObjectType> = [
        HKObjectType.quantityType(forIdentifier: .heartRate)!,
        HKObjectType.quantityType(forIdentifier: .activeEnergyBurned)!,
        HKObjectType.quantityType(forIdentifier: .stepCount)!,
        HKObjectType.quantityType(forIdentifier: .distanceWalkingRunning)!,
        HKObjectType.categoryType(forIdentifier: .sleepAnalysis)!,
        HKObjectType.quantityType(forIdentifier: .basalEnergyBurned)!,
        HKObjectType.quantityType(forIdentifier: .flightsClimbed)!,
        HKObjectType.workoutType()
    ]
    
    // Define the health data types as properties
    private lazy var heartRateType = HKQuantityType.quantityType(forIdentifier: .heartRate)!
    private lazy var stepCountType = HKQuantityType.quantityType(forIdentifier: .stepCount)!
    private lazy var sleepType = HKCategoryType.categoryType(forIdentifier: .sleepAnalysis)!
    private lazy var respiratoryRateType = HKQuantityType.quantityType(forIdentifier: .respiratoryRate)!
    private lazy var hrVariabilityType = HKQuantityType.quantityType(forIdentifier: .heartRateVariabilitySDNN)!
    private lazy var activeEnergyType = HKQuantityType.quantityType(forIdentifier: .activeEnergyBurned)!
    
    init() {
        // Check if HealthKit is available on this device
        guard HKHealthStore.isHealthDataAvailable() else {
            errorMessage = "HealthKit is not available on this device"
            return
        }
    }
    
    func requestAuthorization(completion: @escaping (Bool, Error?) -> Void) {
        healthStore.requestAuthorization(toShare: nil, read: typesToRead) { success, error in
            DispatchQueue.main.async {
                self.isAuthorized = success
                if let error = error {
                    self.errorMessage = error.localizedDescription
                }
                completion(success, error)
            }
        }
    }
    
    func fetchLatestData(for type: HKQuantityType, unit: HKUnit, completion: @escaping ([HealthData]) -> Void) {
        // Get dates for the query - REDUCED to 30 days from 60
        let now = Date()
        let startDate = Calendar.current.date(byAdding: .day, value: -30, to: now)!
        
        // Create the predicate
        let predicate = HKQuery.predicateForSamples(withStart: startDate, end: now, options: .strictStartDate)
        
        // Create the sort descriptor
        let sortDescriptor = NSSortDescriptor(key: HKSampleSortIdentifierEndDate, ascending: false)
        
        // Update progress indicator
        let typeKey = type.identifier
        DispatchQueue.main.async {
            self.dataFetchProgress[typeKey] = 0.0
        }
        
        // Implement pagination to handle large data sets
        self.fetchPaginatedData(
            for: type,
            unit: unit,
            predicate: predicate,
            sortDescriptors: [sortDescriptor],
            typeKey: typeKey,
            completion: completion
        )
    }
    
    private func fetchPaginatedData(
        for type: HKQuantityType,
        unit: HKUnit,
        predicate: NSPredicate,
        sortDescriptors: [NSSortDescriptor],
        typeKey: String,
        pageSize: Int = 1000,  // Increased from 500 to 1000 for better throughput
        currentPage: Int = 0,
        allResults: [HealthData] = [],
        completion: @escaping ([HealthData]) -> Void
    ) {
        // Define the result handler type
        typealias QueryResultsHandler = (HKSampleQuery, [HKSample]?, Error?) -> Void
        
        // Create a wrapper function that returns the closure
        func createResultsHandler() -> QueryResultsHandler {
            let handler: QueryResultsHandler = { [weak self] (query, samples, error) in
                guard let self = self else { return }
                
                DispatchQueue.main.async {
                    if let error = error {
                        self.errorMessage = error.localizedDescription
                        completion(allResults)
                        return
                    }
                    
                    guard let samples = samples, !samples.isEmpty else {
                        // No more samples - finish
                        self.dataFetchProgress[typeKey] = 1.0
                        completion(allResults)
                        return
                    }
                    
                    // Process samples in batches to avoid memory pressure
                    let healthData = self.processQuantitySamples(samples: samples, unit: unit, type: type)
                    
                    // Update progress for UI
                    let hasMoreData = samples.count == pageSize
                    let progress = min(Double(currentPage + 1) * 0.1, 0.9)
                    self.dataFetchProgress[typeKey] = progress
                    
                    // Combine current results with new data
                    let updatedResults = allResults + healthData
                    
                    if hasMoreData {
                        // Instead of recursion, we'll create a new query with a different anchor point
                        let nextAnchorDate = samples.last!.endDate
                        let nextPredicate = HKQuery.predicateForSamples(
                            withStart: nextAnchorDate,
                            end: predicate.predicateFormat.contains("endDate") ? 
                                 (predicate as? NSComparisonPredicate)?.rightExpression.constantValue as? Date : nil,
                            options: .strictStartDate
                        )
                        
                        // Create the next query with a new handler
                        let nextQuery = HKSampleQuery(
                            sampleType: type,
                            predicate: nextPredicate,
                            limit: pageSize,
                            sortDescriptors: sortDescriptors,
                            resultsHandler: createResultsHandler()
                        )
                        
                        // Execute the next page query after a short delay to reduce CPU load
                        DispatchQueue.global(qos: .userInitiated).asyncAfter(deadline: .now() + 0.1) {
                            self.healthStore.execute(nextQuery)
                        }
                        
                        // Pass back the updated results for proper accumulation
                        DispatchQueue.main.async {
                            completion(updatedResults)
                        }
                    } else {
                        // No more data, return final results
                        self.dataFetchProgress[typeKey] = 1.0
                        completion(updatedResults)
                    }
                }
            }
            return handler
        }
        
        // Create the query with the handler
        let query = HKSampleQuery(
            sampleType: type,
            predicate: predicate,
            limit: pageSize,
            sortDescriptors: sortDescriptors,
            resultsHandler: createResultsHandler()
        )
        
        // Execute the query
        healthStore.execute(query)
    }
    
    // Helper method to process samples in a memory-efficient way
    private func processQuantitySamples(samples: [HKSample], unit: HKUnit, type: HKQuantityType) -> [HealthData] {
        var results: [HealthData] = []
        results.reserveCapacity(samples.count)
        
        // Process samples in smaller batches
        let batchSize = 50
        for i in stride(from: 0, to: samples.count, by: batchSize) {
            let endIndex = min(i + batchSize, samples.count)
            let batch = samples[i..<endIndex]
            
            autoreleasepool {
                for sample in batch {
                    if let quantitySample = sample as? HKQuantitySample {
                        let value = self.getValue(from: quantitySample, for: type)
                        let healthData = HealthData(
                            type: type.identifier,
                            startDate: quantitySample.startDate,
                            endDate: quantitySample.endDate,
                            value: value,
                            unit: self.getUnit(for: type)
                        )
                        results.append(healthData)
                    }
                }
            }
        }
        
        return results
    }
    
    func fetchHeartRateData(completion: @escaping ([HealthData]) -> Void) {
        guard let heartRateType = HKQuantityType.quantityType(forIdentifier: .heartRate) else {
            completion([])
            return
        }
        
        fetchLatestData(for: heartRateType, unit: HKUnit(from: "count/min"), completion: completion)
    }
    
    func fetchStepCountData(completion: @escaping ([HealthData]) -> Void) {
        guard let stepCountType = HKQuantityType.quantityType(forIdentifier: .stepCount) else {
            completion([])
            return
        }
        
        fetchLatestData(for: stepCountType, unit: HKUnit.count(), completion: completion)
    }
    
    func fetchActiveEnergyData(completion: @escaping ([HealthData]) -> Void) {
        guard let energyType = HKQuantityType.quantityType(forIdentifier: .activeEnergyBurned) else {
            completion([])
            return
        }
        
        fetchLatestData(for: energyType, unit: HKUnit.kilocalorie(), completion: completion)
    }
    
    func fetchDistanceData(completion: @escaping ([HealthData]) -> Void) {
        guard let distanceType = HKQuantityType.quantityType(forIdentifier: .distanceWalkingRunning) else {
            completion([])
            return
        }
        
        fetchLatestData(for: distanceType, unit: HKUnit.meter(), completion: completion)
    }
    
    func fetchBasalEnergyData(completion: @escaping ([HealthData]) -> Void) {
        guard let energyType = HKQuantityType.quantityType(forIdentifier: .basalEnergyBurned) else {
            completion([])
            return
        }
        
        fetchLatestData(for: energyType, unit: HKUnit.kilocalorie(), completion: completion)
    }
    
    func fetchFlightsClimbedData(completion: @escaping ([HealthData]) -> Void) {
        guard let flightsType = HKQuantityType.quantityType(forIdentifier: .flightsClimbed) else {
            completion([])
            return
        }
        
        fetchLatestData(for: flightsType, unit: HKUnit.count(), completion: completion)
    }
    
    func fetchSleepData(completion: @escaping ([HealthData]) -> Void) {
        // Get sleep category type
        guard let sleepType = HKObjectType.categoryType(forIdentifier: .sleepAnalysis) else {
            errorMessage = "Sleep type is not available"
            completion([])
            return
        }
        
        // Set date range for last 30 days (reduced from 60)
        let now = Date()
        let calendar = Calendar.current
        let startDate = calendar.date(byAdding: .day, value: -30, to: now)!
        
        // Create predicate for the date range
        let predicate = HKQuery.predicateForSamples(withStart: startDate, end: now, options: .strictStartDate)
        
        // Set sort descriptor for date
        let sortDescriptor = NSSortDescriptor(key: HKSampleSortIdentifierStartDate, ascending: true)
        
        // Track sleep data fetch progress
        dataFetchProgress["sleep"] = 0.1
        
        // Use a more memory-efficient approach to collect sleep data with a custom reduction handler
        // that prevents all data from being stored in memory at once
        let resultsHandler = { [weak self] (newData: [HealthData]) in
            guard let self = self else { return }
            
            DispatchQueue.main.async {
                // Pass each batch directly to the completion handler instead of accumulating
                completion(newData)
                
                // When all sleep data is fetched (progress = 1.0), send an empty array with a marker
                if self.dataFetchProgress["sleep"] == 1.0 {
                    let completionMarker = HealthData(
                        type: "sleep_complete_marker",
                        startDate: Date(),
                        endDate: Date(),
                        value: 0,
                        unit: "marker"
                    )
                    completion([completionMarker])
                }
            }
        }
        
        // Start the paginated fetch with an empty initial results array
        fetchPaginatedSleepData(
            sleepType: sleepType,
            predicate: predicate,
            sortDescriptors: [sortDescriptor],
            typeKey: "sleep",
            allResults: [],
            completion: resultsHandler
        )
    }
    
    func fetchPaginatedSleepData(
        sleepType: HKSampleType,
        predicate: NSPredicate,
        sortDescriptors: [NSSortDescriptor],
        typeKey: String,
        pageSize: Int = 200, // Reduced from 1000 to 200 for better memory management
        allResults: [HealthData] = [],
        completion: @escaping ([HealthData]) -> Void
    ) {
        // Create a results handler function instead of a recursive closure
        func resultsHandler(query: HKSampleQuery, samples: [HKSample]?, error: Error?) {
            guard let samples = samples as? [HKCategorySample], error == nil else {
                DispatchQueue.main.async {
                    if let error = error {
                        self.errorMessage = error.localizedDescription
                    }
                    self.dataFetchProgress[typeKey] = 1.0
                    completion([])
                }
                return
            }
            
            // Process the samples in smaller batches to avoid memory pressure
            var healthData: [HealthData] = []
            healthData.reserveCapacity(min(samples.count, 200)) // Reserve reasonable capacity
            
            autoreleasepool {
                // Use smaller batch size for processing
                let batchSize = 20
                for i in stride(from: 0, to: samples.count, by: batchSize) {
                    let endIndex = min(i + batchSize, samples.count)
                    let batch = samples[i..<endIndex]
                    
                    // Process this small batch
                    for sample in batch {
                        let startDate = sample.startDate
                        let endDate = sample.endDate
                        let duration = endDate.timeIntervalSince(startDate)
                        
                        // Only include samples with meaningful duration
                        if duration > 0 {
                            let sleepState = sample.value
                            
                            // Map the sleep state to a descriptive string
                            var sleepStateName = "Unknown"
                            
                            if #available(iOS 16.0, *) {
                                switch sleepState {
                                case HKCategoryValueSleepAnalysis.inBed.rawValue:
                                    sleepStateName = "In Bed"
                                case HKCategoryValueSleepAnalysis.asleepUnspecified.rawValue:
                                    sleepStateName = "Asleep (Unspecified)"
                                case HKCategoryValueSleepAnalysis.asleepCore.rawValue:
                                    sleepStateName = "Core Sleep"
                                case HKCategoryValueSleepAnalysis.asleepDeep.rawValue:
                                    sleepStateName = "Deep Sleep"
                                case HKCategoryValueSleepAnalysis.asleepREM.rawValue:
                                    sleepStateName = "REM Sleep"
                                case HKCategoryValueSleepAnalysis.awake.rawValue:
                                    sleepStateName = "Awake"
                                default:
                                    sleepStateName = "Unknown (\(sleepState))"
                                }
                            } else {
                                // For iOS 15 and below
                                switch sleepState {
                                case HKCategoryValueSleepAnalysis.inBed.rawValue:
                                    sleepStateName = "In Bed"
                                case HKCategoryValueSleepAnalysis.asleep.rawValue:
                                    sleepStateName = "Asleep"
                                case HKCategoryValueSleepAnalysis.awake.rawValue:
                                    sleepStateName = "Awake"
                                default:
                                    sleepStateName = "Unknown (\(sleepState))"
                                }
                            }
                            
                            let data = HealthData(
                                type: "sleep",
                                startDate: startDate,
                                endDate: endDate,
                                value: Double(sleepState),
                                unit: "state",
                                metadata: ["sleepState": sleepStateName, "duration": "\(Int(duration/60)) minutes"]
                            )
                            healthData.append(data)
                        }
                    }
                }
            }
            
            // Update progress for UI
            let hasMoreData = samples.count == pageSize
            let progress = hasMoreData ? min(0.9, self.dataFetchProgress[typeKey] ?? 0 + 0.1) : 1.0
            DispatchQueue.main.async {
                self.dataFetchProgress[typeKey] = progress
            }
            
            // Send this batch immediately to avoid accumulating in memory
            DispatchQueue.main.async {
                completion(healthData)
            }
            
            // Check if we need to fetch more data
            if hasMoreData, let lastSample = samples.last {
                // Create predicate for next page of results (after the last result of this page)
                let nextPredicate = HKQuery.predicateForSamples(
                    withStart: lastSample.endDate,
                    end: predicate.predicateFormat.contains("endDate") ? (predicate as? NSComparisonPredicate)?.rightExpression.constantValue as? Date : nil,
                    options: .strictStartDate
                )
                
                // Create and execute the next query
                let nextQuery = HKSampleQuery(
                    sampleType: sleepType,
                    predicate: nextPredicate,
                    limit: pageSize,
                    sortDescriptors: sortDescriptors,
                    resultsHandler: resultsHandler
                )
                
                // Execute the next page query with a small delay to reduce CPU load
                DispatchQueue.global(qos: .userInitiated).asyncAfter(deadline: .now() + 0.2) {
                    self.healthStore.execute(nextQuery)
                }
            } else {
                // No more data to fetch, mark progress as complete
                DispatchQueue.main.async {
                    self.dataFetchProgress[typeKey] = 1.0
                }
            }
        }
        
        // Start the initial query
        let query = HKSampleQuery(
            sampleType: sleepType,
            predicate: predicate,
            limit: pageSize,
            sortDescriptors: sortDescriptors,
            resultsHandler: resultsHandler
        )
        
        healthStore.execute(query)
    }
    
    func fetchWorkoutData(completion: @escaping ([HealthData]) -> Void) {
        let workoutType = HKObjectType.workoutType()
        
        // Get dates for the query - REDUCED to 30 days from 60
        let now = Date()
        let startDate = Calendar.current.date(byAdding: .day, value: -30, to: now)!
        
        // Create the predicate
        let predicate = HKQuery.predicateForSamples(withStart: startDate, end: now, options: .strictStartDate)
        
        // Create the sort descriptor
        let sortDescriptor = NSSortDescriptor(key: HKSampleSortIdentifierEndDate, ascending: false)
        
        // Update progress indicator
        let typeKey = workoutType.identifier
        DispatchQueue.main.async {
            self.dataFetchProgress[typeKey] = 0.0
        }
        
        // Implement pagination for workout data
        self.fetchPaginatedWorkoutData(
            workoutType: workoutType,
            predicate: predicate,
            sortDescriptors: [sortDescriptor],
            typeKey: typeKey,
            completion: completion
        )
    }
    
    private func fetchPaginatedWorkoutData(
        workoutType: HKWorkoutType,
        predicate: NSPredicate,
        sortDescriptors: [NSSortDescriptor],
        typeKey: String,
        pageSize: Int = 500,  // Increased for better throughput
        currentPage: Int = 0,
        allResults: [HealthData] = [],
        completion: @escaping ([HealthData]) -> Void
    ) {
        // Define the result handler type
        typealias QueryResultsHandler = (HKSampleQuery, [HKSample]?, Error?) -> Void
        
        // Create a wrapper function that returns the closure
        func createResultsHandler() -> QueryResultsHandler {
            let handler: QueryResultsHandler = { [weak self] (query, samples, error) in
                guard let self = self else { return }
                
                DispatchQueue.main.async {
                    if let error = error {
                        self.errorMessage = error.localizedDescription
                        completion(allResults)
                        return
                    }
                    
                    guard let workouts = samples as? [HKWorkout], !workouts.isEmpty else {
                        // No more samples - finish
                        self.dataFetchProgress[typeKey] = 1.0
                        completion(allResults)
                        return
                    }
                    
                    // Process workout samples and avoid memory pressure
                    var healthData: [HealthData] = []
                    healthData.reserveCapacity(workouts.count)
                    
                    autoreleasepool {
                        // Process in smaller batches
                        let batchSize = 50
                        for i in stride(from: 0, to: workouts.count, by: batchSize) {
                            let endIndex = min(i + batchSize, workouts.count)
                            let batch = workouts[i..<endIndex]
                            
                            for workout in batch {
                                // Duration in minutes
                                let durationMinutes = workout.duration / 60
                                
                                // Only add workouts with meaningful duration (at least 1 minute)
                                if durationMinutes >= 1.0 {
                                    let data = HealthData(
                                        type: "HKWorkoutTypeIdentifier",
                                        startDate: workout.startDate,
                                        endDate: workout.endDate,
                                        value: durationMinutes,
                                        unit: "min"
                                    )
                                    healthData.append(data)
                                }
                            }
                        }
                    }
                    
                    // Update progress for UI
                    let hasMoreData = workouts.count == pageSize
                    let progress = min(Double(currentPage + 1) * 0.2, 0.9)
                    self.dataFetchProgress[typeKey] = progress
                    
                    if hasMoreData {
                        // Instead of recursion, create a new query with a different anchor point
                        let nextAnchorDate = workouts.last!.endDate
                        let nextPredicate = HKQuery.predicateForSamples(
                            withStart: nextAnchorDate,
                            end: predicate.predicateFormat.contains("endDate") ? 
                                 (predicate as? NSComparisonPredicate)?.rightExpression.constantValue as? Date : nil,
                            options: .strictStartDate
                        )
                        
                        // Create the next query with a new handler
                        let nextQuery = HKSampleQuery(
                            sampleType: workoutType,
                            predicate: nextPredicate,
                            limit: pageSize,
                            sortDescriptors: sortDescriptors,
                            resultsHandler: createResultsHandler()
                        )
                        
                        // Return this batch immediately to avoid memory buildup
                        completion(healthData)
                        
                        // Execute the next page query with a small delay to reduce CPU load
                        DispatchQueue.global(qos: .userInitiated).asyncAfter(deadline: .now() + 0.1) {
                            self.healthStore.execute(nextQuery)
                        }
                    } else {
                        // No more data, return final results
                        self.dataFetchProgress[typeKey] = 1.0
                        completion(healthData)
                    }
                }
            }
            return handler
        }
        
        // Create the query with the handler
        let query = HKSampleQuery(
            sampleType: workoutType,
            predicate: predicate,
            limit: pageSize,
            sortDescriptors: sortDescriptors,
            resultsHandler: createResultsHandler()
        )
        
        // Execute the query
        healthStore.execute(query)
    }
    
    // Get the overall progress of data fetching (0.0 to 1.0)
    var overallDataFetchProgress: Double {
        guard !dataFetchProgress.isEmpty else { return 0.0 }
        
        let totalProgress = dataFetchProgress.values.reduce(0.0, +)
        return totalProgress / Double(dataFetchProgress.count)
    }
    
    // Reset progress tracking
    func resetProgress() {
        dataFetchProgress.removeAll()
    }
    
    // MARK: - Processing methods
    
    func fetchPaginatedHealthData(
        _ quantityType: HKQuantityType,
        predicate: NSPredicate,
        sortDescriptors: [NSSortDescriptor],
        typeKey: String,
        pageSize: Int = 1000,
        allResults: [HealthData] = [],
        completion: @escaping ([HealthData]) -> Void
    ) {
        // Use a function instead of a recursive closure
        func resultsHandler(query: HKSampleQuery, samples: [HKSample]?, error: Error?) {
            guard let samples = samples as? [HKQuantitySample], error == nil else {
                DispatchQueue.main.async {
                    if let error = error {
                        self.errorMessage = error.localizedDescription
                    }
                    completion([])
                }
                return
            }
            
            // Process the sample data
            var healthData: [HealthData] = []
            healthData.reserveCapacity(samples.count)
            
            autoreleasepool {
                for sample in samples {
                    let startDate = sample.startDate
                    let endDate = sample.endDate
                    let value = self.getValue(from: sample, for: quantityType)
                    let unit = self.getUnit(for: quantityType)
                    
                    let data = HealthData(
                        type: typeKey,
                        startDate: startDate,
                        endDate: endDate,
                        value: value,
                        unit: unit
                    )
                    healthData.append(data)
                }
            }
            
            // Update progress
            let hasMoreData = samples.count == pageSize
            let progress = min(Double(allResults.count + healthData.count) / Double(pageSize * 5), 0.9)
            DispatchQueue.main.async {
                self.dataFetchProgress[typeKey] = progress
            }
            
            // Combine current results with new data
            let updatedResults = allResults + healthData
            
            // If there's more data, query the next page
            if hasMoreData, let lastSample = samples.last {
                // Create predicate for the next page
                let nextPredicate = HKQuery.predicateForSamples(
                    withStart: lastSample.endDate,
                    end: predicate.predicateFormat.contains("endDate") ? (predicate as? NSComparisonPredicate)?.rightExpression.constantValue as? Date : nil,
                    options: .strictStartDate
                )
                
                // Create and execute the next query
                let nextQuery = HKSampleQuery(
                    sampleType: quantityType,
                    predicate: nextPredicate,
                    limit: pageSize,
                    sortDescriptors: sortDescriptors,
                    resultsHandler: resultsHandler
                )
                
                // Execute the next page query with a small delay to reduce CPU load
                DispatchQueue.global(qos: .userInitiated).asyncAfter(deadline: .now() + 0.1) {
                    self.healthStore.execute(nextQuery)
                }
                
                // Return the accumulated results so far
                DispatchQueue.main.async {
                    completion(updatedResults)
                }
            } else {
                // No more data to fetch, mark progress as complete
                DispatchQueue.main.async {
                    self.dataFetchProgress[typeKey] = 1.0
                    // Return all results
                    completion(updatedResults)
                }
            }
        }
        
        // Start the initial query
        let query = HKSampleQuery(
            sampleType: quantityType,
            predicate: predicate,
            limit: pageSize,
            sortDescriptors: sortDescriptors,
            resultsHandler: resultsHandler
        )
        
        healthStore.execute(query)
    }
    
    // Modify the helper method to get value from quantity sample
    func getValue(from sample: HKQuantitySample, for quantityType: HKQuantityType) -> Double {
        let unit = self.getHKUnit(for: quantityType)
        return sample.quantity.doubleValue(for: unit)
    }
    
    // Rename to getHKUnit to return HKUnit object
    func getHKUnit(for quantityType: HKQuantityType) -> HKUnit {
        switch quantityType.identifier {
        case HKQuantityTypeIdentifier.heartRate.rawValue:
            return HKUnit(from: "count/min")
        case HKQuantityTypeIdentifier.stepCount.rawValue:
            return HKUnit.count()
        case HKQuantityTypeIdentifier.activeEnergyBurned.rawValue, 
             HKQuantityTypeIdentifier.basalEnergyBurned.rawValue:
            return HKUnit.kilocalorie()
        case HKQuantityTypeIdentifier.distanceWalkingRunning.rawValue:
            return HKUnit.meter()
        case HKQuantityTypeIdentifier.flightsClimbed.rawValue:
            return HKUnit.count()
        case HKQuantityTypeIdentifier.respiratoryRate.rawValue:
            return HKUnit(from: "count/min")
        case HKQuantityTypeIdentifier.heartRateVariabilitySDNN.rawValue:
            return HKUnit.secondUnit(with: .milli)
        default:
            return HKUnit.count()
        }
    }
    
    // Add a method that returns string unit names
    func getUnit(for quantityType: HKQuantityType) -> String {
        switch quantityType.identifier {
        case HKQuantityTypeIdentifier.heartRate.rawValue:
            return "count/min"
        case HKQuantityTypeIdentifier.stepCount.rawValue:
            return "count"
        case HKQuantityTypeIdentifier.activeEnergyBurned.rawValue, 
             HKQuantityTypeIdentifier.basalEnergyBurned.rawValue:
            return "kcal"
        case HKQuantityTypeIdentifier.distanceWalkingRunning.rawValue:
            return "m"
        case HKQuantityTypeIdentifier.flightsClimbed.rawValue:
            return "count"
        case HKQuantityTypeIdentifier.respiratoryRate.rawValue:
            return "count/min"
        case HKQuantityTypeIdentifier.heartRateVariabilitySDNN.rawValue:
            return "ms"
        default:
            return "count"
        }
    }
} 