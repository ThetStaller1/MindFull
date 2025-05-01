import Foundation
import HealthKit
import Combine

// Enum for sleep stages
enum SleepStage: String, Codable {
    case inBed
    case asleep
    case awake
    case rem
    case core
    case deep
    
    static func fromHKCategoryValue(_ value: Int) -> SleepStage {
        switch value {
        case HKCategoryValueSleepAnalysis.inBed.rawValue:
            return .inBed
        case HKCategoryValueSleepAnalysis.asleepUnspecified.rawValue:
            return .asleep
        case HKCategoryValueSleepAnalysis.awake.rawValue:
            return .awake
        case HKCategoryValueSleepAnalysis.asleepREM.rawValue:
            return .rem
        case HKCategoryValueSleepAnalysis.asleepCore.rawValue:
            return .core
        case HKCategoryValueSleepAnalysis.asleepDeep.rawValue:
            return .deep
        default:
            return .asleep
        }
    }
}

// Extension for array calculations
extension Array where Element == Double {
    var average: Double {
        guard !isEmpty else { return 0 }
        return self.reduce(0, +) / Double(self.count)
    }
    
    var standardDeviation: Double {
        guard count > 1 else { return 0 }
        let mean = self.average
        let variance = self.reduce(0) { $0 + pow($1 - mean, 2) } / Double(self.count - 1)
        return sqrt(variance)
    }
}

class HealthViewModel: ObservableObject {
    private let healthStore = HKHealthStore()
    
    // MARK: - Network Configuration
    // Hardcoded for demo purposes
    private let serverBaseURL = "http://100.65.56.136:8000"
    
    private var cancellables = Set<AnyCancellable>()
    
    @Published var isAuthorized = false
    @Published var isLoading = false
    @Published var errorMessage: String?
    
    // Health data
    @Published var sleepData: [SleepEntry] = []
    @Published var heartRateData: [HeartRateEntry] = []
    @Published var activityData: [ActivityEntry] = []
    @Published var lastUpdated: Date?
    
    // Analysis data
    @Published var analysisResult: AnalysisResult?
    @Published var isAnalysisLoading = false
    @Published var analysisError: String?
    
    // Check if HealthKit is available on this device
    let healthKitAvailable = HKHealthStore.isHealthDataAvailable()
    
    // Types of data we want to read from HealthKit
    let typesToRead: Set<HKObjectType> = [
        // Sleep data
        HKObjectType.categoryType(forIdentifier: .sleepAnalysis)!,
        
        // Heart data
        HKObjectType.quantityType(forIdentifier: .heartRate)!,
        HKObjectType.quantityType(forIdentifier: .heartRateVariabilitySDNN)!,
        HKObjectType.quantityType(forIdentifier: .restingHeartRate)!,
        
        // Activity data
        HKObjectType.quantityType(forIdentifier: .stepCount)!,
        HKObjectType.quantityType(forIdentifier: .distanceWalkingRunning)!,
        HKObjectType.quantityType(forIdentifier: .activeEnergyBurned)!,
        HKObjectType.quantityType(forIdentifier: .basalEnergyBurned)!,
        HKObjectType.quantityType(forIdentifier: .flightsClimbed)!,
        HKObjectType.quantityType(forIdentifier: .appleStandTime)!,
        HKObjectType.quantityType(forIdentifier: .appleExerciseTime)!,
        
        // Walking metrics
        HKObjectType.quantityType(forIdentifier: .walkingSpeed)!,
        HKObjectType.quantityType(forIdentifier: .walkingDoubleSupportPercentage)!,
        HKObjectType.quantityType(forIdentifier: .walkingAsymmetryPercentage)!,
        HKObjectType.quantityType(forIdentifier: .walkingStepLength)!,
        
        // Workouts
        HKObjectType.workoutType()
    ]
    
    init() {
        // Load the last sync time
        if let lastSync = UserDefaults.standard.object(forKey: "lastHealthDataSync") as? Date {
            self.lastUpdated = lastSync
        }
        
        // Check if we're already authorized
        checkAuthorizationStatus()
    }
    
    private func checkAuthorizationStatus() {
        guard healthKitAvailable else { 
            print("HealthKit not available on this device")
            return 
        }
        
        // Check authorization status for each type
        var allAuthorized = true
        for type in typesToRead {
            let status = healthStore.authorizationStatus(for: type)
            print("Authorization status for \(type): \(status.rawValue)")
            if status != .sharingAuthorized {
                allAuthorized = false
                break
            }
        }
        
        print("All HealthKit types authorized: \(allAuthorized)")
        
        DispatchQueue.main.async {
            self.isAuthorized = allAuthorized
        }
    }
    
    func requestAuthorization() {
        guard healthKitAvailable else {
            self.errorMessage = "HealthKit is not available on this device"
            print("HealthKit not available on this device")
            return
        }
        
        print("Requesting HealthKit authorization...")
        
        healthStore.requestAuthorization(toShare: nil, read: typesToRead) { [weak self] success, error in
            DispatchQueue.main.async {
                if success {
                    print("HealthKit authorization granted!")
                    self?.isAuthorized = true
                    self?.fetchHealthData()
                } else if let error = error {
                    print("HealthKit authorization failed: \(error.localizedDescription)")
                    self?.errorMessage = "Authorization failed: \(error.localizedDescription)"
                } else {
                    print("HealthKit authorization denied by user")
                    self?.errorMessage = "Authorization denied by user"
                }
            }
        }
    }
    
    func fetchHealthData() {
        guard isAuthorized else {
            errorMessage = "HealthKit access not authorized"
            return
        }
        
        isLoading = true
        errorMessage = nil
        
        let dispatchGroup = DispatchGroup()
        
        // Always fetch the last 60 days of data, regardless of last update time
        // This ensures we have comprehensive data for the model
        let calendar = Calendar.current
        let startDate = calendar.date(byAdding: .day, value: -60, to: Date())!
        let endDate = Date()
        
        print("Fetching health data from \(startDate) to \(endDate)")
        
        // Fetch sleep data
        dispatchGroup.enter()
        fetchSleepData(from: startDate, to: endDate) { [weak self] in
            dispatchGroup.leave()
        }
        
        // Fetch heart rate data
        dispatchGroup.enter()
        fetchHeartRateData(from: startDate, to: endDate) { [weak self] in
            dispatchGroup.leave()
        }
        
        // Fetch activity data
        dispatchGroup.enter()
        fetchActivityData(from: startDate, to: endDate) { [weak self] in
            dispatchGroup.leave()
        }
        
        // Fetch additional metrics in background for enriched analysis
        fetchAdditionalMetrics(from: startDate, to: endDate)
        
        dispatchGroup.notify(queue: .main) { [weak self] in
            guard let self = self else { return }
            
            // Upload data to backend
            self.uploadHealthData()
            
            // Update the last sync time
            self.lastUpdated = Date()
            UserDefaults.standard.set(self.lastUpdated, forKey: "lastHealthDataSync")
            
            self.isLoading = false
        }
    }
    
    private func fetchSleepData(from startDate: Date, to endDate: Date, completion: @escaping () -> Void) {
        let sleepType = HKObjectType.categoryType(forIdentifier: .sleepAnalysis)!
        
        let predicate = HKQuery.predicateForSamples(withStart: startDate, end: endDate, options: .strictStartDate)
        
        let query = HKSampleQuery(sampleType: sleepType, predicate: predicate, limit: HKObjectQueryNoLimit, sortDescriptors: nil) { [weak self] _, samples, error in
            
            if let error = error {
                DispatchQueue.main.async {
                    self?.errorMessage = "Failed to fetch sleep data: \(error.localizedDescription)"
                }
                completion()
                return
            }
            
            let sleepSamples = samples as? [HKCategorySample] ?? []
            
            // Process sleep samples
            var sleepEntries: [SleepEntry] = []
            
            // Track different sleep stages for extended analysis
            var inBedEntries: [SleepEntry] = []
            var asleepEntries: [SleepEntry] = []
            var awakeEntries: [SleepEntry] = []
            var remEntries: [SleepEntry] = []
            var coreEntries: [SleepEntry] = []
            var deepEntries: [SleepEntry] = []
            
            for sample in sleepSamples {
                // Duration in hours
                let duration = sample.endDate.timeIntervalSince(sample.startDate) / 3600
                
                // Skip very short entries (less than 1 minute)
                if duration < 0.016 { // 1 minute in hours
                    continue
                }
                
                let entry = SleepEntry(
                    startDate: sample.startDate,
                    endDate: sample.endDate,
                    duration: duration,
                    sleepStage: SleepStage.fromHKCategoryValue(sample.value),
                    isAsleep: sample.value == HKCategoryValueSleepAnalysis.asleepUnspecified.rawValue ||
                             sample.value == HKCategoryValueSleepAnalysis.asleepCore.rawValue ||
                             sample.value == HKCategoryValueSleepAnalysis.asleepDeep.rawValue ||
                             sample.value == HKCategoryValueSleepAnalysis.asleepREM.rawValue
                )
                
                sleepEntries.append(entry)
                
                // Categorize by sleep stage
                switch entry.sleepStage {
                    case .inBed:
                        inBedEntries.append(entry)
                    case .asleep:
                        asleepEntries.append(entry)
                    case .awake:
                        awakeEntries.append(entry)
                    case .rem:
                        remEntries.append(entry)
                    case .core:
                        coreEntries.append(entry)
                    case .deep:
                        deepEntries.append(entry)
                }
            }
            
            DispatchQueue.main.async {
                self?.sleepData = sleepEntries
                
                print("Sleep data collected:")
                print("- Total sleep entries: \(sleepEntries.count)")
                print("- In bed entries: \(inBedEntries.count)")
                print("- Asleep entries: \(asleepEntries.count)")
                print("- Awake entries: \(awakeEntries.count)")
                print("- REM entries: \(remEntries.count)")
                print("- Core entries: \(coreEntries.count)")
                print("- Deep entries: \(deepEntries.count)")
                
                // Store different sleep stages for feature extraction
                UserDefaults.standard.set(inBedEntries.count, forKey: "sleep_inbed_count")
                UserDefaults.standard.set(asleepEntries.count, forKey: "sleep_asleep_count")
                UserDefaults.standard.set(awakeEntries.count, forKey: "sleep_awake_count")
                UserDefaults.standard.set(remEntries.count, forKey: "sleep_rem_count")
                UserDefaults.standard.set(coreEntries.count, forKey: "sleep_core_count")
                UserDefaults.standard.set(deepEntries.count, forKey: "sleep_deep_count")
                
                // Calculate and store statistics by sleep stage (mean and std)
                // These are required features for the model
                
                // For asleep entries
                let asleepDurations = asleepEntries.map { $0.duration * 60 } // Convert to minutes
                UserDefaults.standard.set(asleepDurations.average, forKey: "sleep_minute_asleep_mean")
                UserDefaults.standard.set(asleepDurations.standardDeviation, forKey: "sleep_minute_asleep_std")
                UserDefaults.standard.set(asleepDurations.min(), forKey: "sleep_minute_asleep_min")
                UserDefaults.standard.set(asleepDurations.max(), forKey: "sleep_minute_asleep_max")
                
                // For awake entries
                let awakeDurations = awakeEntries.map { $0.duration * 60 }
                UserDefaults.standard.set(awakeDurations.average, forKey: "sleep_minute_awake_mean")
                UserDefaults.standard.set(awakeDurations.standardDeviation, forKey: "sleep_minute_awake_std")
                
                // For REM sleep entries
                let remDurations = remEntries.map { $0.duration * 60 }
                UserDefaults.standard.set(remDurations.average, forKey: "sleep_minute_rem_mean")
                UserDefaults.standard.set(remDurations.standardDeviation, forKey: "sleep_minute_rem_std")
                
                // For core (light) sleep entries
                let coreDurations = coreEntries.map { $0.duration * 60 }
                UserDefaults.standard.set(coreDurations.average, forKey: "sleep_minute_light_mean")
                UserDefaults.standard.set(coreDurations.standardDeviation, forKey: "sleep_minute_light_std")
                
                // For deep sleep entries
                let deepDurations = deepEntries.map { $0.duration * 60 }
                UserDefaults.standard.set(deepDurations.average, forKey: "sleep_minute_deep_mean")
                UserDefaults.standard.set(deepDurations.standardDeviation, forKey: "sleep_minute_deep_std")
                
                completion()
            }
        }
        
        healthStore.execute(query)
    }
    
    private func fetchHeartRateData(from startDate: Date, to endDate: Date, completion: @escaping () -> Void) {
        let heartRateType = HKObjectType.quantityType(forIdentifier: .heartRate)!
        
        let predicate = HKQuery.predicateForSamples(withStart: startDate, end: endDate, options: .strictStartDate)
        
        let query = HKSampleQuery(sampleType: heartRateType, predicate: predicate, limit: HKObjectQueryNoLimit, sortDescriptors: nil) { [weak self] _, samples, error in
            
            if let error = error {
                DispatchQueue.main.async {
                    self?.errorMessage = "Failed to fetch heart rate data: \(error.localizedDescription)"
                }
                completion()
                return
            }
            
            let heartRateSamples = samples as? [HKQuantitySample] ?? []
            
            // Process heart rate samples
            var heartRateEntries: [HeartRateEntry] = []
            
            for sample in heartRateSamples {
                // Heart rate is stored in count/minute
                let heartRate = sample.quantity.doubleValue(for: HKUnit.count().unitDivided(by: .minute()))
                
                let entry = HeartRateEntry(
                    timestamp: sample.startDate,
                    value: heartRate,
                    restingHeartRate: false // We don't know if this is resting HR
                )
                
                heartRateEntries.append(entry)
            }
            
            DispatchQueue.main.async {
                self?.heartRateData = heartRateEntries
                completion()
            }
        }
        
        healthStore.execute(query)
    }
    
    private func fetchActivityData(from startDate: Date, to endDate: Date, completion: @escaping () -> Void) {
        let dispatchGroup = DispatchGroup()
        
        var stepCountEntries: [StepCountEntry] = []
        var calorieEntries: [CalorieEntry] = []
        
        // Fetch step count
        dispatchGroup.enter()
        let stepCountType = HKObjectType.quantityType(forIdentifier: .stepCount)!
        let stepPredicate = HKQuery.predicateForSamples(withStart: startDate, end: endDate, options: .strictStartDate)
        
        let stepQuery = HKSampleQuery(sampleType: stepCountType, predicate: stepPredicate, limit: HKObjectQueryNoLimit, sortDescriptors: nil) { _, samples, error in
            
            if error == nil, let stepSamples = samples as? [HKQuantitySample] {
                for sample in stepSamples {
                    let steps = sample.quantity.doubleValue(for: HKUnit.count())
                    
                    let entry = StepCountEntry(
                        date: sample.startDate,
                        count: Int(steps)
                    )
                    
                    stepCountEntries.append(entry)
                }
            }
            
            dispatchGroup.leave()
        }
        
        healthStore.execute(stepQuery)
        
        // Fetch active energy burned
        dispatchGroup.enter()
        let energyType = HKObjectType.quantityType(forIdentifier: .activeEnergyBurned)!
        let energyPredicate = HKQuery.predicateForSamples(withStart: startDate, end: endDate, options: .strictStartDate)
        
        let energyQuery = HKSampleQuery(sampleType: energyType, predicate: energyPredicate, limit: HKObjectQueryNoLimit, sortDescriptors: nil) { _, samples, error in
            
            if error == nil, let energySamples = samples as? [HKQuantitySample] {
                for sample in energySamples {
                    let calories = sample.quantity.doubleValue(for: HKUnit.kilocalorie())
                    
                    let entry = CalorieEntry(
                        date: sample.startDate,
                        activeCalories: calories
                    )
                    
                    calorieEntries.append(entry)
                }
            }
            
            dispatchGroup.leave()
        }
        
        healthStore.execute(energyQuery)
        
        // Combine all activity data
        dispatchGroup.notify(queue: .main) { [weak self] in
            // Group all data by day
            let calendar = Calendar.current
            
            // Dictionary to store daily step counts
            var dailySteps: [Date: Int] = [:]
            for entry in stepCountEntries {
                let day = calendar.startOfDay(for: entry.date)
                dailySteps[day, default: 0] += entry.count
            }
            
            // Dictionary to store daily calories
            var dailyCalories: [Date: Double] = [:]
            for entry in calorieEntries {
                let day = calendar.startOfDay(for: entry.date)
                dailyCalories[day, default: 0] += entry.activeCalories
            }
            
            // Combine into activity entries
            var activityEntries: [ActivityEntry] = []
            
            // Get the unique set of days
            let allDays = Set(dailySteps.keys).union(dailyCalories.keys)
            
            for day in allDays {
                let entry = ActivityEntry(
                    date: day,
                    stepCount: dailySteps[day] ?? 0,
                    activeCalories: dailyCalories[day] ?? 0
                )
                
                activityEntries.append(entry)
            }
            
            // Sort by date
            activityEntries.sort { $0.date > $1.date }
            
            self?.activityData = activityEntries
            completion()
        }
    }
    
    private func fetchAdditionalMetrics(from startDate: Date, to endDate: Date) {
        // This method fetches additional metrics but doesn't block the main loading flow
        
        // Common predicate for all queries
        let predicate = HKQuery.predicateForSamples(withStart: startDate, end: endDate, options: .strictStartDate)
        
        // List of types to fetch
        let additionalTypes: [(HKQuantityType, HKUnit)] = [
            (HKQuantityType.quantityType(forIdentifier: .heartRateVariabilitySDNN)!, HKUnit.secondUnit(with: .milli)),
            (HKQuantityType.quantityType(forIdentifier: .walkingSpeed)!, HKUnit.meter().unitDivided(by: HKUnit.second())),
            (HKQuantityType.quantityType(forIdentifier: .walkingDoubleSupportPercentage)!, HKUnit.percent()),
            (HKQuantityType.quantityType(forIdentifier: .walkingAsymmetryPercentage)!, HKUnit.percent()),
            (HKQuantityType.quantityType(forIdentifier: .flightsClimbed)!, HKUnit.count()),
            (HKQuantityType.quantityType(forIdentifier: .appleStandTime)!, HKUnit.minute()),
            (HKQuantityType.quantityType(forIdentifier: .restingHeartRate)!, HKUnit.count().unitDivided(by: .minute()))
        ]
        
        // Fetch each type
        for (quantityType, unit) in additionalTypes {
            let query = HKSampleQuery(sampleType: quantityType, predicate: predicate, limit: HKObjectQueryNoLimit, sortDescriptors: nil) { _, samples, error in
                if let error = error {
                    print("Error fetching \(quantityType.identifier): \(error.localizedDescription)")
                    return
                }
                
                if let samples = samples as? [HKQuantitySample], !samples.isEmpty {
                    print("Successfully fetched \(samples.count) samples for \(quantityType.identifier)")
                    
                    // For logging/debugging only - get the average value
                    let values = samples.map { $0.quantity.doubleValue(for: unit) }
                    if !values.isEmpty {
                        let average = values.reduce(0, +) / Double(values.count)
                        print("Average \(quantityType.identifier): \(average) \(unit)")
                    }
                }
            }
            
            healthStore.execute(query)
        }
        
        // Fetch workouts
        let workoutQuery = HKSampleQuery(sampleType: HKObjectType.workoutType(), predicate: predicate, limit: HKObjectQueryNoLimit, sortDescriptors: nil) { _, samples, error in
            if let error = error {
                print("Error fetching workouts: \(error.localizedDescription)")
                return
            }
            
            if let workouts = samples as? [HKWorkout], !workouts.isEmpty {
                print("Successfully fetched \(workouts.count) workouts")
                
                for workout in workouts {
                    print("Workout: \(workout.workoutActivityType.rawValue), duration: \(workout.duration / 60) minutes")
                }
            }
        }
        
        healthStore.execute(workoutQuery)
    }
    
    private func uploadHealthData() {
        guard let token = UserDefaults.standard.string(forKey: "auth_token"), !token.isEmpty else {
            self.errorMessage = "Authentication token not found"
            return
        }
        
        // Set loading state for analysis
        isAnalysisLoading = true
        analysisError = nil
        
        print("Beginning data transformation for analysis...")
        
        // Convert health data to API format expected by backend
        let heartRateData = self.heartRateData.map { entry -> [String: Any] in
            return [
                "type": "HKQuantityTypeIdentifierHeartRate",
                "startDate": ISO8601DateFormatter().string(from: entry.timestamp),
                "endDate": ISO8601DateFormatter().string(from: entry.timestamp.addingTimeInterval(5)), // Add 5 seconds
                "value": entry.value,
                "unit": "count/min"
            ]
        }
        
        let stepsData = self.activityData.map { entry -> [String: Any] in
            return [
                "type": "HKQuantityTypeIdentifierStepCount",
                "startDate": ISO8601DateFormatter().string(from: entry.date),
                "endDate": ISO8601DateFormatter().string(from: entry.date.addingTimeInterval(3600)), // Add an hour
                "value": entry.stepCount,
                "unit": "count"
            ]
        }
        
        let activeEnergyData = self.activityData.map { entry -> [String: Any] in
            return [
                "type": "HKQuantityTypeIdentifierActiveEnergyBurned",
                "startDate": ISO8601DateFormatter().string(from: entry.date),
                "endDate": ISO8601DateFormatter().string(from: entry.date.addingTimeInterval(3600)), // Add an hour
                "value": entry.activeCalories,
                "unit": "kcal"
            ]
        }
        
        // Enhanced sleep data with sleep stages
        let sleepData = self.sleepData.map { entry -> [String: Any] in
            return [
                "type": "HKCategoryTypeIdentifierSleepAnalysis",
                "startDate": ISO8601DateFormatter().string(from: entry.startDate),
                "endDate": ISO8601DateFormatter().string(from: entry.endDate),
                "value": entry.duration * 60, // Convert hours to minutes
                "unit": "min",
                "isAsleep": entry.isAsleep,
                "sleepStage": entry.sleepStage.rawValue
            ]
        }
        
        // Calculate workout data from activity metrics
        let workoutData: [[String: Any]] = self.activityData.map { entry -> [String: Any] in
            // Use active calories to estimate workout duration (rough approximation)
            let estimatedDurationMinutes = max(entry.activeCalories / 10, 15) // At least 15 minutes if there's data
            return [
                "type": "HKWorkoutTypeIdentifier",
                "startDate": ISO8601DateFormatter().string(from: entry.date),
                "endDate": ISO8601DateFormatter().string(from: entry.date.addingTimeInterval(estimatedDurationMinutes * 60)),
                "value": estimatedDurationMinutes,
                "unit": "min"
            ]
        }
        
        // Calculate distance data from steps (rough approximation)
        let distanceData = self.activityData.map { entry -> [String: Any] in
            // Estimate distance from steps (rough approximation: 1300 steps per km on average)
            let estimatedDistance = Double(entry.stepCount) / 1300.0
            return [
                "type": "HKQuantityTypeIdentifierDistanceWalkingRunning",
                "startDate": ISO8601DateFormatter().string(from: entry.date),
                "endDate": ISO8601DateFormatter().string(from: entry.date.addingTimeInterval(3600)), // Add an hour
                "value": estimatedDistance,
                "unit": "km"
            ]
        }
        
        // Calculate basal energy data from activity metrics
        let basalEnergyData = self.activityData.map { entry -> [String: Any] in
            // Estimate basal energy (rough approximation: typically 60-70% of total energy)
            let estimatedBasalCalories = entry.activeCalories * 1.5 // Approximate basal as 1.5x active
            return [
                "type": "HKQuantityTypeIdentifierBasalEnergyBurned",
                "startDate": ISO8601DateFormatter().string(from: entry.date),
                "endDate": ISO8601DateFormatter().string(from: entry.date.addingTimeInterval(3600 * 24)), // Full day
                "value": estimatedBasalCalories,
                "unit": "kcal"
            ]
        }
        
        // Include all the additional sleep metrics we've calculated
        let sleepMeasurements: [String: Any] = [
            // Asleep metrics
            "sleep_minute_asleep_mean": UserDefaults.standard.double(forKey: "sleep_minute_asleep_mean"),
            "sleep_minute_asleep_std": UserDefaults.standard.double(forKey: "sleep_minute_asleep_std"),
            "sleep_minute_asleep_min": UserDefaults.standard.double(forKey: "sleep_minute_asleep_min"),
            "sleep_minute_asleep_max": UserDefaults.standard.double(forKey: "sleep_minute_asleep_max"),
            
            // Awake metrics
            "sleep_minute_awake_mean": UserDefaults.standard.double(forKey: "sleep_minute_awake_mean"),
            "sleep_minute_awake_std": UserDefaults.standard.double(forKey: "sleep_minute_awake_std"),
            
            // REM metrics
            "sleep_minute_rem_mean": UserDefaults.standard.double(forKey: "sleep_minute_rem_mean"),
            "sleep_minute_rem_std": UserDefaults.standard.double(forKey: "sleep_minute_rem_std"),
            
            // Light sleep (core) metrics
            "sleep_minute_light_mean": UserDefaults.standard.double(forKey: "sleep_minute_light_mean"),
            "sleep_minute_light_std": UserDefaults.standard.double(forKey: "sleep_minute_light_std"),
            
            // Deep sleep metrics
            "sleep_minute_deep_mean": UserDefaults.standard.double(forKey: "sleep_minute_deep_mean"),
            "sleep_minute_deep_std": UserDefaults.standard.double(forKey: "sleep_minute_deep_std"),
            
            // Sleep stage counts
            "sleep_inbed_count": UserDefaults.standard.integer(forKey: "sleep_inbed_count"),
            "sleep_asleep_count": UserDefaults.standard.integer(forKey: "sleep_asleep_count"),
            "sleep_awake_count": UserDefaults.standard.integer(forKey: "sleep_awake_count"),
            "sleep_rem_count": UserDefaults.standard.integer(forKey: "sleep_rem_count"),
            "sleep_core_count": UserDefaults.standard.integer(forKey: "sleep_core_count"),
            "sleep_deep_count": UserDefaults.standard.integer(forKey: "sleep_deep_count")
        ]
        
        // Calculate flights climbed from activity metrics
        let flightsClimbedData = self.activityData.map { entry -> [String: Any] in
            // Estimate flights climbed from steps (rough approximation: ~20 steps per flight)
            let estimatedFlights = max(entry.stepCount / 20, 0)
            return [
                "type": "HKQuantityTypeIdentifierFlightsClimbed",
                "startDate": ISO8601DateFormatter().string(from: entry.date),
                "endDate": ISO8601DateFormatter().string(from: entry.date.addingTimeInterval(3600)), // Add an hour
                "value": estimatedFlights,
                "unit": "count"
            ]
        }
        
        // Calculate heart rate variability from heart rate data
        let hrvData = self.heartRateData.map { entry -> [String: Any] in
            // Rough estimate of HRV based on heart rate (inverse relationship)
            // Higher heart rate often correlates with lower HRV
            let estimatedHRV = max(100 - entry.value * 0.5, 20) // Range roughly 20-100 ms
            return [
                "type": "HKQuantityTypeIdentifierHeartRateVariabilitySDNN",
                "startDate": ISO8601DateFormatter().string(from: entry.timestamp),
                "endDate": ISO8601DateFormatter().string(from: entry.timestamp.addingTimeInterval(60)),
                "value": estimatedHRV,
                "unit": "ms"
            ]
        }
        
        // Prepare the full data payload for direct analysis
        let dataPayload: [String: Any] = [
            "heartRate": heartRateData,
            "steps": stepsData,
            "activeEnergy": activeEnergyData,
            "sleep": sleepData,
            "workout": workoutData,
            "distance": distanceData,
            "basalEnergy": basalEnergyData,
            "flightsClimbed": flightsClimbedData,
            "heartRateVariability": hrvData,
            "sleepMeasurements": sleepMeasurements, // Add the sleep measurements data
            "userInfo": [
                "personId": UserDefaults.standard.string(forKey: "user_id") ?? "1001",
                "age": UserDefaults.standard.integer(forKey: "user_age") > 0 ? 
                    UserDefaults.standard.integer(forKey: "user_age") : 30, // Default age if not set
                "genderBinary": UserDefaults.standard.integer(forKey: "user_gender") > 0 ?
                    UserDefaults.standard.integer(forKey: "user_gender") : 1 // Default to female if not set
            ]
        ]
        
        print("Data transformation complete. Sending to server for analysis...")
        print("Data payload contains \(dataPayload.keys.count) data categories with feature count breakdown:")
        dataPayload.forEach { key, value in
            if let array = value as? [[String: Any]] {
                print("- \(key): \(array.count) entries")
            } else if let dict = value as? [String: Any] {
                print("- \(key): \(dict.count) key-value pairs")
            }
        }
        
        // Create URL request for direct analysis
        let url = URL(string: "\(serverBaseURL)/analyze")!
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.addValue("application/json", forHTTPHeaderField: "Content-Type")
        request.addValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        
        // Encode data
        do {
            request.httpBody = try JSONSerialization.data(withJSONObject: dataPayload)
        } catch {
            self.errorMessage = "Failed to encode health data: \(error.localizedDescription)"
            self.isAnalysisLoading = false
            print("JSON encoding error: \(error.localizedDescription)")
            return
        }
        
        // Send data to server for direct analysis
        URLSession.shared.dataTaskPublisher(for: request)
            .map(\.data)
            .decode(type: AnalysisResult.self, decoder: JSONDecoder())
            .receive(on: DispatchQueue.main)
            .sink(receiveCompletion: { [weak self] completion in
                self?.isAnalysisLoading = false
                if case .failure(let error) = completion {
                    self?.errorMessage = "Analysis failed: \(error.localizedDescription)"
                    self?.analysisError = "Analysis failed: \(error.localizedDescription)"
                    print("Analysis failed with error: \(error.localizedDescription)")
                    
                    // Simpler approach to logging errors
                    if let urlError = error as? URLError {
                        print("URL Error: \(urlError.code), \(urlError.localizedDescription)")
                    }
                } else {
                    print("Analysis completed successfully!")
                }
            }, receiveValue: { [weak self] result in
                print("Received analysis result: \(result)")
                self?.analysisResult = result
                
                // Store the result in UserDefaults as a backup
                if let resultData = try? JSONEncoder().encode(result) {
                    UserDefaults.standard.set(resultData, forKey: "latest_analysis_result")
                }
            })
            .store(in: &cancellables)
    }
    
    func requestAnalysis() {
        // Since we're directly analyzing data when we upload, this function
        // can simply trigger a new data fetch and upload
        fetchHealthData()
    }
    
    // Request health analysis from API
    func requestHealthAnalysis() {
        guard let token = UserDefaults.standard.string(forKey: "auth_token") else {
            analysisError = "Not authenticated"
            return
        }
        
        isAnalysisLoading = true
        analysisError = nil
        
        let url = URL(string: "\(serverBaseURL)/analyze")!
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.addValue("application/json", forHTTPHeaderField: "Content-Type")
        request.addValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
    }
}

// MARK: - Data Models

struct SleepEntry: Codable, Identifiable {
    var id: UUID = UUID()
    let startDate: Date
    let endDate: Date
    let duration: Double // in hours
    let sleepStage: SleepStage
    let isAsleep: Bool
    
    enum CodingKeys: String, CodingKey {
        case startDate, endDate, duration, sleepStage, isAsleep
    }
}

struct HeartRateEntry: Codable, Identifiable {
    var id: UUID = UUID()
    let timestamp: Date
    let value: Double // in BPM
    let restingHeartRate: Bool
    
    enum CodingKeys: String, CodingKey {
        case timestamp, value, restingHeartRate
    }
}

struct StepCountEntry: Codable {
    let date: Date
    let count: Int
}

struct CalorieEntry: Codable {
    let date: Date
    let activeCalories: Double
}

struct ActivityEntry: Codable, Identifiable {
    var id: UUID = UUID()
    let date: Date
    let stepCount: Int
    let activeCalories: Double
    
    enum CodingKeys: String, CodingKey {
        case date, stepCount, activeCalories
    }
}

struct HealthDataUpload: Codable {
    let sleepData: [SleepEntry]
    let heartRateData: [HeartRateEntry]
    let activityData: [ActivityEntry]
}

struct UploadResponse: Codable {
    let success: Bool
    let message: String?
}

struct AnalysisResult: Codable {
    let userId: String
    let prediction: Int
    let riskLevel: String
    let riskScore: Float
    let contributingFactors: [String: Float]
    let analysisDate: String
    
    // Add computed properties for compatibility with existing UI code
    var timestamp: Date {
        let formatter = ISO8601DateFormatter()
        return formatter.date(from: analysisDate) ?? Date()
    }
    
    var mentalHealthScore: Int {
        return prediction
    }
    
    var sleepQuality: Double {
        // Extract sleep quality from contributing factors or default to medium (0.5)
        return Double(contributingFactors.first(where: { $0.key.contains("sleep") })?.value ?? 0.5)
    }
    
    var stressLevel: Double {
        // Extract stress level from risk score (invert it as higher risk means higher stress)
        return Double(riskScore)
    }
    
    var moodState: String {
        // Determine mood based on risk level
        return riskLevel == "POSITIVE" ? "At Risk" : "Healthy"
    }
    
    var recommendations: [String] {
        // Generate recommendations based on contributing factors
        var recommendations: [String] = []
        
        // Add default recommendation
        recommendations.append("Continue monitoring your health data for personalized insights.")
        
        // Add specific recommendations based on risk level
        if riskLevel == "POSITIVE" {
            recommendations.append("Consider speaking with a healthcare professional about your mental health.")
        }
        
        // Add recommendations based on contributing factors
        let sortedFactors = contributingFactors.sorted(by: { $0.value > $1.value })
        if let topFactor = sortedFactors.first {
            if topFactor.key.contains("sleep") {
                recommendations.append("Improve your sleep schedule for better mental health.")
            } else if topFactor.key.contains("activity") {
                recommendations.append("Increase your physical activity to improve your mood.")
            } else if topFactor.key.contains("hr") {
                recommendations.append("Practice relaxation techniques to manage stress levels.")
            }
        }
        
        return recommendations
    }
} 