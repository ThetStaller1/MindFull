import Foundation
import HealthKit
import Combine

class HealthViewModel: ObservableObject {
    private let healthStore = HKHealthStore()
    private let apiBaseURL = "http://localhost:8000" // Change in production
    
    @Published var isAuthorized = false
    @Published var isLoading = false
    @Published var errorMessage: String? = nil
    @Published var lastSyncDate: Date? = nil
    @Published var analysisResult: AnalysisResult? = nil
    
    // Health data storage
    @Published var heartRateData: [HealthDataPoint] = []
    @Published var stepData: [HealthDataPoint] = []
    @Published var activeEnergyData: [HealthDataPoint] = []
    @Published var sleepData: [HealthDataPoint] = []
    @Published var workoutData: [WorkoutDataPoint] = []
    
    // Types of health data to collect
    private let typesToRead = [
        HKQuantityType.quantityType(forIdentifier: .heartRate)!,
        HKQuantityType.quantityType(forIdentifier: .stepCount)!,
        HKQuantityType.quantityType(forIdentifier: .activeEnergyBurned)!,
        HKQuantityType.quantityType(forIdentifier: .basalEnergyBurned)!,
        HKCategoryType.categoryType(forIdentifier: .sleepAnalysis)!
    ]
    
    private let workoutType = HKObjectType.workoutType()
    
    // Date formatter for API dates
    private let apiDateFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd'T'HH:mm:ss.SSS'Z'"
        formatter.timeZone = TimeZone(secondsFromGMT: 0)
        return formatter
    }()
    
    init() {
        loadLastSyncDate()
    }
    
    private func loadLastSyncDate() {
        if let dateString = UserDefaults.standard.string(forKey: "last_sync_date"),
           let date = apiDateFormatter.date(from: dateString) {
            self.lastSyncDate = date
        }
    }
    
    private func saveLastSyncDate(_ date: Date) {
        let dateString = apiDateFormatter.string(from: date)
        UserDefaults.standard.set(dateString, forKey: "last_sync_date")
        self.lastSyncDate = date
    }
    
    // Request authorization to access HealthKit data
    func requestAuthorization() {
        // Check if HealthKit is available on this device
        guard HKHealthStore.isHealthDataAvailable() else {
            self.errorMessage = "HealthKit is not available on this device"
            return
        }
        
        healthStore.requestAuthorization(toShare: nil, read: Set(typesToRead + [workoutType])) { success, error in
            DispatchQueue.main.async {
                if success {
                    self.isAuthorized = true
                } else if let error = error {
                    self.errorMessage = "Failed to authorize HealthKit: \(error.localizedDescription)"
                }
            }
        }
    }
    
    // Collect health data from HealthKit
    func collectHealthData(completion: @escaping (Bool) -> Void) {
        isLoading = true
        errorMessage = nil
        
        // Clear previous data
        heartRateData = []
        stepData = []
        activeEnergyData = []
        sleepData = []
        workoutData = []
        
        // Set the start date for data collection (use last sync date or default to 60 days ago)
        let startDate = lastSyncDate ?? Calendar.current.date(byAdding: .day, value: -60, to: Date())!
        let endDate = Date()
        
        let dispatchGroup = DispatchGroup()
        
        // Collect heart rate data
        dispatchGroup.enter()
        collectQuantityData(for: .heartRate, unit: HKUnit(from: "count/min"), startDate: startDate, endDate: endDate) { results in
            self.heartRateData = results
            dispatchGroup.leave()
        }
        
        // Collect step count data
        dispatchGroup.enter()
        collectQuantityData(for: .stepCount, unit: HKUnit.count(), startDate: startDate, endDate: endDate) { results in
            self.stepData = results
            dispatchGroup.leave()
        }
        
        // Collect active energy data
        dispatchGroup.enter()
        collectQuantityData(for: .activeEnergyBurned, unit: HKUnit.kilocalorie(), startDate: startDate, endDate: endDate) { results in
            self.activeEnergyData = results
            dispatchGroup.leave()
        }
        
        // Collect sleep data
        dispatchGroup.enter()
        collectSleepData(startDate: startDate, endDate: endDate) { results in
            self.sleepData = results
            dispatchGroup.leave()
        }
        
        // Collect workout data
        dispatchGroup.enter()
        collectWorkoutData(startDate: startDate, endDate: endDate) { results in
            self.workoutData = results
            dispatchGroup.leave()
        }
        
        dispatchGroup.notify(queue: .main) {
            self.isLoading = false
            self.saveLastSyncDate(endDate)
            
            // Check if we have data
            let hasData = !self.heartRateData.isEmpty || 
                          !self.stepData.isEmpty || 
                          !self.activeEnergyData.isEmpty || 
                          !self.sleepData.isEmpty ||
                          !self.workoutData.isEmpty
            
            completion(hasData)
        }
    }
    
    private func collectQuantityData(for typeIdentifier: HKQuantityTypeIdentifier, unit: HKUnit, startDate: Date, endDate: Date, completion: @escaping ([HealthDataPoint]) -> Void) {
        
        guard let quantityType = HKQuantityType.quantityType(forIdentifier: typeIdentifier) else {
            print("Failed to get quantity type for \(typeIdentifier)")
            completion([])
            return
        }
        
        let predicate = HKQuery.predicateForSamples(withStart: startDate, end: endDate, options: .strictStartDate)
        
        // Define the query to fetch health data
        let query = HKSampleQuery(sampleType: quantityType, predicate: predicate, limit: HKObjectQueryNoLimit, sortDescriptors: nil) { _, samples, error in
            
            guard let samples = samples as? [HKQuantitySample], error == nil else {
                print("Error fetching \(typeIdentifier) samples: \(String(describing: error))")
                DispatchQueue.main.async {
                    completion([])
                }
                return
            }
            
            // Process the samples
            let healthData = samples.map { sample in
                return HealthDataPoint(
                    type: typeIdentifier.rawValue,
                    startDate: self.apiDateFormatter.string(from: sample.startDate),
                    endDate: self.apiDateFormatter.string(from: sample.endDate),
                    value: "\(sample.quantity.doubleValue(for: unit))",
                    unit: unit.unitString
                )
            }
            
            DispatchQueue.main.async {
                completion(healthData)
            }
        }
        
        healthStore.execute(query)
    }
    
    private func collectSleepData(startDate: Date, endDate: Date, completion: @escaping ([HealthDataPoint]) -> Void) {
        guard let sleepType = HKObjectType.categoryType(forIdentifier: .sleepAnalysis) else {
            print("Failed to get sleep type")
            completion([])
            return
        }
        
        let predicate = HKQuery.predicateForSamples(withStart: startDate, end: endDate, options: .strictStartDate)
        
        let query = HKSampleQuery(sampleType: sleepType, predicate: predicate, limit: HKObjectQueryNoLimit, sortDescriptors: nil) { _, samples, error in
            
            guard let samples = samples as? [HKCategorySample], error == nil else {
                print("Error fetching sleep samples: \(String(describing: error))")
                DispatchQueue.main.async {
                    completion([])
                }
                return
            }
            
            // Process the samples - filter for sleep periods
            let sleepData = samples.compactMap { sample in
                // Only take inBed or asleep values
                if sample.value == HKCategoryValueSleepAnalysis.inBed.rawValue || 
                   sample.value == HKCategoryValueSleepAnalysis.asleep.rawValue {
                    
                    // Calculate duration in minutes
                    let duration = sample.endDate.timeIntervalSince(sample.startDate) / 60
                    
                    return HealthDataPoint(
                        type: HKCategoryTypeIdentifier.sleepAnalysis.rawValue,
                        startDate: self.apiDateFormatter.string(from: sample.startDate),
                        endDate: self.apiDateFormatter.string(from: sample.endDate),
                        value: "\(Int(duration))",
                        unit: "min"
                    )
                }
                return nil
            }
            
            DispatchQueue.main.async {
                completion(sleepData)
            }
        }
        
        healthStore.execute(query)
    }
    
    private func collectWorkoutData(startDate: Date, endDate: Date, completion: @escaping ([WorkoutDataPoint]) -> Void) {
        let predicate = HKQuery.predicateForSamples(withStart: startDate, end: endDate, options: .strictStartDate)
        
        let query = HKSampleQuery(sampleType: HKObjectType.workoutType(), predicate: predicate, limit: HKObjectQueryNoLimit, sortDescriptors: nil) { _, samples, error in
            
            guard let workouts = samples as? [HKWorkout], error == nil else {
                print("Error fetching workout samples: \(String(describing: error))")
                DispatchQueue.main.async {
                    completion([])
                }
                return
            }
            
            // Process the samples
            let workoutData = workouts.map { workout in
                return WorkoutDataPoint(
                    type: "HKWorkoutTypeIdentifier",
                    startDate: self.apiDateFormatter.string(from: workout.startDate),
                    endDate: self.apiDateFormatter.string(from: workout.endDate),
                    duration: "\(workout.duration)",
                    workoutActivityType: "\(workout.workoutActivityType.rawValue)",
                    totalEnergyBurned: workout.totalEnergyBurned?.doubleValue(for: HKUnit.kilocalorie()).description ?? "0",
                    totalDistance: workout.totalDistance?.doubleValue(for: HKUnit.meter()).description ?? "0"
                )
            }
            
            DispatchQueue.main.async {
                completion(workoutData)
            }
        }
        
        healthStore.execute(query)
    }
    
    // Upload health data to the backend
    func uploadHealthData(authToken: String?, completion: @escaping (Bool) -> Void) {
        guard let authToken = authToken else {
            errorMessage = "Not authenticated"
            completion(false)
            return
        }
        
        guard let url = URL(string: "\(apiBaseURL)/health-data/upload") else {
            errorMessage = "Invalid API URL"
            completion(false)
            return
        }
        
        // Create request
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.addValue("application/json", forHTTPHeaderField: "Content-Type")
        request.addValue("Bearer \(authToken)", forHTTPHeaderField: "Authorization")
        
        // Create payload
        let userInfo: [String: Any] = [
            "personId": UserDefaults.standard.string(forKey: "user_id") ?? "1001",
            "age": 33, // Default value - should get from profile
            "genderBinary": 1 // Default value - should get from profile
        ]
        
        let payload: [String: Any] = [
            "heartRate": heartRateData.map { $0.toDictionary() },
            "steps": stepData.map { $0.toDictionary() },
            "activeEnergy": activeEnergyData.map { $0.toDictionary() },
            "sleep": sleepData.map { $0.toDictionary() },
            "workout": workoutData.map { $0.toDictionary() },
            "distance": [],
            "basalEnergy": [],
            "flightsClimbed": [],
            "userInfo": userInfo
        ]
        
        do {
            request.httpBody = try JSONSerialization.data(withJSONObject: payload)
        } catch {
            errorMessage = "Error creating request: \(error.localizedDescription)"
            completion(false)
            return
        }
        
        // Make the request
        URLSession.shared.dataTask(with: request) { [weak self] data, response, error in
            DispatchQueue.main.async {
                if let error = error {
                    self?.errorMessage = "Network error: \(error.localizedDescription)"
                    completion(false)
                    return
                }
                
                guard let data = data else {
                    self?.errorMessage = "No data received"
                    completion(false)
                    return
                }
                
                do {
                    if let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
                       let success = json["success"] as? Bool, success {
                        completion(true)
                    } else {
                        self?.errorMessage = "Upload failed"
                        completion(false)
                    }
                } catch {
                    self?.errorMessage = "Error parsing response: \(error.localizedDescription)"
                    completion(false)
                }
            }
        }.resume()
    }
    
    // Request mental health analysis
    func requestAnalysis(authToken: String?, completion: @escaping (Bool) -> Void) {
        guard let authToken = authToken else {
            errorMessage = "Not authenticated"
            completion(false)
            return
        }
        
        guard let url = URL(string: "\(apiBaseURL)/analyze") else {
            errorMessage = "Invalid API URL"
            completion(false)
            return
        }
        
        // Create request
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.addValue("application/json", forHTTPHeaderField: "Content-Type")
        request.addValue("Bearer \(authToken)", forHTTPHeaderField: "Authorization")
        
        // Create empty payload - we'll use data from the backend
        let userInfo: [String: Any] = [
            "personId": UserDefaults.standard.string(forKey: "user_id") ?? "1001",
            "age": 33,
            "genderBinary": 1
        ]
        
        let payload: [String: Any] = [
            "heartRate": [],
            "steps": [],
            "activeEnergy": [],
            "sleep": [],
            "workout": [],
            "distance": [],
            "basalEnergy": [],
            "flightsClimbed": [],
            "userInfo": userInfo
        ]
        
        do {
            request.httpBody = try JSONSerialization.data(withJSONObject: payload)
        } catch {
            errorMessage = "Error creating request: \(error.localizedDescription)"
            completion(false)
            return
        }
        
        // Make the request
        URLSession.shared.dataTask(with: request) { [weak self] data, response, error in
            DispatchQueue.main.async {
                if let error = error {
                    self?.errorMessage = "Network error: \(error.localizedDescription)"
                    completion(false)
                    return
                }
                
                guard let data = data else {
                    self?.errorMessage = "No data received"
                    completion(false)
                    return
                }
                
                do {
                    if let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] {
                        if let errorDetail = json["detail"] as? String {
                            self?.errorMessage = errorDetail
                            completion(false)
                            return
                        }
                        
                        let analysisResult = AnalysisResult(
                            userId: json["userId"] as? String ?? "",
                            prediction: json["prediction"] as? Int ?? 0,
                            riskLevel: json["riskLevel"] as? String ?? "UNKNOWN",
                            riskScore: json["riskScore"] as? Double ?? 0.0,
                            contributingFactors: json["contributingFactors"] as? [String: Double] ?? [:],
                            analysisDate: json["analysisDate"] as? String ?? ""
                        )
                        
                        self?.analysisResult = analysisResult
                        completion(true)
                    } else {
                        self?.errorMessage = "Could not parse response"
                        completion(false)
                    }
                } catch {
                    self?.errorMessage = "Error parsing response: \(error.localizedDescription)"
                    completion(false)
                }
            }
        }.resume()
    }
    
    // Get the latest analysis result
    func fetchLatestAnalysis(authToken: String?, completion: @escaping (Bool) -> Void) {
        guard let authToken = authToken else {
            errorMessage = "Not authenticated"
            completion(false)
            return
        }
        
        guard let url = URL(string: "\(apiBaseURL)/latest-analysis") else {
            errorMessage = "Invalid API URL"
            completion(false)
            return
        }
        
        // Create request
        var request = URLRequest(url: url)
        request.httpMethod = "GET"
        request.addValue("application/json", forHTTPHeaderField: "Content-Type")
        request.addValue("Bearer \(authToken)", forHTTPHeaderField: "Authorization")
        
        // Make the request
        URLSession.shared.dataTask(with: request) { [weak self] data, response, error in
            DispatchQueue.main.async {
                if let error = error {
                    self?.errorMessage = "Network error: \(error.localizedDescription)"
                    completion(false)
                    return
                }
                
                guard let data = data else {
                    self?.errorMessage = "No data received"
                    completion(false)
                    return
                }
                
                do {
                    if let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
                       let hasAnalysis = json["hasAnalysis"] as? Bool, hasAnalysis,
                       let analysis = json["analysis"] as? [String: Any] {
                        
                        let analysisResult = AnalysisResult(
                            userId: analysis["person_id"] as? String ?? "",
                            prediction: analysis["prediction"] as? Int ?? 0,
                            riskLevel: analysis["risk_level"] as? String ?? "UNKNOWN",
                            riskScore: analysis["risk_score"] as? Double ?? 0.0,
                            contributingFactors: analysis["contributing_factors"] as? [String: Double] ?? [:],
                            analysisDate: analysis["analysis_date"] as? String ?? ""
                        )
                        
                        self?.analysisResult = analysisResult
                        completion(true)
                    } else {
                        self?.analysisResult = nil
                        completion(false)
                    }
                } catch {
                    self?.errorMessage = "Error parsing response: \(error.localizedDescription)"
                    completion(false)
                }
            }
        }.resume()
    }
}

// Data models
struct HealthDataPoint: Identifiable, Codable {
    var id: String { type + startDate }
    let type: String
    let startDate: String
    let endDate: String
    let value: String
    let unit: String?
    
    func toDictionary() -> [String: Any] {
        var dict: [String: Any] = [
            "type": type,
            "startDate": startDate,
            "endDate": endDate,
            "value": value
        ]
        
        if let unit = unit {
            dict["unit"] = unit
        }
        
        return dict
    }
}

struct WorkoutDataPoint: Identifiable, Codable {
    var id: String { type + startDate }
    let type: String
    let startDate: String
    let endDate: String
    let duration: String
    let workoutActivityType: String
    let totalEnergyBurned: String
    let totalDistance: String
    
    func toDictionary() -> [String: Any] {
        return [
            "type": type,
            "startDate": startDate,
            "endDate": endDate,
            "duration": duration,
            "workoutActivityType": workoutActivityType,
            "totalEnergyBurned": totalEnergyBurned,
            "totalDistance": totalDistance,
            "value": duration
        ]
    }
}

struct AnalysisResult: Identifiable, Codable {
    var id: String { userId + analysisDate }
    let userId: String
    let prediction: Int
    let riskLevel: String
    let riskScore: Double
    let contributingFactors: [String: Double]
    let analysisDate: String
} 