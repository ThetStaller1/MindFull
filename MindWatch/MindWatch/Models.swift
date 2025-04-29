import Foundation

// Health data models
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
    var id = UUID()
    let workoutType: String
    let startDate: String
    let endDate: String
    let duration: String
    let energyBurned: String?
    let distance: String?
    
    func toDictionary() -> [String: Any] {
        var dict: [String: Any] = [
            "workoutType": workoutType,
            "startDate": startDate,
            "endDate": endDate,
            "duration": duration
        ]
        
        if let energyBurned = energyBurned {
            dict["energyBurned"] = energyBurned
        }
        
        if let distance = distance {
            dict["distance"] = distance
        }
        
        return dict
    }
}

// Backend-compatible structure exactly matching backend's HealthKitData model
struct HealthKitDataRequest {
    var heartRate: [[String: Any]] = []
    var steps: [[String: Any]] = []
    var activeEnergy: [[String: Any]] = []
    var sleep: [[String: Any]] = []
    var workout: [[String: Any]] = []
    var distance: [[String: Any]] = []
    var basalEnergy: [[String: Any]] = []
    var flightsClimbed: [[String: Any]] = []
    var userInfo: [String: Any] = [
        "personId": "",
        "age": 33,
        "genderBinary": 1  // 1 = female, 0 = male per backend model
    ]
    
    // Convert to a dictionary that matches backend's HealthKitData model
    func toDictionary() -> [String: Any] {
        // Make sure entries have the exact format expected by the backend
        let processedHeartRate = heartRate.map { entry -> [String: Any] in
            var item = entry
            // Ensure all required fields exist
            if item["value"] is String {
                item["value"] = Double(item["value"] as! String) ?? 0
            }
            return item
        }
        
        let processedSteps = steps.map { entry -> [String: Any] in
            var item = entry
            // Ensure all required fields exist
            if item["value"] is String {
                item["value"] = Double(item["value"] as! String) ?? 0
            }
            return item
        }
        
        let processedActiveEnergy = activeEnergy.map { entry -> [String: Any] in
            var item = entry
            // Ensure all required fields exist
            if item["value"] is String {
                item["value"] = Double(item["value"] as! String) ?? 0
            }
            return item
        }
        
        // Return the dictionary structure exactly matching backend's HealthKitData model
        return [
            "heartRate": processedHeartRate,
            "steps": processedSteps,
            "activeEnergy": processedActiveEnergy,
            "sleep": sleep,
            "workout": workout,
            "distance": distance,
            "basalEnergy": basalEnergy,
            "flightsClimbed": flightsClimbed,
            "userInfo": userInfo
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
    
    // Additional fields from the backend that might be present
    var dataQuality: DataQuality?
    
    struct DataQuality: Codable {
        let qualityScore: Int
        let dataTypes: [String: Bool]
        let message: String
    }
    
    // Add CodingKeys to handle missing fields
    enum CodingKeys: String, CodingKey {
        case userId, prediction, riskLevel, riskScore, contributingFactors, analysisDate, dataQuality
    }
    
    // Custom init to make optional fields work
    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        
        userId = try container.decode(String.self, forKey: .userId)
        prediction = try container.decode(Int.self, forKey: .prediction)
        riskLevel = try container.decode(String.self, forKey: .riskLevel)
        riskScore = try container.decode(Double.self, forKey: .riskScore)
        contributingFactors = try container.decode([String: Double].self, forKey: .contributingFactors)
        analysisDate = try container.decode(String.self, forKey: .analysisDate)
        dataQuality = try? container.decode(DataQuality.self, forKey: .dataQuality)
    }
} 