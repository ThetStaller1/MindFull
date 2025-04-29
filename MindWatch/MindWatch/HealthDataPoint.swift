import Foundation

struct HealthKitDataPoint {
    enum DataType: String, Codable {
        case heartRate = "heart_rate"
        case stepCount = "step_count"
        case sleepAnalysis = "sleep_analysis"
        case activeEnergy = "active_energy"
        case basalEnergy = "basal_energy"
        case oxygenSaturation = "oxygen_saturation"
        case respiratoryRate = "respiratory_rate"
        case workout = "workout"
    }
    
    enum SleepStage: String, Codable {
        case inBed = "in_bed"
        case asleep = "asleep"
        case awake = "awake"
        case deep = "deep"
        case rem = "rem"
        case core = "core"
        case unknown = "unknown"
        
        static func fromHealthKitValue(_ value: String) -> SleepStage {
            switch value {
            case "0", "HKCategoryValueSleepAnalysisInBed":
                return .inBed
            case "1", "HKCategoryValueSleepAnalysisAsleep":
                return .asleep
            case "2", "HKCategoryValueSleepAnalysisAwake":
                return .awake
            case "3", "HKCategoryValueSleepAnalysisDeep":
                return .deep
            case "4", "HKCategoryValueSleepAnalysisREM":
                return .rem
            case "5", "HKCategoryValueSleepAnalysisCore":
                return .core
            default:
                return .unknown
            }
        }
    }
    
    let id: UUID
    let type: DataType
    let timestamp: Date
    let value: Double
    let unit: String
    let sleepStage: SleepStage?
    let source: String
    
    // Workout-specific fields
    let workoutActivityType: Int?
    let totalEnergyBurned: Double?
    let totalDistance: Double?
    
    init(id: UUID = UUID(), type: DataType, timestamp: Date, value: Double, unit: String, sleepStage: SleepStage? = nil, source: String, workoutActivityType: Int? = nil, totalEnergyBurned: Double? = nil, totalDistance: Double? = nil) {
        self.id = id
        self.type = type
        self.timestamp = timestamp
        self.value = value
        self.unit = unit
        self.sleepStage = sleepStage
        self.source = source
        self.workoutActivityType = workoutActivityType
        self.totalEnergyBurned = totalEnergyBurned
        self.totalDistance = totalDistance
    }
    
    func toDictionary() -> [String: Any] {
        // Format to match exact backend expectations
        let formattedTimestamp = formatDate(timestamp)
        
        var dict: [String: Any] = [
            "type": type.rawValue,  // Will be converted to HealthKit identifier format in APIService
            "timestamp": formattedTimestamp,
            "startDate": formattedTimestamp,  // Include both formats to ensure compatibility
            "endDate": formattedTimestamp,
            "value": value,
            "unit": unit,
            "source": source
        ]
        
        // Handle sleep data specifically
        if type == .sleepAnalysis, let sleepStage = sleepStage {
            dict["sleep_stage"] = sleepStage.rawValue
            
            // Add additional fields needed for sleep analysis
            // Convert sleep stage to value for backend
            switch sleepStage {
            case .inBed:
                dict["value"] = 0
            case .asleep:
                dict["value"] = 1
            case .awake:
                dict["value"] = 2
            case .deep:
                dict["value"] = 3
            case .rem:
                dict["value"] = 4
            case .core:
                dict["value"] = 5
            case .unknown:
                dict["value"] = 0
            }
        }
        
        // Add workout-specific fields if this is a workout
        if type == .workout {
            if let workoutActivityType = workoutActivityType {
                dict["workoutActivityType"] = workoutActivityType
            }
            
            if let totalEnergyBurned = totalEnergyBurned {
                dict["totalEnergyBurned"] = totalEnergyBurned
            }
            
            if let totalDistance = totalDistance {
                dict["totalDistance"] = totalDistance
            }
            
            // Add duration for the workout
            dict["duration"] = value
        }
        
        return dict
    }
    
    private func formatDate(_ date: Date) -> String {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd'T'HH:mm:ss'Z'"
        formatter.timeZone = TimeZone(abbreviation: "UTC")
        return formatter.string(from: date)
    }
} 