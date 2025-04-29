import Foundation

extension WorkoutDataPoint {
    // Returns a dictionary formatted specifically to match what the backend expects
    func toBackendFormat() -> [String: Any] {
        var dict: [String: Any] = [
            "type": "HKWorkoutTypeIdentifier",
            "startDate": startDate,
            "endDate": endDate,
            "duration": duration,
            "workoutActivityType": Int(workoutType) ?? 37
        ]
        
        if let energyBurned = energyBurned {
            dict["totalEnergyBurned"] = energyBurned
        }
        
        if let distance = distance {
            dict["distance"] = distance
        }
        
        // Set value field for compatibility with backend model
        dict["value"] = duration
        
        return dict
    }
} 