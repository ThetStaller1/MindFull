import Foundation
import SwiftData

// This file serves as a central export point for all model types
// It helps ensure that model types are properly accessible
// throughout the project

// Re-export the HealthData model from this file
@Model
public final class HealthData {
    public var id: UUID
    public var type: String
    public var startDate: Date
    public var endDate: Date
    public var value: Double
    public var unit: String
    public var metadata: [String: String]?
    
    public init(id: UUID = UUID(), type: String, startDate: Date, endDate: Date, value: Double, unit: String, metadata: [String: String]? = nil) {
        self.id = id
        self.type = type
        self.startDate = startDate
        self.endDate = endDate
        self.value = value
        self.unit = unit
        self.metadata = metadata
    }
}

// You can add other models here if needed 