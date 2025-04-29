import Foundation
import CoreGraphics

struct MentalHealthAnalysisResult: Codable {
    let id: String
    let date: Date
    let overallScore: Int
    let categories: [CategoryScore]
    let recommendations: [String]
    
    struct CategoryScore: Codable {
        let name: String
        let score: Int
        let description: String
    }
}

extension MentalHealthAnalysisResult {
    var mentalHealthStatus: MentalHealthStatus {
        if overallScore >= 80 {
            return .excellent
        } else if overallScore >= 60 {
            return .good
        } else if overallScore >= 40 {
            return .moderate
        } else if overallScore >= 20 {
            return .poor
        } else {
            return .critical
        }
    }
}

enum MentalHealthStatus: String {
    case excellent = "Excellent"
    case good = "Good"
    case moderate = "Moderate"
    case poor = "Poor"
    case critical = "Critical"
    
    var color: CGColor {
        switch self {
        case .excellent:
            return CGColor(red: 0/255, green: 200/255, blue: 83/255, alpha: 1.0) // Green
        case .good:
            return CGColor(red: 76/255, green: 175/255, blue: 80/255, alpha: 1.0) // Light Green
        case .moderate:
            return CGColor(red: 255/255, green: 193/255, blue: 7/255, alpha: 1.0) // Yellow
        case .poor:
            return CGColor(red: 255/255, green: 87/255, blue: 34/255, alpha: 1.0) // Orange
        case .critical:
            return CGColor(red: 244/255, green: 67/255, blue: 54/255, alpha: 1.0) // Red
        }
    }
    
    var message: String {
        switch self {
        case .excellent:
            return "Your mental health appears to be excellent! Keep up the good work."
        case .good:
            return "Your mental health is in good shape. Continue your healthy habits."
        case .moderate:
            return "Your mental health shows some signs of stress. Consider implementing some self-care activities."
        case .poor:
            return "Your mental health needs attention. Consider speaking with a professional."
        case .critical:
            return "Your mental health requires immediate attention. Please reach out to a healthcare professional."
        }
    }
} 