import Foundation
import Combine

enum APIError: Error {
    case invalidURL
    case requestFailed(Error)
    case invalidResponse
    case decodingFailed(Error)
    case serverError(String)
    
    var description: String {
        switch self {
        case .invalidURL:
            return "Invalid URL"
        case .requestFailed(let error):
            return "Request failed: \(error.localizedDescription)"
        case .invalidResponse:
            return "Invalid response from server"
        case .decodingFailed(let error):
            return "Failed to decode response: \(error.localizedDescription)"
        case .serverError(let message):
            return "Server error: \(message)"
        }
    }
}

// API response models
struct AnalysisResult: Codable {
    let userId: String
    let prediction: Int
    let riskLevel: String
    let riskScore: Float
    let contributingFactors: [String: Double]
    let analysisDate: String
}

class APIService {
    #if DEBUG
    // Development server
    // For simulator use localhost, for physical device use Mac's IP address
    // Change this to your Mac's IP address when testing on a physical device
    private let macIPAddress = "100.65.36.67" // Updated to actual current WiFi IP address
    
    private var baseURL: String {
        // Check if we're running on a simulator or a real device
        #if targetEnvironment(simulator)
        return "http://localhost:8000"
        #else
        return "http://\(macIPAddress):8000"
        #endif
    }
    #else
    // Production server - replace with your actual server URL
    private let baseURL = "https://api.mindfull-app.com"
    #endif
    
    private var cancellables = Set<AnyCancellable>()
    
    func checkServerHealth() -> AnyPublisher<Bool, APIError> {
        guard let url = URL(string: "\(baseURL)/health") else {
            return Fail(error: APIError.invalidURL).eraseToAnyPublisher()
        }
        
        return URLSession.shared.dataTaskPublisher(for: url)
            .tryMap { data, response in
                guard let httpResponse = response as? HTTPURLResponse,
                      httpResponse.statusCode == 200 else {
                    throw APIError.invalidResponse
                }
                return true
            }
            .mapError { error in
                if let apiError = error as? APIError {
                    return apiError
                }
                return APIError.requestFailed(error)
            }
            .eraseToAnyPublisher()
    }
    
    func analyzeHealthData(healthData: [String: Any]) -> AnyPublisher<AnalysisResult, APIError> {
        guard let url = URL(string: "\(baseURL)/analyze") else {
            return Fail(error: APIError.invalidURL).eraseToAnyPublisher()
        }
        
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.addValue("application/json", forHTTPHeaderField: "Content-Type")
        
        do {
            request.httpBody = try JSONSerialization.data(withJSONObject: healthData, options: [])
        } catch {
            return Fail(error: APIError.requestFailed(error)).eraseToAnyPublisher()
        }
        
        return URLSession.shared.dataTaskPublisher(for: request)
            .tryMap { data, response in
                guard let httpResponse = response as? HTTPURLResponse else {
                    throw APIError.invalidResponse
                }
                
                if httpResponse.statusCode != 200 {
                    // Try to decode error message if available
                    if let errorResponse = try? JSONDecoder().decode([String: String].self, from: data),
                       let detail = errorResponse["detail"] {
                        throw APIError.serverError(detail)
                    } else {
                        throw APIError.serverError("Status code: \(httpResponse.statusCode)")
                    }
                }
                
                return data
            }
            .decode(type: AnalysisResult.self, decoder: JSONDecoder())
            .mapError { error in
                if let apiError = error as? APIError {
                    return apiError
                }
                if error is DecodingError {
                    return APIError.decodingFailed(error)
                }
                return APIError.requestFailed(error)
            }
            .eraseToAnyPublisher()
    }
} 