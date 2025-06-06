import Foundation

class APIService {
    static let shared = APIService()
    
    private let baseURL = URL(string: "http://192.168.1.241:8000")!
    private var authToken: String?
    
    private init() {}
    
    // MARK: - Authentication Methods
    
    func login(email: String, password: String) async throws -> Void {
        let endpoint = baseURL.appendingPathComponent("login")
        
        // Create request body
        let requestBody: [String: Any] = [
            "email": email,
            "password": password
        ]
        
        // Create request
        var request = URLRequest(url: endpoint)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONSerialization.data(withJSONObject: requestBody)
        
        // Send request
        let (data, response) = try await URLSession.shared.data(for: request)
        
        // Check response status
        guard let httpResponse = response as? HTTPURLResponse else {
            throw APIError.networkError
        }
        
        if httpResponse.statusCode != 200 {
            if let errorResponse = try? JSONDecoder().decode(ErrorResponse.self, from: data) {
                throw APIError.serverError(message: errorResponse.message)
            } else {
                throw APIError.serverError(message: "Login failed with status code: \(httpResponse.statusCode)")
            }
        }
        
        // Parse response - backend returns a TokenData object
        guard let responseDict = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let session = responseDict["session"] as? [String: Any],
              let accessToken = session["access_token"] as? String else {
            throw APIError.decodingError
        }
        
        // Store auth token
        self.authToken = accessToken
    }
    
    func register(email: String, password: String, name: String) async throws -> Void {
        let endpoint = baseURL.appendingPathComponent("register")
        
        // Create request body
        let requestBody: [String: Any] = [
            "email": email,
            "password": password,
            "name": name
        ]
        
        // Create request
        var request = URLRequest(url: endpoint)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONSerialization.data(withJSONObject: requestBody)
        
        // Send request
        let (data, response) = try await URLSession.shared.data(for: request)
        
        // Check response status
        guard let httpResponse = response as? HTTPURLResponse else {
            throw APIError.networkError
        }
        
        if httpResponse.statusCode != 200 && httpResponse.statusCode != 201 {
            if let errorResponse = try? JSONDecoder().decode(ErrorResponse.self, from: data) {
                throw APIError.serverError(message: errorResponse.message)
            } else {
                throw APIError.serverError(message: "Registration failed with status code: \(httpResponse.statusCode)")
            }
        }
        
        // Parse response - backend returns TokenData in the same format as login
        guard let responseDict = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let session = responseDict["session"] as? [String: Any],
              let accessToken = session["access_token"] as? String else {
            throw APIError.decodingError
        }
        
        // Store auth token
        self.authToken = accessToken
    }
    
    // MARK: - Health Data Methods
    
    func getLastSyncDate() async throws -> Date? {
        // Use "me" to get the current user's last sync date
        let endpoint = baseURL.appendingPathComponent("latest-data-timestamp/me")
        
        // Create request
        var request = URLRequest(url: endpoint)
        request.httpMethod = "GET"
        
        // Add auth token
        guard let authToken = authToken else {
            throw APIError.notAuthenticated
        }
        
        request.setValue("Bearer \(authToken)", forHTTPHeaderField: "Authorization")
        
        // Send request
        let (data, response) = try await URLSession.shared.data(for: request)
        
        // Check response status
        guard let httpResponse = response as? HTTPURLResponse else {
            throw APIError.networkError
        }
        
        if httpResponse.statusCode != 200 {
            if let errorResponse = try? JSONDecoder().decode(ErrorResponse.self, from: data) {
                throw APIError.serverError(message: errorResponse.message)
            } else {
                throw APIError.serverError(message: "Failed to get last sync date with status code: \(httpResponse.statusCode)")
            }
        }
        
        // Parse response
        guard let responseDict = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw APIError.decodingError
        }
        
        // Check if hasData is true and get latestTimestamp
        if let hasData = responseDict["hasData"] as? Bool, hasData,
           let latestTimestampString = responseDict["latestTimestamp"] as? String {
            let dateFormatter = DateFormatter()
            dateFormatter.dateFormat = "yyyy-MM-dd'T'HH:mm:ss.SSSZ"
            
            if let date = dateFormatter.date(from: latestTimestampString) {
                return date
            } else {
                dateFormatter.dateFormat = "yyyy-MM-dd'T'HH:mm:ss"
                return dateFormatter.date(from: latestTimestampString)
            }
        } else {
            // No sync date found, return nil
            return nil
        }
    }
    
    func sendHealthData(_ healthData: [HealthKitDataPoint]) async throws -> Void {
        let endpoint = baseURL.appendingPathComponent("analyze")
        
        // Create a properly structured request body
        var request = HealthKitDataRequest()
        
        // Set user ID if available
        if let authToken = authToken,
           let userId = getUserIdFromToken(authToken) {
            request.userInfo["userId"] = userId
        } else {
            print("Warning: No user ID available for health data")
        }
        
        // Load and apply user profile data
        if let profileData = UserDefaults.standard.data(forKey: "user_profile"),
           let profile = try? JSONDecoder().decode(UserProfile.self, from: profileData) {
            request.updateWithUserProfile(profile)
            print("Applied user profile: age=\(profile.getAge()), gender=\(profile.gender)")
        } else {
            print("Warning: No user profile found, using default values")
        }
        
        // Group health data by type
        var processedCount = 0
        let totalCount = healthData.count
        
        for point in healthData {
            var dataPoint = point.toDictionary()
            
            // Format fields to match backend expectations
            let dateFormatter = DateFormatter()
            dateFormatter.dateFormat = "yyyy-MM-dd'T'HH:mm:ssZ"
            
            if let timestampStr = dataPoint["timestamp"] as? String {
                dataPoint["startDate"] = timestampStr
                dataPoint["endDate"] = timestampStr
                dataPoint.removeValue(forKey: "timestamp")
            }
            
            // Map internal type to HealthKit identifier format
            if let type = dataPoint["type"] as? String {
                switch type {
                case "heart_rate":
                    dataPoint["type"] = "HKQuantityTypeIdentifierHeartRate"
                    request.heartRate.append(dataPoint)
                case "step_count":
                    dataPoint["type"] = "HKQuantityTypeIdentifierStepCount"
                    request.steps.append(dataPoint)
                case "active_energy":
                    dataPoint["type"] = "HKQuantityTypeIdentifierActiveEnergyBurned"
                    request.activeEnergy.append(dataPoint)
                case "basal_energy":
                    dataPoint["type"] = "HKQuantityTypeIdentifierBasalEnergyBurned"
                    request.basalEnergy.append(dataPoint)
                case "flights_climbed":
                    dataPoint["type"] = "HKQuantityTypeIdentifierFlightsClimbed"
                    request.flightsClimbed.append(dataPoint)
                case "sleep_analysis":
                    dataPoint["type"] = "HKCategoryTypeIdentifierSleepAnalysis"
                    
                    // Save the original duration value
                    let duration = dataPoint["duration"] as? Double
                    
                    // Convert numeric sleep stage values to HealthKit string identifiers
                    if let value = dataPoint["value"] as? Int {
                        switch value {
                        case 0:
                            dataPoint["value"] = "HKCategoryValueSleepAnalysisInBed"
                        case 1:
                            dataPoint["value"] = "HKCategoryValueSleepAnalysisAsleepUnspecified"
                        case 2:
                            dataPoint["value"] = "HKCategoryValueSleepAnalysisAwake"
                        case 3:
                            dataPoint["value"] = "HKCategoryValueSleepAnalysisAsleepDeep"
                        case 4:
                            dataPoint["value"] = "HKCategoryValueSleepAnalysisAsleepREM"
                        case 5:
                            dataPoint["value"] = "HKCategoryValueSleepAnalysisAsleepCore"
                        default:
                            dataPoint["value"] = "HKCategoryValueSleepAnalysisAsleepUnspecified"
                        }
                    }
                    
                    // Re-apply the duration value after converting sleep stage
                    if let duration = duration {
                        dataPoint["duration"] = duration
                    }
                    
                    request.sleep.append(dataPoint)
                case "workout":
                    dataPoint["type"] = "HKWorkoutTypeIdentifier"
                    request.workout.append(dataPoint)
                default:
                    // Skip unknown types
                    print("Warning: Unknown health data type: \(type)")
                    continue
                }
            } else {
                // Skip entries without a type
                continue
            }
            
            processedCount += 1
            
            // For debugging
            if processedCount % 100 == 0 {
                print("Processed \(processedCount) of \(totalCount) health data points")
            }
        }
        
        // Verify we have data to send
        let totalDataPoints = request.heartRate.count + request.steps.count + request.activeEnergy.count + 
                             request.basalEnergy.count + request.sleep.count + request.workout.count + 
                             request.flightsClimbed.count
        
        if totalDataPoints == 0 {
            throw APIError.serverError(message: "No valid health data to send")
        }
        
        print("Sending health data: \(totalDataPoints) points total")
        print("- Heart rate: \(request.heartRate.count)")
        print("- Steps: \(request.steps.count)")
        print("- Active energy: \(request.activeEnergy.count)")
        print("- Basal energy: \(request.basalEnergy.count)")
        print("- Sleep: \(request.sleep.count)")
        print("- Workout: \(request.workout.count)")
        print("- Flights climbed: \(request.flightsClimbed.count)")
        
        // Create request
        var urlRequest = URLRequest(url: endpoint)
        urlRequest.httpMethod = "POST"
        urlRequest.setValue("application/json", forHTTPHeaderField: "Content-Type")
        
        // Add auth token
        guard let authToken = authToken else {
            throw APIError.notAuthenticated
        }
        
        urlRequest.setValue("Bearer \(authToken)", forHTTPHeaderField: "Authorization")
        
        // Try to convert request to JSON
        do {
            urlRequest.httpBody = try JSONSerialization.data(withJSONObject: request.toDictionary())
        } catch {
            print("Error serializing health data: \(error)")
            throw APIError.serverError(message: "Failed to encode health data: \(error.localizedDescription)")
        }
        
        // First attempt with a generous timeout for complete synchronous processing
        var urlConfig = URLSessionConfiguration.default
        urlConfig.timeoutIntervalForRequest = 300.0 // Increased timeout (5 minutes)
        let session = URLSession(configuration: urlConfig)
        
        do {
            // Try to upload and get a complete response within the timeout
            let (data, response) = try await session.data(for: urlRequest)
            
            // Check response status
            guard let httpResponse = response as? HTTPURLResponse else {
                throw APIError.networkError
            }
            
            if httpResponse.statusCode != 200 {
                if let errorResponse = try? JSONDecoder().decode(ErrorResponse.self, from: data) {
                    throw APIError.serverError(message: errorResponse.message)
                } else {
                    throw APIError.serverError(message: "Failed to send health data with status code: \(httpResponse.statusCode)")
                }
            }
            
            print("Health data upload successful")
            return
        } catch let urlError as URLError {
            if urlError.code == .timedOut {
                // The request timed out, but the server might still be processing the data
                // Switch to polling mode to check if data was processed successfully
                print("Upload request timed out, polling for data processing status...")
                
                // Extract user ID for checking latest sync timestamp
                guard let userId = getUserIdFromToken(authToken) else {
                    throw APIError.serverError(message: "Unable to determine user ID for polling")
                }
                
                // Poll for a max of 5 minutes (30 attempts, 10 seconds apart)
                let maxAttempts = 30
                for attempt in 1...maxAttempts {
                    print("Polling attempt \(attempt) of \(maxAttempts)...")
                    
                    do {
                        // Wait before checking
                        try await Task.sleep(nanoseconds: 10_000_000_000) // 10 seconds
                        
                        // Check if data has been processed by checking the latest sync date
                        let syncDate = try await getLastSyncDate()
                        
                        // If we have a sync date that's recent (within the last 2 minutes), the upload was successful
                        if let date = syncDate, date.timeIntervalSinceNow > -120 {
                            print("Data processing completed successfully during polling")
                            return
                        }
                    } catch {
                        print("Error during polling: \(error)")
                        // Continue polling despite errors
                    }
                }
                
                // If we get here, polling timed out
                throw APIError.serverError(message: "Server processing timeout. The data was sent but processing took too long.")
            } else {
                throw APIError.networkError
            }
        }
    }
    
    // Helper function to extract user ID from JWT token
    private func getUserIdFromToken(_ token: String) -> String? {
        let segments = token.components(separatedBy: ".")
        guard segments.count > 1 else { return nil }
        
        // Get the payload part (2nd segment)
        var base64 = segments[1]
            .replacingOccurrences(of: "-", with: "+")
            .replacingOccurrences(of: "_", with: "/")
        
        // Add padding if needed
        while base64.count % 4 != 0 {
            base64 += "="
        }
        
        // Decode base64
        guard let data = Data(base64Encoded: base64) else { return nil }
        
        // Parse JSON
        do {
            if let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
               let sub = json["sub"] as? String {
                return sub
            }
        } catch {
            print("Error parsing JWT token: \(error)")
        }
        
        return nil
    }
    
    // Helper method to load user profile from UserDefaults
    private func loadUserProfile() -> UserProfile {
        if let data = UserDefaults.standard.data(forKey: "user_profile") {
            let decoder = JSONDecoder()
            if let profile = try? decoder.decode(UserProfile.self, from: data) {
                return profile
            }
        }
        return UserProfile() // Return default profile if not found
    }
    
    // MARK: - Analysis Methods
    
    func requestAnalysis() async throws -> Void {
        let endpoint = baseURL.appendingPathComponent("check-analysis")
        
        // Get user ID from auth token
        guard let authToken = authToken else {
            throw APIError.notAuthenticated
        }
        
        let userId = getUserIdFromToken(authToken) ?? ""
        if userId.isEmpty {
            throw APIError.serverError(message: "Could not determine user ID")
        }
        
        // Create request body with forceRun parameter and userId
        let requestBody: [String: Any] = [
            "forceRun": true,
            "userId": userId
        ]
        
        print("DEBUG - check-analysis request: \(requestBody)")
        
        // Create request
        var request = URLRequest(url: endpoint)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        
        // Encode request body
        do {
        request.httpBody = try JSONSerialization.data(withJSONObject: requestBody)
        } catch {
            throw APIError.serverError(message: "Failed to encode request: \(error.localizedDescription)")
        }
        
        // Add auth token
        request.setValue("Bearer \(authToken)", forHTTPHeaderField: "Authorization")
        
        // Send request
        let (data, response) = try await URLSession.shared.data(for: request)
        
        // Check response status
        guard let httpResponse = response as? HTTPURLResponse else {
            throw APIError.networkError
        }
        
        if httpResponse.statusCode != 200 {
            if let errorResponse = try? JSONDecoder().decode(ErrorResponse.self, from: data) {
                throw APIError.serverError(message: errorResponse.message)
            } else if let errorData = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                      let detail = errorData["detail"] as? String {
                throw APIError.serverError(message: "Check analysis failed: \(detail)")
            } else {
                throw APIError.serverError(message: "Failed to request analysis with status code: \(httpResponse.statusCode)")
            }
        }
    }
    
    func getLatestAnalysis() async throws -> AnalysisResult? {
        // Get latest analysis for the current user
        let endpoint = baseURL.appendingPathComponent("latest-analysis/me")
        
        // Create request
        var request = URLRequest(url: endpoint)
        request.httpMethod = "GET"
        
        // Add auth token
        guard let authToken = authToken else {
            throw APIError.notAuthenticated
        }
        
        request.setValue("Bearer \(authToken)", forHTTPHeaderField: "Authorization")
        
        // Send request
        let (data, response) = try await URLSession.shared.data(for: request)
        
        // Check response status
        guard let httpResponse = response as? HTTPURLResponse else {
            throw APIError.networkError
        }
        
        if httpResponse.statusCode != 200 {
            if let errorResponse = try? JSONDecoder().decode(ErrorResponse.self, from: data) {
                throw APIError.serverError(message: errorResponse.message)
            } else {
                throw APIError.serverError(message: "Failed to get analysis with status code: \(httpResponse.statusCode)")
            }
        }
        
        // Parse response
        if httpResponse.statusCode == 200 {
            let decoder = JSONDecoder()
            
            do {
                let result = try decoder.decode(AnalysisResult.self, from: data)
                return result
            } catch {
                print("Decoding error: \(error)")
                throw APIError.decodingError
            }
        }
        
        return nil
    }
    
    func getAnalysisHistory() async throws -> [AnalysisResult] {
        // Get analysis history for the current user
        let endpoint = baseURL.appendingPathComponent("analysis-history/me")
        
        // Create request
        var request = URLRequest(url: endpoint)
        request.httpMethod = "GET"
        
        // Add auth token
        guard let authToken = authToken else {
            throw APIError.notAuthenticated
        }
        
        request.setValue("Bearer \(authToken)", forHTTPHeaderField: "Authorization")
        
        // Send request
        let (data, response) = try await URLSession.shared.data(for: request)
        
        // Check response status
        guard let httpResponse = response as? HTTPURLResponse else {
            throw APIError.networkError
        }
        
        if httpResponse.statusCode != 200 {
            if let errorResponse = try? JSONDecoder().decode(ErrorResponse.self, from: data) {
                throw APIError.serverError(message: errorResponse.message)
            } else {
                throw APIError.serverError(message: "Failed to get analysis history with status code: \(httpResponse.statusCode)")
            }
        }
        
        // Parse response
        if httpResponse.statusCode == 200 {
            let decoder = JSONDecoder()
            
            do {
                let results = try decoder.decode([AnalysisResult].self, from: data)
                return results
            } catch {
                print("Decoding error: \(error)")
                throw APIError.decodingError
            }
        }
        
        return []
    }
}

// MARK: - API Error

enum APIError: Error, CustomStringConvertible {
    case networkError
    case serverError(message: String)
    case decodingError
    case notAuthenticated
    
    var description: String {
        switch self {
        case .networkError:
            return "Network connection error"
        case .serverError(let message):
            return "Server error: \(message)"
        case .decodingError:
            return "Error decoding response"
        case .notAuthenticated:
            return "User not authenticated"
        }
    }
    
    // Add integer raw values for easier error identification
    var code: Int {
        switch self {
        case .networkError: return 1
        case .serverError: return 2
        case .decodingError: return 3
        case .notAuthenticated: return 4
        }
    }
}

// MARK: - Error Response

struct ErrorResponse: Codable {
    let message: String
    
    // Handle different error message formats
    enum CodingKeys: String, CodingKey {
        case message
        case detail
    }
    
    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        
        // Try to decode the message from 'message' field first
        if let message = try? container.decode(String.self, forKey: .message) {
            self.message = message
        }
        // If not found, try 'detail' field (used by FastAPI)
        else if let detail = try? container.decode(String.self, forKey: .detail) {
            self.message = detail
        }
        // If neither field exists, provide a generic message
        else {
            self.message = "Unknown error occurred"
        }
    }
    
    // Add encode method to conform to Encodable protocol
    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(message, forKey: .message)
    }
}

// MARK: - Token Management

extension APIService {
    func setAuthToken(_ token: String) {
        self.authToken = token
    }
    
    func clearAuthToken() {
        self.authToken = nil
    }
} 