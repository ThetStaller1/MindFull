import Foundation
import Combine

class AuthViewModel: ObservableObject {
    @Published var isAuthenticated = false
    @Published var isLoading = false
    @Published var errorMessage: String? = nil
    @Published var userId: String? = nil
    @Published var userEmail: String? = nil
    
    private let apiBaseURL = "http://192.168.1.241:8000" // Updated IP address
    private var authToken: String? = nil
    
    private let tokenKey = "auth_token"
    private let userIdKey = "user_id"
    private let userEmailKey = "user_email"
    
    init() {
        // Check if user is already authenticated
        loadSavedCredentials()
    }
    
    private func loadSavedCredentials() {
        if let token = UserDefaults.standard.string(forKey: tokenKey),
           let userId = UserDefaults.standard.string(forKey: userIdKey),
           let email = UserDefaults.standard.string(forKey: userEmailKey) {
            self.authToken = token
            self.userId = userId
            self.userEmail = email
            self.isAuthenticated = true
            
            // Set the token in APIService
            APIService.shared.setAuthToken(token)
        }
    }
    
    private func saveCredentials(token: String, userId: String, email: String) {
        UserDefaults.standard.set(token, forKey: tokenKey)
        UserDefaults.standard.set(userId, forKey: userIdKey)
        UserDefaults.standard.set(email, forKey: userEmailKey)
        
        self.authToken = token
        self.userId = userId
        self.userEmail = email
        self.isAuthenticated = true
        
        // Set the token in APIService
        APIService.shared.setAuthToken(token)
    }
    
    private func clearCredentials() {
        UserDefaults.standard.removeObject(forKey: tokenKey)
        UserDefaults.standard.removeObject(forKey: userIdKey)
        UserDefaults.standard.removeObject(forKey: userEmailKey)
        
        self.authToken = nil
        self.userId = nil
        self.userEmail = nil
        self.isAuthenticated = false
        
        // Clear the token in APIService
        APIService.shared.clearAuthToken()
    }
    
    func login(email: String, password: String) {
        isLoading = true
        errorMessage = nil
        
        guard let url = URL(string: "\(apiBaseURL)/login") else {
            errorMessage = "Invalid API URL"
            isLoading = false
            return
        }
        
        // Create login request
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.addValue("application/json", forHTTPHeaderField: "Content-Type")
        
        // Create request body
        let body: [String: Any] = [
            "email": email,
            "password": password
        ]
        
        do {
            request.httpBody = try JSONSerialization.data(withJSONObject: body)
        } catch {
            errorMessage = "Error creating request: \(error.localizedDescription)"
            isLoading = false
            return
        }
        
        // Make the request
        URLSession.shared.dataTask(with: request) { [weak self] data, response, error in
            DispatchQueue.main.async {
                self?.isLoading = false
                
                if let error = error {
                    self?.errorMessage = "Network error: \(error.localizedDescription)"
                    return
                }
                
                guard let data = data else {
                    self?.errorMessage = "No data received"
                    return
                }
                
                do {
                    if let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] {
                        if let errorMsg = json["detail"] as? String {
                            self?.errorMessage = errorMsg
                            return
                        }
                        
                        // Parse token and user data
                        if let token = json["access_token"] as? String,
                           let user = json["user"] as? [String: Any],
                           let userId = user["id"] as? String,
                           let userEmail = user["email"] as? String {
                            
                            self?.saveCredentials(token: token, userId: userId, email: userEmail)
                        } else {
                            self?.errorMessage = "Invalid response format"
                        }
                    } else {
                        self?.errorMessage = "Could not parse response"
                    }
                } catch {
                    self?.errorMessage = "Error parsing response: \(error.localizedDescription)"
                }
            }
        }.resume()
    }
    
    func register(email: String, password: String) {
        isLoading = true
        errorMessage = nil
        
        guard let url = URL(string: "\(apiBaseURL)/register") else {
            errorMessage = "Invalid API URL"
            isLoading = false
            return
        }
        
        // Create registration request
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.addValue("application/json", forHTTPHeaderField: "Content-Type")
        
        // Create request body
        let body: [String: Any] = [
            "email": email,
            "password": password
        ]
        
        do {
            request.httpBody = try JSONSerialization.data(withJSONObject: body)
        } catch {
            errorMessage = "Error creating request: \(error.localizedDescription)"
            isLoading = false
            return
        }
        
        // Make the request
        URLSession.shared.dataTask(with: request) { [weak self] data, response, error in
            DispatchQueue.main.async {
                self?.isLoading = false
                
                if let error = error {
                    self?.errorMessage = "Network error: \(error.localizedDescription)"
                    return
                }
                
                guard let data = data else {
                    self?.errorMessage = "No data received"
                    return
                }
                
                do {
                    if let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] {
                        if let errorMsg = json["detail"] as? String {
                            self?.errorMessage = errorMsg
                            return
                        }
                        
                        // Parse token and user data
                        if let token = json["access_token"] as? String,
                           let user = json["user"] as? [String: Any],
                           let userId = user["id"] as? String,
                           let userEmail = user["email"] as? String {
                            
                            self?.saveCredentials(token: token, userId: userId, email: userEmail)
                        } else {
                            self?.errorMessage = "Invalid response format"
                        }
                    } else {
                        self?.errorMessage = "Could not parse response"
                    }
                } catch {
                    self?.errorMessage = "Error parsing response: \(error.localizedDescription)"
                }
            }
        }.resume()
    }
    
    func logout() {
        clearCredentials()
    }
    
    func getAuthToken() -> String? {
        return authToken
    }
} 