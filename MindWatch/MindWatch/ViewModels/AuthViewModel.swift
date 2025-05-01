import Foundation
import Combine

class AuthViewModel: ObservableObject {
    @Published var isAuthenticated = false
    @Published var isLoading = false
    @Published var errorMessage: String?
    
    // MARK: - Network Configuration
    // Hardcoded for demo purposes
    private let serverBaseURL = "http://100.65.56.136:8000"
    
    private var cancellables = Set<AnyCancellable>()
    
    init() {
        // Check if we have a stored token
        checkAuthenticationStatus()
    }
    
    func checkAuthenticationStatus() {
        if let token = UserDefaults.standard.string(forKey: "auth_token"), !token.isEmpty {
            // Validate token with backend
            validateToken(token)
        } else {
            self.isAuthenticated = false
        }
    }
    
    private func validateToken(_ token: String) {
        isLoading = true
        
        let url = URL(string: "\(serverBaseURL)/auth/validate-token")!
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.addValue("application/json", forHTTPHeaderField: "Content-Type")
        request.addValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        
        URLSession.shared.dataTaskPublisher(for: request)
            .map(\.data)
            .decode(type: TokenValidationResponse.self, decoder: JSONDecoder())
            .receive(on: DispatchQueue.main)
            .sink(receiveCompletion: { [weak self] completion in
                self?.isLoading = false
                if case .failure = completion {
                    self?.isAuthenticated = false
                    UserDefaults.standard.removeObject(forKey: "auth_token")
                }
            }, receiveValue: { [weak self] response in
                self?.isAuthenticated = response.valid
                if !response.valid {
                    UserDefaults.standard.removeObject(forKey: "auth_token")
                }
            })
            .store(in: &cancellables)
    }
    
    func login(email: String, password: String) {
        isLoading = true
        errorMessage = nil
        
        let url = URL(string: "\(serverBaseURL)/auth/login")!
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.addValue("application/json", forHTTPHeaderField: "Content-Type")
        
        let body: [String: String] = ["email": email, "password": password]
        request.httpBody = try? JSONEncoder().encode(body)
        
        URLSession.shared.dataTaskPublisher(for: request)
            .map(\.data)
            .decode(type: LoginResponse.self, decoder: JSONDecoder())
            .receive(on: DispatchQueue.main)
            .sink(receiveCompletion: { [weak self] completion in
                self?.isLoading = false
                if case .failure(let error) = completion {
                    self?.errorMessage = "Login failed: \(error.localizedDescription)"
                }
            }, receiveValue: { [weak self] response in
                UserDefaults.standard.set(response.token, forKey: "auth_token")
                UserDefaults.standard.set(email, forKey: "user_email")
                self?.isAuthenticated = true
            })
            .store(in: &cancellables)
    }
    
    func register(email: String, password: String) {
        isLoading = true
        errorMessage = nil
        
        let url = URL(string: "\(serverBaseURL)/auth/register")!
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.addValue("application/json", forHTTPHeaderField: "Content-Type")
        
        let body: [String: String] = ["email": email, "password": password]
        request.httpBody = try? JSONEncoder().encode(body)
        
        URLSession.shared.dataTaskPublisher(for: request)
            .map(\.data)
            .decode(type: RegisterResponse.self, decoder: JSONDecoder())
            .receive(on: DispatchQueue.main)
            .sink(receiveCompletion: { [weak self] completion in
                self?.isLoading = false
                if case .failure(let error) = completion {
                    self?.errorMessage = "Registration failed: \(error.localizedDescription)"
                }
            }, receiveValue: { [weak self] response in
                if response.success {
                    // After registration success, perform login
                    self?.login(email: email, password: password)
                } else {
                    self?.errorMessage = response.message ?? "Registration failed"
                }
            })
            .store(in: &cancellables)
    }
    
    func logout() {
        UserDefaults.standard.removeObject(forKey: "auth_token")
        UserDefaults.standard.removeObject(forKey: "user_email")
        isAuthenticated = false
    }
}

// Response Models
struct LoginResponse: Codable {
    let access_token: String
    let refresh_token: String
    let expires_at: Int
    let user: UserInfo
    
    var token: String {
        return access_token
    }
}

struct UserInfo: Codable {
    let id: String
    let email: String
}

struct RegisterResponse: Codable {
    let success: Bool
    let message: String?
}

struct TokenValidationResponse: Codable {
    let valid: Bool
} 