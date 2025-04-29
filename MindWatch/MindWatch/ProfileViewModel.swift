import Foundation
import Combine

class ProfileViewModel: ObservableObject {
    @Published var userProfile = UserProfile()
    @Published var isSaving = false
    @Published var errorMessage: String? = nil
    
    private let profileKey = "user_profile"
    
    init() {
        loadProfile()
    }
    
    private func loadProfile() {
        if let data = UserDefaults.standard.data(forKey: profileKey) {
            let decoder = JSONDecoder()
            if let profile = try? decoder.decode(UserProfile.self, from: data) {
                self.userProfile = profile
            }
        }
    }
    
    func saveProfile() {
        let encoder = JSONEncoder()
        if let data = try? encoder.encode(userProfile) {
            UserDefaults.standard.set(data, forKey: profileKey)
        }
    }
    
    func updateProfile(gender: String, birthYear: Int, birthMonth: Int) {
        userProfile.gender = gender
        userProfile.birthYear = birthYear
        userProfile.birthMonth = birthMonth
        
        // Save to UserDefaults
        saveProfile()
        
        // Send to backend
        saveProfileToBackend()
    }
    
    func saveProfileToBackend() {
        guard let userId = UserDefaults.standard.string(forKey: "user_id"),
              let authToken = UserDefaults.standard.string(forKey: "auth_token") else {
            errorMessage = "Not logged in"
            return
        }
        
        isSaving = true
        errorMessage = nil
        
        guard let baseURL = URL(string: "http://192.168.1.241:8000") else {
            errorMessage = "Invalid API URL"
            isSaving = false
            return
        }
        
        let endpoint = baseURL.appendingPathComponent("update-profile")
        
        // Create request
        var request = URLRequest(url: endpoint)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("Bearer \(authToken)", forHTTPHeaderField: "Authorization")
        
        // Create request body
        var body: [String: Any] = userProfile.toBackendDict()
        body["userId"] = userId
        
        do {
            request.httpBody = try JSONSerialization.data(withJSONObject: body)
        } catch {
            errorMessage = "Error creating request: \(error.localizedDescription)"
            isSaving = false
            return
        }
        
        // Make the request
        URLSession.shared.dataTask(with: request) { [weak self] data, response, error in
            DispatchQueue.main.async {
                self?.isSaving = false
                
                if let error = error {
                    self?.errorMessage = "Network error: \(error.localizedDescription)"
                    return
                }
                
                guard let httpResponse = response as? HTTPURLResponse else {
                    self?.errorMessage = "Invalid response"
                    return
                }
                
                if httpResponse.statusCode != 200 {
                    if let data = data,
                       let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                       let detail = json["detail"] as? String {
                        self?.errorMessage = detail
                    } else {
                        self?.errorMessage = "Failed to update profile: status \(httpResponse.statusCode)"
                    }
                    return
                }
                
                // Update was successful
                print("Profile updated successfully")
            }
        }.resume()
    }
} 