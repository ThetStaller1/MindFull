import SwiftUI
import HealthKit

@main
struct MindWatchApp: App {
    @StateObject private var authViewModel = AuthViewModel()
    @StateObject private var healthViewModel = HealthViewModel()
    
    var body: some Scene {
        WindowGroup {
            if authViewModel.isAuthenticated {
                MainTabView()
                    .environmentObject(authViewModel)
                    .environmentObject(healthViewModel)
                    .onAppear {
                        if !healthViewModel.isAuthorized {
                            healthViewModel.requestAuthorization()
                        }
                    }
            } else {
                AuthView()
                    .environmentObject(authViewModel)
            }
        }
    }
} 