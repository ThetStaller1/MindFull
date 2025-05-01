//
//  MindWatchApp.swift
//  MindWatch
//
//  Created by Nico on 4/22/25.
//

import SwiftUI
import HealthKit

// MARK: - App with direct data flow to backend
// In this presentation branch, the app sends HealthKit data directly to the backend
// for analysis without storing it in Supabase first. Only auth and analysis results
// use Supabase. This simplifies the flow for demonstration purposes.

@main
struct MindWatchApp: App {
    @StateObject private var authViewModel = AuthViewModel()
    @StateObject private var healthViewModel = HealthViewModel()
    
    var body: some Scene {
        WindowGroup {
            if authViewModel.isAuthenticated {
                MainTabView(authViewModel: authViewModel)
                    .environmentObject(authViewModel)
                    .environmentObject(healthViewModel)
                    .onAppear {
                        print("MainTabView appeared, checking HealthKit authorization")
                        // Request HealthKit authorization as soon as the main view appears
                        healthViewModel.requestAuthorization()
                    }
            } else {
                AuthView(authViewModel: authViewModel)
                    .environmentObject(authViewModel)
                    .onChange(of: authViewModel.isAuthenticated) { newValue in
                        if newValue {
                            // User just logged in, request HealthKit authorization
                            print("User authenticated, requesting HealthKit authorization")
                            healthViewModel.requestAuthorization()
                        }
                    }
            }
        }
    }
}
