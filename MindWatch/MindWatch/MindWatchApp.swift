//
//  MindWatchApp.swift
//  MindWatch
//
//  Created by Nico on 4/22/25.
//

import SwiftUI
import HealthKit

@main
struct MindWatchApp: App {
    @StateObject private var authViewModel = AuthViewModel()
    @StateObject private var healthViewModel = HealthViewModel()
    @StateObject private var profileViewModel = ProfileViewModel()
    
    var body: some Scene {
        WindowGroup {
            if authViewModel.isAuthenticated {
                MainTabView()
                    .environmentObject(authViewModel)
                    .environmentObject(healthViewModel)
                    .environmentObject(profileViewModel)
                    .onAppear {
                        // Request HealthKit authorization when the app appears
                        healthViewModel.requestHealthKitAuthorization()
                    }
            } else {
                AuthView()
                    .environmentObject(authViewModel)
            }
        }
    }
}
