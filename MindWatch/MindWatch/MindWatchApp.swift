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
    
    var body: some Scene {
        WindowGroup {
            if authViewModel.isAuthenticated {
                MainTabView()
                    .environmentObject(authViewModel)
                    .environmentObject(healthViewModel)
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
