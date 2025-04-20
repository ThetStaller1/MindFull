//
//  ContentView.swift
//  MindWatch
//
//  Created by Nico on 4/14/25.
//

import SwiftUI
import SwiftData

struct ContentView: View {
    @Environment(\.modelContext) private var modelContext
    @State private var selectedTab = 0
    
    // Create view model to be shared across views
    @StateObject private var healthViewModel: HealthViewModel
    
    init() {
        // Create a temporary view model - this will be replaced when the view appears
        _healthViewModel = StateObject(wrappedValue: HealthViewModel(modelContext: ModelContext(try! ModelContainer(for: HealthData.self, configurations: ModelConfiguration(isStoredInMemoryOnly: true)))))
    }
    
    var body: some View {
        TabView(selection: $selectedTab) {
            HealthDataView(viewModel: healthViewModel)
                .tabItem {
                    Label("Health Data", systemImage: "heart.fill")
                }
                .tag(0)
            
            MentalHealthView(viewModel: healthViewModel)
                .tabItem {
                    Label("Mental Health", systemImage: "brain.fill")
                }
                .tag(1)
        }
        .onAppear {
            // Replace the view model with one using the environment's model context
            healthViewModel.updateModelContext(modelContext)
        }
    }
}

#Preview {
    ContentView()
        .modelContainer(for: HealthData.self, inMemory: true)
}
