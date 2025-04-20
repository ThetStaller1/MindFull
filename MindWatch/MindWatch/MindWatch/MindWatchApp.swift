//
//  MindWatchApp.swift
//  MindWatch
//
//  Created by Nico on 4/14/25.
//

import SwiftUI
import SwiftData
import HealthKit

@main
struct MindWatchApp: App {
    @State private var modelContainer: ModelContainer?
    @State private var loadError: String?
    
    init() {
        setupModelContainer()
    }
    
    private func setupModelContainer() {
        let schema = Schema([
            HealthData.self,
        ])
        
        // Get the documents directory URL
        let documentsDirectory = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        let storeURL = documentsDirectory.appendingPathComponent("mindwatch.store")
        
        // First try with the persistent storage
        do {
            modelContainer = try ModelContainer(
                for: schema,
                configurations: ModelConfiguration(schema: schema, url: storeURL)
            )
        } catch {
            print("Failed to create model container with persistent storage: \(error)")
            
            // Fallback to in-memory storage
            do {
                modelContainer = try ModelContainer(
                    for: schema,
                    configurations: ModelConfiguration(isStoredInMemoryOnly: true)
                )
            } catch {
                print("Could not create even in-memory ModelContainer: \(error)")
                loadError = error.localizedDescription
            }
        }
    }

    var body: some Scene {
        WindowGroup {
            if let container = modelContainer {
                ContentView()
                    .modelContainer(container)
            } else {
                VStack {
                    Text("Failed to load data model")
                    if let error = loadError {
                        Text(error)
                            .foregroundColor(.red)
                            .font(.caption)
                    }
                    Button("Retry") {
                        setupModelContainer()
                    }
                }
            }
        }
    }
}
