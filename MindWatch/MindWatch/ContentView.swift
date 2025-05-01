//
//  ContentView.swift
//  MindWatch
//
//  Created by Nico on 4/22/25.
//

import SwiftUI
import SwiftData

struct ContentView: View {
    @StateObject private var authViewModel = AuthViewModel()
    
    var body: some View {
        Group {
            if authViewModel.isAuthenticated {
                MainTabView(authViewModel: authViewModel)
            } else {
                AuthView(authViewModel: authViewModel)
            }
        }
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
