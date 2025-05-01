import SwiftUI

struct MainTabView: View {
    @StateObject private var healthViewModel = HealthViewModel()
    @ObservedObject var authViewModel: AuthViewModel
    
    var body: some View {
        TabView {
            // Home Tab
            AnalysisView()
                .tabItem {
                    Label("Home", systemImage: "house.fill")
                }
            
            // Health Data Tab
            HealthDataView()
                .tabItem {
                    Label("Health", systemImage: "heart.fill")
                }
            
            // Profile Tab
            ProfileView(authViewModel: authViewModel)
                .tabItem {
                    Label("Profile", systemImage: "person.fill")
                }
        }
        .environmentObject(healthViewModel)
        .onAppear {
            print("MainTabView appeared, checking HealthKit authorization")
            if !healthViewModel.isAuthorized {
                healthViewModel.requestAuthorization()
            }
        }
    }
}

struct MainTabView_Previews: PreviewProvider {
    static var previews: some View {
        MainTabView(authViewModel: AuthViewModel())
    }
} 