import SwiftUI

struct MainTabView: View {
    @EnvironmentObject private var authViewModel: AuthViewModel
    @EnvironmentObject private var healthViewModel: HealthViewModel
    
    var body: some View {
        TabView {
            HealthDataView()
                .tabItem {
                    Label("Health Data", systemImage: "heart.fill")
                }
            
            AnalysisView()
                .tabItem {
                    Label("Analysis", systemImage: "brain.head.profile")
                }
            
            ProfileView()
                .tabItem {
                    Label("Profile", systemImage: "person.fill")
                }
        }
    }
}

struct ProfileView: View {
    @EnvironmentObject private var authViewModel: AuthViewModel
    
    var body: some View {
        NavigationView {
            VStack(spacing: 20) {
                // User information
                VStack(spacing: 12) {
                    Image(systemName: "person.circle.fill")
                        .resizable()
                        .aspectRatio(contentMode: .fit)
                        .frame(width: 100, height: 100)
                        .foregroundColor(.accentColor)
                    
                    Text(UserDefaults.standard.string(forKey: "user_email") ?? "User")
                        .font(.title)
                        .fontWeight(.bold)
                }
                .padding(.top, 40)
                
                Spacer()
                
                // Logout button
                Button(action: {
                    authViewModel.logout()
                }) {
                    Text("Log Out")
                        .fontWeight(.bold)
                        .foregroundColor(.white)
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(Color.red)
                        .cornerRadius(10)
                }
                .padding(.horizontal, 40)
                .padding(.bottom, 40)
            }
            .navigationTitle("Profile")
        }
    }
} 