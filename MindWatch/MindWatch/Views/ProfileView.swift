import SwiftUI

struct ProfileView: View {
    @ObservedObject var authViewModel: AuthViewModel
    @EnvironmentObject var healthViewModel: HealthViewModel
    
    @State private var showingLogoutConfirmation = false
    
    var body: some View {
        NavigationView {
            Form {
                Section(header: Text("Account")) {
                    Text(UserDefaults.standard.string(forKey: "user_email") ?? "Unknown")
                        .foregroundColor(.secondary)
                }
                
                Section(header: Text("Health Data")) {
                    if let lastUpdated = healthViewModel.lastUpdated {
                        HStack {
                            Text("Last sync")
                            Spacer()
                            Text(lastUpdated, style: .date)
                                .foregroundColor(.secondary)
                        }
                        
                        HStack {
                            Text("Last sync time")
                            Spacer()
                            Text(lastUpdated, style: .time)
                                .foregroundColor(.secondary)
                        }
                    } else {
                        Text("No health data synced yet")
                            .foregroundColor(.secondary)
                    }
                    
                    Button("Sync Health Data Now") {
                        healthViewModel.fetchHealthData()
                    }
                    .disabled(healthViewModel.isLoading)
                }
                
                Section {
                    Button("Log Out") {
                        showingLogoutConfirmation = true
                    }
                    .foregroundColor(.red)
                }
            }
            .navigationTitle("Profile")
            .alert(isPresented: $showingLogoutConfirmation) {
                Alert(
                    title: Text("Log Out"),
                    message: Text("Are you sure you want to log out?"),
                    primaryButton: .destructive(Text("Log Out")) {
                        authViewModel.logout()
                    },
                    secondaryButton: .cancel()
                )
            }
        }
    }
}

struct ProfileView_Previews: PreviewProvider {
    static var previews: some View {
        ProfileView(authViewModel: AuthViewModel())
            .environmentObject(HealthViewModel())
    }
} 