import SwiftUI

struct MainTabView: View {
    @EnvironmentObject private var authViewModel: AuthViewModel
    @EnvironmentObject private var healthViewModel: HealthViewModel
    
    var body: some View {
        TabView {
            AnalysisView()
                .tabItem {
                    Label("Analysis", systemImage: "brain.head.profile")
                }
            
            HealthDataView()
                .tabItem {
                    Label("Health Data", systemImage: "heart.fill")
                }
            
            EducationView()
                .tabItem {
                    Label("Education", systemImage: "book.fill")
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
    @StateObject private var profileViewModel = ProfileViewModel()
    
    @State private var selectedGender = "PREFER_NOT_TO_ANSWER"
    @State private var birthYear = Calendar.current.component(.year, from: Date()) - 30
    @State private var birthMonth = Calendar.current.component(.month, from: Date())
    @State private var showingSaveConfirmation = false
    
    let years = Array((Calendar.current.component(.year, from: Date()) - 100)...(Calendar.current.component(.year, from: Date())))
    let months = [
        (1, "January"), (2, "February"), (3, "March"), (4, "April"),
        (5, "May"), (6, "June"), (7, "July"), (8, "August"),
        (9, "September"), (10, "October"), (11, "November"), (12, "December")
    ]
    
    var body: some View {
        NavigationView {
            ScrollView {
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
                    
                    // Profile information section
                    VStack(alignment: .leading, spacing: 20) {
                        Text("Your Information")
                            .font(.headline)
                            .padding(.horizontal)
                        
                        // Gender selection
                        VStack(alignment: .leading) {
                            Text("Gender")
                                .font(.subheadline)
                                .foregroundColor(.secondary)
                                .padding(.horizontal)
                            
                            Picker("Gender", selection: $selectedGender) {
                                Text("Male").tag("MALE")
                                Text("Female").tag("FEMALE")
                                Text("Other").tag("OTHER")
                                Text("Prefer not to answer").tag("PREFER_NOT_TO_ANSWER")
                            }
                            .pickerStyle(SegmentedPickerStyle())
                            .padding(.horizontal)
                        }
                        
                        // Birth year selection
                        VStack(alignment: .leading) {
                            Text("Birth Year")
                                .font(.subheadline)
                                .foregroundColor(.secondary)
                                .padding(.horizontal)
                            
                            Picker("Birth Year", selection: $birthYear) {
                                ForEach(years, id: \.self) { year in
                                    Text(String(year)).tag(year)
                                }
                            }
                            .pickerStyle(WheelPickerStyle())
                            .frame(height: 100)
                            .clipped()
                            .padding(.horizontal)
                        }
                        
                        // Birth month selection
                        VStack(alignment: .leading) {
                            Text("Birth Month")
                                .font(.subheadline)
                                .foregroundColor(.secondary)
                                .padding(.horizontal)
                            
                            Picker("Birth Month", selection: $birthMonth) {
                                ForEach(months, id: \.0) { month in
                                    Text(month.1).tag(month.0)
                                }
                            }
                            .pickerStyle(WheelPickerStyle())
                            .frame(height: 100)
                            .clipped()
                            .padding(.horizontal)
                        }
                        
                        // Save button
                        Button(action: {
                            profileViewModel.updateProfile(
                                gender: selectedGender,
                                birthYear: birthYear,
                                birthMonth: birthMonth
                            )
                            showingSaveConfirmation = true
                        }) {
                            Text("Save Profile")
                                .fontWeight(.bold)
                                .foregroundColor(.white)
                                .frame(maxWidth: .infinity)
                                .padding()
                                .background(Color.blue)
                                .cornerRadius(10)
                        }
                        .padding(.horizontal)
                        .alert(isPresented: $showingSaveConfirmation) {
                            Alert(
                                title: Text("Profile Saved"),
                                message: Text("Your profile information has been updated."),
                                dismissButton: .default(Text("OK"))
                            )
                        }
                        
                        if let error = profileViewModel.errorMessage {
                            Text(error)
                                .foregroundColor(.red)
                                .padding(.horizontal)
                        }
                    }
                    .padding(.vertical)
                    .background(Color(.systemGray6))
                    .cornerRadius(10)
                    .padding(.horizontal)
                    
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
            }
            .navigationTitle("Profile")
            .onAppear {
                // Load existing profile data
                let profile = profileViewModel.userProfile
                selectedGender = profile.gender
                birthYear = profile.birthYear > 0 ? profile.birthYear : Calendar.current.component(.year, from: Date()) - 30
                birthMonth = profile.birthMonth > 0 ? profile.birthMonth : Calendar.current.component(.month, from: Date())
            }
        }
    }
} 