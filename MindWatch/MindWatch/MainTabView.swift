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
    @State private var showYearPicker = false
    @State private var showMonthPicker = false
    
    // Colors
    private let primaryColor = Color.blue
    private let backgroundColor = Color(.systemBackground)
    private let cardBackgroundColor = Color(.secondarySystemBackground)
    
    let years = Array((Calendar.current.component(.year, from: Date()) - 100)...(Calendar.current.component(.year, from: Date())))
    let months = [
        (1, "January"), (2, "February"), (3, "March"), (4, "April"),
        (5, "May"), (6, "June"), (7, "July"), (8, "August"),
        (9, "September"), (10, "October"), (11, "November"), (12, "December")
    ]
    
    var body: some View {
        NavigationView {
            ZStack {
                backgroundColor.edgesIgnoringSafeArea(.all)
                
                ScrollView {
                    VStack(spacing: 24) {
                        // Profile Header
                        profileHeader
                        
                        // Profile Information Card
                        profileInfoCard
                        
                        // Logout Button
                        logoutButton
                    }
                    .padding(.bottom, 20)
                }
            }
            .navigationTitle("Profile")
            .navigationBarTitleDisplayMode(.inline)
            .onAppear {
                // Load existing profile data
                let profile = profileViewModel.userProfile
                selectedGender = profile.gender
                birthYear = profile.birthYear > 0 ? profile.birthYear : Calendar.current.component(.year, from: Date()) - 30
                birthMonth = profile.birthMonth > 0 ? profile.birthMonth : Calendar.current.component(.month, from: Date())
            }
        }
    }
    
    // MARK: - UI Components
    
    private var profileHeader: some View {
        VStack(spacing: 16) {
            // Profile Image
            ZStack {
                Circle()
                    .fill(cardBackgroundColor)
                    .frame(width: 110, height: 110)
                    .shadow(color: Color.black.opacity(0.1), radius: 4, x: 0, y: 2)
                
                Image(systemName: "person.crop.circle.fill")
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .frame(width: 90, height: 90)
                    .foregroundColor(primaryColor)
            }
            .padding(.top, 20)
            
            // User Email
            Text(UserDefaults.standard.string(forKey: "user_email") ?? "User")
                .font(.headline)
                .fontWeight(.semibold)
                .foregroundColor(.primary)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 24)
    }
    
    private var profileInfoCard: some View {
        VStack(spacing: 20) {
            // Card Header
            HStack {
                Text("Personal Information")
                    .font(.headline)
                    .fontWeight(.bold)
                
                Spacer()
                
                Image(systemName: "person.text.rectangle")
                    .foregroundColor(primaryColor)
            }
            .padding(.bottom, 8)
            
            // Gender Selection
            VStack(alignment: .leading, spacing: 8) {
                Text("Gender")
                    .font(.subheadline)
                    .fontWeight(.medium)
                    .foregroundColor(.secondary)
                
                // Modern segmented picker
                HStack {
                    ForEach(["MALE", "FEMALE", "OTHER", "PREFER_NOT_TO_ANSWER"], id: \.self) { gender in
                        genderButton(gender)
                    }
                }
            }
            .padding(.bottom, 16)
            
            // Birth Year Selection
            VStack(alignment: .leading, spacing: 8) {
                Text("Birth Year")
                    .font(.subheadline)
                    .fontWeight(.medium)
                    .foregroundColor(.secondary)
                
                Button(action: {
                    withAnimation {
                        showYearPicker.toggle()
                    }
                }) {
                    HStack {
                        Text(String(birthYear))
                            .fontWeight(.medium)
                            .foregroundColor(.primary)
                        
                        Spacer()
                        
                        Image(systemName: "chevron.up.chevron.down")
                            .foregroundColor(primaryColor)
                            .font(.system(size: 14))
                    }
                    .padding()
                    .background(Color(.systemGray6))
                    .cornerRadius(10)
                }
                
                if showYearPicker {
                    Picker("Birth Year", selection: $birthYear) {
                        ForEach(years, id: \.self) { year in
                            Text(String(year)).tag(year)
                        }
                    }
                    .pickerStyle(WheelPickerStyle())
                    .frame(height: 150)
                    .clipped()
                    .transition(.opacity)
                }
            }
            .padding(.bottom, 16)
            
            // Birth Month Selection
            VStack(alignment: .leading, spacing: 8) {
                Text("Birth Month")
                    .font(.subheadline)
                    .fontWeight(.medium)
                    .foregroundColor(.secondary)
                
                Button(action: {
                    withAnimation {
                        showMonthPicker.toggle()
                    }
                }) {
                    HStack {
                        Text(months.first { $0.0 == birthMonth }?.1 ?? "Select Month")
                            .fontWeight(.medium)
                            .foregroundColor(.primary)
                        
                        Spacer()
                        
                        Image(systemName: "chevron.up.chevron.down")
                            .foregroundColor(primaryColor)
                            .font(.system(size: 14))
                    }
                    .padding()
                    .background(Color(.systemGray6))
                    .cornerRadius(10)
                }
                
                if showMonthPicker {
                    Picker("Birth Month", selection: $birthMonth) {
                        ForEach(months, id: \.0) { month in
                            Text(month.1).tag(month.0)
                        }
                    }
                    .pickerStyle(WheelPickerStyle())
                    .frame(height: 150)
                    .clipped()
                    .transition(.opacity)
                }
            }
            .padding(.bottom, 16)
            
            // Save Button
            Button(action: {
                profileViewModel.updateProfile(
                    gender: selectedGender,
                    birthYear: birthYear,
                    birthMonth: birthMonth
                )
                showingSaveConfirmation = true
            }) {
                HStack {
                    Spacer()
                    
                    if profileViewModel.isSaving {
                        ProgressView()
                            .progressViewStyle(CircularProgressViewStyle(tint: .white))
                            .padding(.trailing, 5)
                    }
                    
                    Text("Save Changes")
                        .fontWeight(.semibold)
                    
                    Spacer()
                }
                .padding()
                .background(primaryColor)
                .foregroundColor(.white)
                .cornerRadius(15)
                .shadow(color: primaryColor.opacity(0.3), radius: 5, x: 0, y: 3)
            }
            .padding(.top, 8)
            .alert(isPresented: $showingSaveConfirmation) {
                Alert(
                    title: Text("Profile Saved"),
                    message: Text("Your profile information has been updated."),
                    dismissButton: .default(Text("OK"))
                )
            }
            
            // Error Message
            if let error = profileViewModel.errorMessage {
                Text(error)
                    .font(.footnote)
                    .foregroundColor(.red)
                    .padding(.top, 8)
                    .transition(.opacity)
            }
        }
        .padding(20)
        .background(cardBackgroundColor)
        .cornerRadius(20)
        .shadow(color: Color.black.opacity(0.05), radius: 10, x: 0, y: 5)
        .padding(.horizontal, 16)
    }
    
    private var logoutButton: some View {
        Button(action: {
            authViewModel.logout()
        }) {
            HStack {
                Image(systemName: "rectangle.portrait.and.arrow.right")
                Text("Log Out")
                    .fontWeight(.medium)
            }
            .padding()
            .frame(maxWidth: .infinity)
            .background(Color.red.opacity(0.9))
            .foregroundColor(.white)
            .cornerRadius(15)
            .shadow(color: Color.red.opacity(0.3), radius: 5, x: 0, y: 3)
        }
        .padding(.horizontal, 20)
        .padding(.top, 10)
    }
    
    private func genderButton(_ gender: String) -> some View {
        let displayText: String
        switch gender {
        case "MALE":
            displayText = "Male"
        case "FEMALE":
            displayText = "Female"
        case "OTHER":
            displayText = "Other"
        case "PREFER_NOT_TO_ANSWER":
            displayText = "Not Specified"
        default:
            displayText = gender
        }
        
        return Button(action: {
            withAnimation {
                selectedGender = gender
            }
        }) {
            Text(displayText)
                .font(.subheadline)
                .padding(.vertical, 8)
                .padding(.horizontal, 12)
                .frame(maxWidth: .infinity)
                .background(selectedGender == gender ? primaryColor : Color(.systemGray6))
                .foregroundColor(selectedGender == gender ? .white : .primary)
                .cornerRadius(8)
                .animation(.easeInOut(duration: 0.2), value: selectedGender)
        }
    }
} 