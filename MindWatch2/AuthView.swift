import SwiftUI

struct AuthView: View {
    @EnvironmentObject private var authViewModel: AuthViewModel
    @State private var email = ""
    @State private var password = ""
    @State private var isRegistering = false
    
    var body: some View {
        NavigationView {
            VStack {
                // Logo and app name
                VStack(spacing: 12) {
                    Image(systemName: "brain.head.profile")
                        .resizable()
                        .aspectRatio(contentMode: .fit)
                        .frame(width: 80, height: 80)
                        .foregroundColor(.accentColor)
                    
                    Text("MindWatch")
                        .font(.largeTitle)
                        .fontWeight(.bold)
                    
                    Text("Monitor your mental wellbeing with health data")
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                        .multilineTextAlignment(.center)
                        .padding(.bottom, 20)
                }
                .padding(.top, 60)
                
                // Form fields
                VStack(spacing: 20) {
                    TextField("Email", text: $email)
                        .keyboardType(.emailAddress)
                        .autocapitalization(.none)
                        .padding()
                        .background(Color(.systemGray6))
                        .cornerRadius(8)
                    
                    SecureField("Password", text: $password)
                        .padding()
                        .background(Color(.systemGray6))
                        .cornerRadius(8)
                    
                    if let errorMessage = authViewModel.errorMessage {
                        Text(errorMessage)
                            .foregroundColor(.red)
                            .font(.caption)
                            .padding(.vertical, 8)
                    }
                    
                    // Action button
                    Button(action: {
                        if isRegistering {
                            authViewModel.register(email: email, password: password)
                        } else {
                            authViewModel.login(email: email, password: password)
                        }
                    }) {
                        if authViewModel.isLoading {
                            ProgressView()
                                .progressViewStyle(CircularProgressViewStyle(tint: .white))
                                .frame(maxWidth: .infinity)
                                .padding()
                                .background(Color.accentColor)
                                .foregroundColor(.white)
                                .cornerRadius(8)
                        } else {
                            Text(isRegistering ? "Sign Up" : "Log In")
                                .fontWeight(.bold)
                                .frame(maxWidth: .infinity)
                                .padding()
                                .background(Color.accentColor)
                                .foregroundColor(.white)
                                .cornerRadius(8)
                        }
                    }
                    .disabled(email.isEmpty || password.isEmpty || authViewModel.isLoading)
                    
                    // Switch between login and register
                    Button(action: {
                        isRegistering.toggle()
                        authViewModel.errorMessage = nil
                    }) {
                        Text(isRegistering ? "Already have an account? Log In" : "Don't have an account? Sign Up")
                            .foregroundColor(.accentColor)
                            .font(.subheadline)
                    }
                    .padding(.top, 8)
                }
                .padding(.horizontal, 30)
                
                Spacer()
                
                // Privacy info
                VStack(spacing: 8) {
                    Text("Your health data is securely stored")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    
                    Text("Privacy Policy • Terms of Service")
                        .font(.caption2)
                        .foregroundColor(.blue)
                }
                .padding(.bottom, 20)
            }
            .navigationBarHidden(true)
        }
    }
} 