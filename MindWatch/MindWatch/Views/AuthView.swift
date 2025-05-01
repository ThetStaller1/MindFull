import SwiftUI

struct AuthView: View {
    @ObservedObject var authViewModel: AuthViewModel
    
    @State private var email = ""
    @State private var password = ""
    @State private var isRegistering = false
    
    var body: some View {
        VStack {
            // App Logo
            Image(systemName: "brain.head.profile")
                .resizable()
                .aspectRatio(contentMode: .fit)
                .frame(width: 100, height: 100)
                .foregroundColor(.blue)
                .padding(.bottom, 20)
            
            Text("MindWatch")
                .font(.system(size: 32, weight: .bold))
                .padding(.bottom, 40)
            
            // Input fields
            VStack(spacing: 20) {
                TextField("Email", text: $email)
                    .padding()
                    .background(Color(.systemGray6))
                    .cornerRadius(8)
                    .keyboardType(.emailAddress)
                    .autocapitalization(.none)
                
                SecureField("Password", text: $password)
                    .padding()
                    .background(Color(.systemGray6))
                    .cornerRadius(8)
            }
            .padding(.horizontal)
            
            // Error message
            if let errorMessage = authViewModel.errorMessage {
                Text(errorMessage)
                    .foregroundColor(.red)
                    .font(.caption)
                    .padding()
            }
            
            // Login/Register button
            Button(action: {
                if isRegistering {
                    authViewModel.register(email: email, password: password)
                } else {
                    authViewModel.login(email: email, password: password)
                }
            }) {
                HStack {
                    if authViewModel.isLoading {
                        ProgressView()
                            .progressViewStyle(CircularProgressViewStyle(tint: .white))
                    }
                    
                    Text(isRegistering ? "Register" : "Log In")
                        .fontWeight(.semibold)
                }
                .frame(maxWidth: .infinity)
                .padding()
                .background(Color.blue)
                .foregroundColor(.white)
                .cornerRadius(10)
            }
            .padding(.horizontal)
            .disabled(authViewModel.isLoading)
            
            // Switch between login and register
            Button(action: {
                isRegistering.toggle()
            }) {
                Text(isRegistering ? "Already have an account? Log In" : "Don't have an account? Register")
                    .foregroundColor(.blue)
            }
            .padding()
            
            Spacer()
        }
        .padding()
    }
}

struct AuthView_Previews: PreviewProvider {
    static var previews: some View {
        AuthView(authViewModel: AuthViewModel())
    }
} 