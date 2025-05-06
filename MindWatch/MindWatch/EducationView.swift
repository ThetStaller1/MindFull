import SwiftUI

struct EducationView: View {
    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 20) {
                    // Introduction section
                    VStack(alignment: .leading, spacing: 15) {
                        Text("Educational Resources")
                            .font(.title)
                            .fontWeight(.bold)
                            .padding(.horizontal)
                        
                        Text("Learn about mental health conditions and how to seek help.")
                            .font(.subheadline)
                            .foregroundColor(.secondary)
                            .padding(.horizontal)
                    }
                    .padding(.top)
                    
                    // Mental Health Conditions Section
                    resourceSection(
                        title: "Understanding Mental Health",
                        icon: "brain.head.profile",
                        color: .blue,
                        description: "Learn about common mental health conditions, their symptoms, and treatments.",
                        resources: [
                            Resource(
                                title: "Understanding Anxiety Disorders",
                                description: "Anxiety disorders involve excessive worry or fear that interferes with daily activities.",
                                link: "https://www.nimh.nih.gov/health/topics/anxiety-disorders"
                            ),
                            Resource(
                                title: "Depression Basics",
                                description: "Depression is a serious mood disorder that affects how you feel, think, and handle daily activities.",
                                link: "https://www.nimh.nih.gov/health/publications/depression"
                            ),
                            Resource(
                                title: "Bipolar Disorder",
                                description: "Bipolar disorder causes unusual shifts in mood, energy, activity levels, and the ability to carry out day-to-day tasks.",
                                link: "https://www.nimh.nih.gov/health/topics/bipolar-disorder"
                            ),
                            Resource(
                                title: "Post-Traumatic Stress Disorder",
                                description: "PTSD can develop after experiencing or witnessing a traumatic event.",
                                link: "https://www.nimh.nih.gov/health/topics/post-traumatic-stress-disorder-ptsd"
                            ),
                            Resource(
                                title: "The Teen Brain",
                                description: "Learn about how the teen brain develops and adapts to the world.",
                                link: "https://www.nimh.nih.gov/health/publications/the-teen-brain-7-things-to-know"
                            )
                        ]
                    )
                    
                    // Seeking Help Section
                    resourceSection(
                        title: "Finding Professional Help",
                        icon: "hand.raised.fill",
                        color: .green,
                        description: "Resources to help you find and connect with mental health professionals.",
                        resources: [
                            Resource(
                                title: "Types of Mental Health Professionals",
                                description: "Learn about the different types of mental health providers and their roles.",
                                link: "https://www.nami.org/About-Mental-Illness/Treatment/Types-of-Mental-Health-Professionals"
                            ),
                            Resource(
                                title: "How to Choose a Mental Health Provider",
                                description: "Tips for finding the right mental health professional for your needs.",
                                link: "https://www.nami.org/Your-Journey/Individuals-with-Mental-Illness/Finding-a-Mental-Health-Professional"
                            ),
                            Resource(
                                title: "Mental Health Support Groups",
                                description: "Connect with others who understand what you're going through.",
                                link: "https://www.nami.org/Support-Education/Support-Groups"
                            ),
                            Resource(
                                title: "Crisis Resources",
                                description: "Immediate help for mental health crises. In emergencies, call or text 988.",
                                link: "https://988lifeline.org"
                            ),
                            Resource(
                                title: "Help for Family and Caregivers",
                                description: "Resources for those supporting someone with mental health challenges.",
                                link: "https://www.nami.org/Your-Journey/Family-Members-and-Caregivers"
                            )
                        ]
                    )
                    
                    // Quick Mental Health Tips Section
                    VStack(alignment: .leading, spacing: 15) {
                        HStack {
                            Image(systemName: "lightbulb.fill")
                                .foregroundColor(.yellow)
                                .font(.title2)
                            
                            Text("Quick Mental Health Tips")
                                .font(.title2)
                                .fontWeight(.semibold)
                        }
                        .padding(.horizontal)
                        
                        Text("Simple strategies that can help improve your mental wellbeing.")
                            .font(.subheadline)
                            .foregroundColor(.secondary)
                            .padding(.horizontal)
                        
                        VStack(spacing: 15) {
                            mentalHealthTip(
                                title: "Practice Mindfulness",
                                description: "Focus on the present moment without judgment. Try breathing deeply for 5 minutes daily.",
                                icon: "brain.head.profile"
                            )
                            
                            mentalHealthTip(
                                title: "Stay Physically Active",
                                description: "Regular exercise can reduce anxiety and improve mood by releasing endorphins.",
                                icon: "figure.walk"
                            )
                            
                            mentalHealthTip(
                                title: "Maintain Social Connections",
                                description: "Spend time with supportive friends and family, even if it's just a short call.",
                                icon: "person.2.fill"
                            )
                            
                            mentalHealthTip(
                                title: "Get Enough Sleep",
                                description: "Aim for 7-9 hours of quality sleep each night to support mental health.",
                                icon: "bed.double.fill"
                            )
                            
                            mentalHealthTip(
                                title: "Limit Media Consumption",
                                description: "Take breaks from news and social media to reduce stress and anxiety.",
                                icon: "phone.down.fill"
                            )
                        }
                        .padding(.horizontal)
                    }
                    .padding(.vertical)
                    .background(
                        RoundedRectangle(cornerRadius: 12)
                            .fill(Color(.systemGray6))
                    )
                    .padding(.horizontal)
                    
                    // Crisis Information
                    VStack(alignment: .center, spacing: 10) {
                        Text("In a crisis?")
                            .font(.headline)
                        
                        Text("Call or text 988")
                            .font(.title2)
                            .fontWeight(.bold)
                            .foregroundColor(.red)
                        
                        Text("The 988 Suicide & Crisis Lifeline provides 24/7 support")
                            .font(.subheadline)
                            .multilineTextAlignment(.center)
                    }
                    .padding()
                    .background(
                        RoundedRectangle(cornerRadius: 12)
                            .fill(Color(.systemGray6))
                    )
                    .padding(.horizontal)
                    
                    Spacer(minLength: 40)
                }
                .padding(.bottom)
                .navigationTitle("Education")
            }
        }
    }
    
    // Helper function for mental health tips
    private func mentalHealthTip(title: String, description: String, icon: String) -> some View {
        HStack(alignment: .top, spacing: 15) {
            Image(systemName: icon)
                .font(.title3)
                .foregroundColor(.blue)
                .frame(width: 30)
            
            VStack(alignment: .leading, spacing: 5) {
                Text(title)
                    .font(.headline)
                
                Text(description)
                    .font(.subheadline)
                    .foregroundColor(.secondary)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 8)
                .stroke(Color(.systemGray4), lineWidth: 1)
        )
    }
    
    // Helper function to create consistent resource sections
    private func resourceSection(title: String, icon: String, color: Color, description: String, resources: [Resource]) -> some View {
        VStack(alignment: .leading, spacing: 15) {
            HStack {
                Image(systemName: icon)
                    .foregroundColor(color)
                    .font(.title2)
                
                Text(title)
                    .font(.title2)
                    .fontWeight(.semibold)
            }
            .padding(.horizontal)
            
            Text(description)
                .font(.subheadline)
                .foregroundColor(.secondary)
                .padding(.horizontal)
            
            VStack(spacing: 15) {
                ForEach(resources) { resource in
                    ResourceItemView(resource: resource)
                        .padding(.horizontal)
                }
            }
        }
        .padding(.vertical)
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(Color(.systemGray6))
        )
        .padding(.horizontal)
    }
}

// Model for educational resources
struct Resource: Identifiable {
    let id = UUID()
    let title: String
    let description: String
    let link: String
}

// View for individual resource items
struct ResourceItemView: View {
    let resource: Resource
    @State private var showSafari = false
    
    var body: some View {
        NavigationLink(destination: ResourceDetailView(resource: resource)) {
            VStack(alignment: .leading, spacing: 5) {
                Text(resource.title)
                    .font(.headline)
                    .foregroundColor(.primary)
                
                Text(resource.description)
                    .font(.subheadline)
                    .foregroundColor(.secondary)
                    .lineLimit(3)
                
                HStack {
                    Text("Learn more")
                        .font(.caption)
                        .foregroundColor(.blue)
                    
                    Image(systemName: "arrow.right")
                        .font(.caption)
                        .foregroundColor(.blue)
                }
            }
            .padding()
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(
                RoundedRectangle(cornerRadius: 8)
                    .stroke(Color(.systemGray4), lineWidth: 1)
            )
        }
        .buttonStyle(PlainButtonStyle())
    }
}

struct EducationView_Previews: PreviewProvider {
    static var previews: some View {
        EducationView()
    }
} 