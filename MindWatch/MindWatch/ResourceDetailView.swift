import SwiftUI
import WebKit

struct ResourceDetailView: View {
    let resource: Resource
    @State private var showWebView = false
    
    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                Text(resource.title)
                    .font(.largeTitle)
                    .fontWeight(.bold)
                    .padding(.horizontal)
                
                Text(resource.description)
                    .font(.body)
                    .foregroundColor(.secondary)
                    .padding(.horizontal)
                
                Divider()
                    .padding(.horizontal)
                
                // Additional content can be added here based on resource type
                
                Button(action: {
                    showWebView = true
                }) {
                    HStack {
                        Text("View Full Resource")
                            .fontWeight(.semibold)
                        
                        Image(systemName: "arrow.up.right.square")
                    }
                    .padding()
                    .foregroundColor(.white)
                    .background(Color.blue)
                    .cornerRadius(10)
                }
                .padding(.horizontal)
                .sheet(isPresented: $showWebView) {
                    SafariWebView(url: URL(string: resource.link)!)
                }
                
                Spacer()
            }
            .padding(.vertical)
        }
        .navigationTitle("Resource Details")
        .navigationBarTitleDisplayMode(.inline)
    }
}

// WebKit WebView implementation for displaying web content
struct SafariWebView: UIViewRepresentable {
    let url: URL
    
    func makeUIView(context: Context) -> WKWebView {
        let webView = WKWebView()
        webView.load(URLRequest(url: url))
        return webView
    }
    
    func updateUIView(_ uiView: WKWebView, context: Context) {
        // Not needed for this implementation
    }
}

struct ResourceDetailView_Previews: PreviewProvider {
    static var previews: some View {
        NavigationView {
            ResourceDetailView(
                resource: Resource(
                    title: "Understanding Anxiety Disorders",
                    description: "Anxiety disorders involve excessive worry or fear that interferes with daily activities.",
                    link: "https://www.nimh.nih.gov/health/topics/anxiety-disorders"
                )
            )
        }
    }
} 