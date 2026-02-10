"""
Streamlit Web Interface for YouTube Interview Prep Generator
Run with: streamlit run app.py
"""

import streamlit as st
from src.youtube_interview_prep import YouTubeAnalyzer, AnalysisResult
import os


def main():
    st.set_page_config(
        page_title="YouTube Interview Prep Generator",
        page_icon="🎓",
        layout="wide"
    )
    
    st.title("🎓 YouTube Interview Prep Generator")
    st.markdown("""
    Analyze educational YouTube videos and generate comprehensive interview preparation materials including:
    - Common interview questions asked by top companies
    - Real-world use cases and implementations
    - Key concepts and technical topics
    """)
    
    # Sidebar for API key
    with st.sidebar:
        st.header("⚙️ Configuration")
        provider = st.selectbox(
            "Provider",
            ["anthropic", "google"],
            index=0,
            help="Select the AI provider to use"
        )
        if provider == "anthropic":
            api_key = st.text_input(
                "Anthropic API Key",
                type="password",
                help="Get your API key from https://console.anthropic.com/"
            )
        elif provider == "google":
            api_key = st.text_input(
                "Google API Key",
                type="password",
                help="Get your API key from https://aistudio.google.com/app/api-keys"
            )
        
        if not api_key:
            api_key = os.environ.get('ANTHROPIC_API_KEY')
        
        st.markdown("---")
        st.markdown("### 📖 How to Use")
        st.markdown("""
        1. Enter your Anthropic API key
        2. Paste a YouTube video URL
        3. Click 'Analyze Video'
        4. Review the generated interview prep
        5. Download as Markdown or JSON
        """)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        video_url = st.text_input(
            "YouTube Video URL",
            placeholder="https://www.youtube.com/watch?v=..."
        )
    
    with col2:
        video_title = st.text_input(
            "Video Title (Optional)",
            placeholder="e.g., Python Data Structures"
        )
    
    analyze_button = st.button("🚀 Analyze Video", type="primary", use_container_width=True)
    
    if analyze_button:
        if not api_key:
            st.error("⚠️ Please provide an Anthropic API key in the sidebar!")
            return
        
        if not video_url:
            st.error("⚠️ Please provide a YouTube video URL!")
            return
        
        try:
            with st.spinner("🎥 Fetching video transcript..."):
                analyzer = YouTubeAnalyzer(api_key=api_key,)
                transcript = analyzer.get_transcript(video_url)
                st.success(f"✅ Transcript retrieved ({len(transcript)} characters)")
            
            with st.spinner("🤖 Analyzing content with AI... This may take 30-60 seconds..."):
                result = analyzer.analyze_content(transcript, video_title, provider = provider )
            
            st.success("✅ Analysis complete!")
            
            # Store result in session state
            st.session_state['result'] = result
            st.session_state['analyzer'] = analyzer
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            return
    
    # Display results if available
    if 'result' in st.session_state:
        result = st.session_state['result']
        analyzer = st.session_state['analyzer']
        
        # Summary section
        st.markdown("## 📝 Summary")
        st.info(result.summary)
        
        # Stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Main Topics", len(result.main_topics))
        with col2:
            st.metric("Key Concepts", len(result.key_concepts))
        with col3:
            st.metric("Interview Questions", len(result.interview_questions))
        with col4:
            st.metric("Use Cases", len(result.use_cases))
        
        # Tabs for different sections
        tab1, tab2, tab3, tab4 = st.tabs([
            "🎯 Topics & Concepts",
            "❓ Interview Questions",
            "🏢 Use Cases",
            "📥 Export"
        ])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Main Topics")
                for i, topic in enumerate(result.main_topics, 1):
                    st.markdown(f"**{i}.** {topic}")
            
            with col2:
                st.subheader("Key Concepts")
                for concept in result.key_concepts:
                    st.markdown(f"• {concept}")
        
        with tab2:
            # Filter by difficulty
            difficulty_filter = st.selectbox(
                "Filter by Difficulty",
                ["All", "Easy", "Medium", "Hard"]
            )
            
            questions = result.interview_questions
            if difficulty_filter != "All":
                questions = [q for q in questions if q.difficulty == difficulty_filter]
            
            for i, q in enumerate(questions, 1):
                with st.expander(f"**{q.difficulty}** - {q.question}"):
                    st.markdown(f"**Topic:** {q.topic}")
                    st.markdown(f"**Asked by:** {', '.join(q.companies)}")
                    st.markdown("**Sample Answer:**")
                    st.write(q.sample_answer)
        
        with tab3:
            for i, uc in enumerate(result.use_cases, 1):
                with st.expander(f"**{uc.company}** - {uc.technology}"):
                    st.markdown(f"**Scenario:** {uc.scenario}")
                    st.markdown(f"**Technology:** {uc.technology}")
                    st.markdown(f"**Implementation:** {uc.implementation}")
        
        with tab4:
            st.subheader("Export Interview Prep")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📄 Download as Markdown", use_container_width=True):
                    analyzer.export_to_markdown(result, "/tmp/interview_prep.md")
                    with open("/tmp/interview_prep.md", "r", encoding='utf-8') as f:
                        md_content = f.read()
                    st.download_button(
                        label="💾 Save Markdown File",
                        data=md_content,
                        file_name="interview_prep.md",
                        mime="text/markdown",
                        use_container_width=True
                    )
            
            with col2:
                if st.button("📋 Download as JSON", use_container_width=True):
                    analyzer.export_to_json(result, "/tmp/interview_prep.json")
                    with open("/tmp/interview_prep.json", "r", encoding='utf-8') as f:
                        json_content = f.read()
                    st.download_button(
                        label="💾 Save JSON File",
                        data=json_content,
                        file_name="interview_prep.json",
                        mime="application/json",
                        use_container_width=True
                    )


if __name__ == "__main__":
    main()
