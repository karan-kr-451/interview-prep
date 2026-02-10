"""
Streamlit Web Interface for YouTube Interview Prep Generator
Run with: streamlit run app.py
"""

import streamlit as st
from youtube_interview_prep import YouTubeAnalyzer, AnalysisResult
import os
import subprocess
import json


def get_ollama_models():
    """Get list of locally available Ollama models"""
    try:
        result = subprocess.run(
            ['ollama', 'list'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')[1:]  # Skip header
            models = []
            for line in lines:
                if line.strip():
                    # Extract model name (first column)
                    model_name = line.split()[0]
                    models.append(model_name)
            return models
        return []
    except Exception:
        return []


def get_cloud_models_list():
    """Get list of popular cloud models available on Ollama"""
    # These are popular cloud-based models that can be run via Ollama
    return [
        "kimi-k2:1t-cloud",
        "kimi-k2:4t-cloud", 
        "deepseek-r1:1.5b",
        "deepseek-r1:3b",
        "deepseek-r1:7b",
        "deepseek-r1:8b",
        "deepseek-r1:14b",
        "deepseek-r1:32b",
        "deepseek-r1:70b",
        "deepseek-r1:671b",
        "qwen2.5:0.5b",
        "qwen2.5:1.5b",
        "qwen2.5:3b",
        "qwen2.5:7b",
        "qwen2.5:14b",
        "qwen2.5:32b",
        "qwen2.5:72b",
        "llama3.2:1b",
        "llama3.2:3b",
        "llama3.1:8b",
        "llama3.1:70b",
        "mistral:7b",
        "mixtral:8x7b",
        "gemma2:2b",
        "gemma2:9b",
        "gemma2:27b",
        "phi4:14b",
        "command-r:35b",
        "aya-expanse:8b",
        "aya-expanse:32b",
    ]


def check_ollama_running():
    """Check if Ollama server is running"""
    try:
        result = subprocess.run(
            ['ollama', 'list'],
            capture_output=True,
            timeout=2
        )
        return result.returncode == 0
    except Exception:
        return False


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
            ["anthropic", "google", "ollama"],
            index=0,
            help="Select the AI provider to use"
        )
        
        api_key = None
        selected_model = None
        
        if provider == "anthropic":
            api_key = st.text_input(
                "Anthropic API Key",
                type="password",
                help="Get your API key from https://console.anthropic.com/"
            )
            if not api_key:
                api_key = os.environ.get('ANTHROPIC_API_KEY')
            
            # Model selection for Anthropic
            st.selectbox(
                "Model",
                ["claude-3-5-sonnet-20241022", "claude-3-opus-20240229", "claude-3-haiku-20240307"],
                index=0,
                key="anthropic_model",
                help="Select Claude model"
            )
                
        elif provider == "google":
            api_key = st.text_input(
                "Google API Key",
                type="password",
                help="Get your API key from https://aistudio.google.com/app/api-keys"
            )
            if not api_key:
                api_key = os.environ.get('GOOGLE_API_KEY')
            
            # Model selection for Google
            st.selectbox(
                "Model",
                ["gemini-2.0-flash-exp", "gemini-2.0-pro-exp", "gemini-1.5-pro", "gemini-1.5-flash"],
                index=0,
                key="google_model",
                help="Select Gemini model"
            )
                
        elif provider == "ollama":
            # Check if Ollama is running
            ollama_running = check_ollama_running()
            
            if ollama_running:
                st.success("✅ Ollama is running")
            else:
                st.error("❌ Ollama not detected")
                st.markdown("""
                **To use Ollama:**
                1. Install: https://ollama.com/download
                2. Run: `ollama serve`
                3. Pull a model: `ollama pull llama3.2`
                """)
            
            # Get locally installed models
            local_models = get_ollama_models()
            
            # Model type selection
            model_type = st.radio(
                "Model Type",
                ["Local Models", "Cloud Models"],
                help="Choose between locally installed models or cloud-based models"
            )
            
            if model_type == "Local Models":
                if local_models:
                    st.success(f"✅ {len(local_models)} local models found")
                    selected_model = st.selectbox(
                        "Select Local Model",
                        local_models,
                        help="Choose from your locally installed Ollama models"
                    )
                else:
                    st.warning("⚠️ No local models found")
                    st.markdown("""
                    **Pull a model:**
                    ```bash
                    ollama pull llama3.2
                    ollama pull mistral
                    ```
                    """)
                    # Fallback to manual entry
                    selected_model = st.text_input(
                        "Enter Model Name",
                        value="llama3.2",
                        help="Enter the Ollama model name manually"
                    )
            
            else:  # Cloud Models
                st.info("🌐 Cloud models run via Ollama but may require special setup")
                
                cloud_models = get_cloud_models_list()
                
                # Organize by model family
                model_families = {
                    "Kimi (Cloud)": [m for m in cloud_models if m.startswith("kimi")],
                    "DeepSeek R1 (Reasoning)": [m for m in cloud_models if m.startswith("deepseek-r1")],
                    "Qwen 2.5": [m for m in cloud_models if m.startswith("qwen2.5")],
                    "Llama 3": [m for m in cloud_models if m.startswith("llama3")],
                    "Mistral/Mixtral": [m for m in cloud_models if "mistral" in m or "mixtral" in m],
                    "Gemma 2": [m for m in cloud_models if m.startswith("gemma2")],
                    "Other": [m for m in cloud_models if not any(m.startswith(prefix) for prefix in ["kimi", "deepseek", "qwen", "llama", "mistral", "mixtral", "gemma"])]
                }
                
                # Family selection
                selected_family = st.selectbox(
                    "Model Family",
                    list(model_families.keys()),
                    help="Select the model family"
                )
                
                # Model selection within family
                family_models = model_families[selected_family]
                selected_model = st.selectbox(
                    "Select Model",
                    family_models,
                    help="Choose a specific model variant"
                )
                
                # Show model info
                if "kimi" in selected_model:
                    st.info("💡 Kimi models are cloud-based. Run with: `ollama run " + selected_model + "`")
                elif "deepseek-r1" in selected_model:
                    st.info("💡 DeepSeek R1 is a reasoning model. Great for complex analysis.")
                elif "671b" in selected_model or "70b" in selected_model or "72b" in selected_model:
                    st.warning("⚠️ Large model - requires significant RAM/VRAM")
                
                # Pull instructions
                with st.expander("📥 How to pull this model"):
                    st.code(f"ollama pull {selected_model}", language="bash")
                    st.markdown(f"""
                    Or run directly:
                    ```bash
                    ollama run {selected_model}
                    ```
                    """)
            
            # Custom Ollama URL
            with st.expander("🔧 Advanced Settings"):
                custom_url = st.text_input(
                    "Custom Ollama URL",
                    value="http://localhost:11434",
                    help="Change if Ollama is running on a different host/port"
                )
                if custom_url != "http://localhost:11434":
                    os.environ['OLLAMA_BASE_URL'] = custom_url
        
        st.markdown("---")
        st.markdown("### 📖 How to Use")
        
        if provider == "ollama":
            st.markdown("""
            1. Ensure Ollama is running
            2. Select a model (local or cloud)
            3. Paste a YouTube video URL
            4. Click 'Analyze Video'
            5. Download your interview prep
            
            **Benefits:**
            - ✅ 100% FREE
            - 🔒 Fully private
            - 🚀 No API limits
            """)
        else:
            st.markdown("""
            1. Enter your API key
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
        # Validation based on provider
        if provider in ["anthropic", "google"] and not api_key:
            st.error(f"⚠️ Please provide a {provider.capitalize()} API key in the sidebar!")
            return
        
        if provider == "ollama" and not selected_model:
            st.error("⚠️ Please select an Ollama model in the sidebar!")
            return
        
        if not video_url:
            st.error("⚠️ Please provide a YouTube video URL!")
            return
        
        try:
            # Initialize analyzer based on provider
            if provider == "ollama":
                with st.spinner("🦙 Initializing Ollama..."):
                    analyzer = YouTubeAnalyzer(
                        provider="ollama",
                        model=selected_model
                    )
                st.info(f"Using Ollama model: **{selected_model}**")
            else:
                analyzer = YouTubeAnalyzer(
                    api_key=api_key,
                    provider=provider
                )
            
            # Fetch transcript
            with st.spinner("🎥 Fetching video transcript..."):
                transcript = analyzer.get_transcript(video_url)
                st.success(f"✅ Transcript retrieved ({len(transcript)} characters)")
            
            # Analyze content
            analysis_time_msg = "🤖 Analyzing content with AI..."
            if provider == "ollama":
                analysis_time_msg += f" Using {selected_model}. This may take a few minutes..."
            else:
                analysis_time_msg += " This may take 30-60 seconds..."
            
            with st.spinner(analysis_time_msg):
                # Get the selected model from session state if available
                model_to_use = None
                if provider == "anthropic" and "anthropic_model" in st.session_state:
                    model_to_use = st.session_state.anthropic_model
                elif provider == "google" and "google_model" in st.session_state:
                    model_to_use = st.session_state.google_model
                elif provider == "ollama":
                    model_to_use = selected_model
                
                result = analyzer.analyze_content(
                    transcript, 
                    video_title,
                    model=model_to_use,
                    provider=provider
                )
            
            st.success("✅ Analysis complete!")
            
            # Store result in session state
            st.session_state['result'] = result
            st.session_state['analyzer'] = analyzer
            st.session_state['provider'] = provider
            st.session_state['model'] = model_to_use or "default"
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            
            # Provide helpful error messages for Ollama
            if provider == "ollama":
                st.markdown("""
                **Troubleshooting Ollama:**
                1. Is Ollama running? Run `ollama serve` in terminal
                2. Is the model pulled? Run `ollama pull {model}`
                3. Check model name is correct with `ollama list`
                """.format(model=selected_model))
            return
    
    # Display results if available
    if 'result' in st.session_state:
        result = st.session_state['result']
        analyzer = st.session_state['analyzer']
        
        # Show analysis info
        if 'provider' in st.session_state:
            provider_name = st.session_state['provider'].capitalize()
            model_name = st.session_state.get('model', 'default')
            
            info_cols = st.columns([3, 1])
            with info_cols[0]:
                st.markdown("## 📝 Summary")
            with info_cols[1]:
                if st.session_state['provider'] == 'ollama':
                    st.success(f"🦙 {provider_name}")
                else:
                    st.info(f"🤖 {provider_name}")
                st.caption(f"Model: {model_name}")
        else:
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