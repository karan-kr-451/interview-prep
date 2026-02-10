"""
YouTube Educational Content Analyzer & Interview Prep Generator

This application analyzes educational YouTube videos and generates:
- Common interview questions based on the content
- Real-world use cases from companies
- Key concepts and technical topics
"""

import os
import re
from youtube_transcript_api import YouTubeTranscriptApi
from anthropic import Anthropic
from google import genai
from google.genai import types
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
import json
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict


@dataclass
class InterviewQuestion:
    """Structure for interview questions"""
    question: str
    difficulty: str  # Easy, Medium, Hard
    topic: str
    sample_answer: str
    companies: List[str]


@dataclass
class UseCase:
    """Structure for real-world use cases"""
    company: str
    scenario: str
    technology: str
    implementation: str


@dataclass
class AnalysisResult:
    """Complete analysis result"""
    video_title: str
    main_topics: List[str]
    key_concepts: List[str]
    interview_questions: List[InterviewQuestion]
    use_cases: List[UseCase]
    summary: str


class YouTubeAnalyzer:
    """Main class for analyzing YouTube educational content"""
    
    def _get_client(self, provider: str, api_key: Optional[str] = None, model: str = "llama3.2"):
        """Get the appropriate client based on provider"""
        if provider == "anthropic":
            if not api_key:
                raise ValueError("API key required for Anthropic")
            return Anthropic(api_key=api_key)
        elif provider == "google":
            if not api_key:
                raise ValueError("API key required for Google")
            return genai.Client(api_key=api_key)
        elif provider == "ollama":
            # Ollama runs locally, no API key needed
            # You can specify base_url if Ollama is running on a different host
            base_url = os.environ.get('OLLAMA_BASE_URL', 'http://localhost:11434')
            return ChatOllama(model=model, base_url=base_url)
        else:
            raise ValueError(f"Unsupported provider: {provider}. Choose from: anthropic, google, ollama")

    def __init__(self, api_key: Optional[str] = None, provider: str = "anthropic", model: Optional[str] = None):
        """
        Initialize with API key and provider
        
        Args:
            api_key: API key for cloud providers (not needed for Ollama)
            provider: One of 'anthropic', 'google', or 'ollama'
            model: Model name (optional, will use defaults if not specified)
        """
        self.provider = provider
        
        # For Ollama, API key is not required
        if provider == "ollama":
            self.api_key = None
            self.model = model or "llama3.2"  # Default Ollama model
        else:
            # For cloud providers, try to get API key
            self.api_key = api_key or os.environ.get('ANTHROPIC_API_KEY') or os.environ.get('GOOGLE_API_KEY')
            
            if not self.api_key:
                raise ValueError(
                    f"API key is required for {provider}. "
                    "Set ANTHROPIC_API_KEY or GOOGLE_API_KEY environment variable, "
                    "or pass api_key parameter."
                )
            self.model = model
            
        self.client = self._get_client(provider, self.api_key, self.model if provider == "ollama" else "llama3.2")
    
    def extract_video_id(self, url: str) -> str:
        """Extract video ID from YouTube URL"""
        patterns = [
            r'(?:youtube\.com\/watch\?v=|youtu\.be\/)([^&\n?]+)',
            r'youtube\.com\/embed\/([^&\n?]+)',
            r'youtube\.com\/shorts\/([^&\n?]+)',
            r'youtube\.com\/v\/([^&\n?]+)',
            r'youtube\.com\/watch\?v=([^&\n?]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        
        raise ValueError("Invalid YouTube URL")
    
    def get_transcript(self, video_url: str) -> str:
        """Get transcript from YouTube video"""
        try:
            ytt_api = YouTubeTranscriptApi()
            video_id = self.extract_video_id(video_url)
            transcript_list = ytt_api.fetch(video_id)
            # print(transcript_list)
            
            # Combine all transcript segments
            full_transcript = " ".join([FetchedTranscriptSnippet.text for FetchedTranscriptSnippet in transcript_list])
            return full_transcript
        
        except Exception as e:
            raise Exception(f"Error fetching transcript: {str(e)}")
    
    def analyze_content(self, transcript: str, video_title: str = "", model: Optional[str] = None, provider: Optional[str] = None) -> AnalysisResult:
        """Analyze transcript and generate interview prep content"""
        
        # Use instance provider if not specified
        provider = provider or self.provider
        
        analysis_prompt = f"""You are an expert technical interviewer and career coach. Analyze this educational content from a YouTube video and create comprehensive interview preparation materials.

Video Title: {video_title}

Transcript:
{transcript[:15000]}  # Limiting to avoid token limits

Please analyze this content and provide:

1. Main Topics: List 3-5 main technical topics covered
2. Key Concepts: List 8-10 key concepts or terms explained
3. Interview Questions: Generate 10 interview questions that interviewers commonly ask about these topics, including:
   - 3 easy questions (fundamental understanding)
   - 4 medium questions (practical application)
   - 3 hard questions (advanced concepts)
   For each question provide: difficulty level, specific topic, sample answer, and 2-3 companies known to ask similar questions

4. Real-World Use Cases: Identify 5 real-world scenarios where companies use these technologies/concepts, including:
   - Company name (real companies like Google, Netflix, Amazon, etc.)
   - Specific scenario/problem
   - Technology/concept applied
   - Brief implementation description

5. Summary: A 2-3 sentence summary of what was covered

Format your response as JSON with this structure:
{{
  "main_topics": ["topic1", "topic2"],
  "key_concepts": ["concept1", "concept2"],
  "interview_questions": [
    {{
      "question": "...",
      "difficulty": "Easy|Medium|Hard",
      "topic": "...",
      "sample_answer": "...",
      "companies": ["Company1", "Company2"]
    }}
  ],
  "use_cases": [
    {{
      "company": "...",
      "scenario": "...",
      "technology": "...",
      "implementation": "..."
    }}
  ],
  "summary": "..."
}}

Provide ONLY the JSON response, no additional text."""

        try:
            # Use the correct client for the requested provider
            if provider != self.provider:
                client = self._get_client(provider, self.api_key, model or "llama3.2")
            else:
                client = self.client

            # Determine the model if not provided
            if not model:
                if provider == "anthropic":
                    model = "claude-3-5-sonnet-20241022"
                elif provider == "google":
                    model = "gemini-2.0-flash-exp"
                elif provider == "ollama":
                    model = self.model or "llama3.2"

            # Make API call based on provider
            if provider == "anthropic":
                response = client.messages.create(
                    model=model,
                    max_tokens=4000,
                    messages=[
                        {"role": "user", "content": analysis_prompt}
                    ]
                )
                response_text = response.content[0].text
            
            elif provider == "google":
                response = client.models.generate_content(
                    model=model,
                    contents=analysis_prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema={
                            "type": "object",
                            "properties": {
                                "main_topics": {"type": "array", "items": {"type": "string"}},
                                "key_concepts": {"type": "array", "items": {"type": "string"}},
                                "interview_questions": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "question": {"type": "string"},
                                            "difficulty": {"type": "string"},
                                            "topic": {"type": "string"},
                                            "sample_answer": {"type": "string"},
                                            "companies": {"type": "array", "items": {"type": "string"}}
                                        },
                                        "required": ["question", "difficulty", "topic", "sample_answer", "companies"]
                                    }
                                },
                                "use_cases": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "company": {"type": "string"},
                                            "scenario": {"type": "string"},
                                            "technology": {"type": "string"},
                                            "implementation": {"type": "string"}
                                        },
                                        "required": ["company", "scenario", "technology", "implementation"]
                                    }
                                },
                                "summary": {"type": "string"}
                            },
                            "required": ["main_topics", "key_concepts", "interview_questions", "use_cases", "summary"]
                        }
                    )
                )
                response_text = response.text
            
            elif provider == "ollama":
                # LangChain Ollama uses ChatOllama with invoke method
                messages = [HumanMessage(content=analysis_prompt)]
                response = client.invoke(messages)
                response_text = response.content
            
            else:
                raise ValueError(f"Unsupported provider: {provider}")
            
            # Try to parse JSON
            # Sometimes models add markdown code blocks, so we need to clean it
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                analysis_data = json.loads(json_str)
            else:
                analysis_data = json.loads(response_text)
            
            # Convert to dataclass structure
            interview_questions = [
                InterviewQuestion(**q) for q in analysis_data.get('interview_questions', [])
            ]
            
            use_cases = [
                UseCase(**u) for u in analysis_data.get('use_cases', [])
            ]
            
            result = AnalysisResult(
                video_title=video_title,
                main_topics=analysis_data.get('main_topics', []),
                key_concepts=analysis_data.get('key_concepts', []),
                interview_questions=interview_questions,
                use_cases=use_cases,
                summary=analysis_data.get('summary', '')
            )
            
            return result
            
        except Exception as e:
            raise Exception(f"Error analyzing content with {provider}: {str(e)}")
    
    def analyze_youtube_video(self, video_url: str, video_title: str = "", model: Optional[str] = None, provider: Optional[str] = None) -> AnalysisResult:
        """Complete workflow: Extract transcript and analyze"""
        provider = provider or self.provider
        
        # Set default models for each provider
        if not model:
            if provider == "anthropic":
                model = "claude-3-5-sonnet-20241022"
            elif provider == "google":
                model = "gemini-2.0-flash-exp"
            elif provider == "ollama":
                model = self.model or "llama3.2"
            
        print(f"📺 Fetching transcript from: {video_url}")
        transcript = self.get_transcript(video_url)
        
        print(f"✅ Transcript retrieved ({len(transcript)} characters)")
        print(f"🤖 Analyzing content with {provider} ({model})...")
        
        result = self.analyze_content(transcript, video_title, model=model, provider=provider)
        
        print("✅ Analysis complete!")
        return result
    
    def export_to_markdown(self, result: AnalysisResult, output_file: str = "interview_prep.md"):
        """Export analysis to a nicely formatted markdown file"""
        
        md_content = f"""# Interview Prep: {result.video_title}

## 📝 Summary
{result.summary}

## 🎯 Main Topics Covered
"""
        for i, topic in enumerate(result.main_topics, 1):
            md_content += f"{i}. {topic}\n"
        
        md_content += "\n## 💡 Key Concepts\n"
        for concept in result.key_concepts:
            md_content += f"- {concept}\n"
        
        md_content += "\n## ❓ Interview Questions\n\n"
        
        # Group by difficulty
        for difficulty in ["Easy", "Medium", "Hard"]:
            questions = [q for q in result.interview_questions if q.difficulty == difficulty]
            if questions:
                md_content += f"### {difficulty} Questions\n\n"
                for i, q in enumerate(questions, 1):
                    md_content += f"**Q{i}. {q.question}**\n\n"
                    md_content += f"*Topic:* {q.topic}\n\n"
                    md_content += f"*Asked by:* {', '.join(q.companies)}\n\n"
                    md_content += f"*Sample Answer:*\n{q.sample_answer}\n\n"
                    md_content += "---\n\n"
        
        md_content += "## 🏢 Real-World Use Cases\n\n"
        for i, uc in enumerate(result.use_cases, 1):
            md_content += f"### {i}. {uc.company}\n\n"
            md_content += f"**Scenario:** {uc.scenario}\n\n"
            md_content += f"**Technology:** {uc.technology}\n\n"
            md_content += f"**Implementation:** {uc.implementation}\n\n"
            md_content += "---\n\n"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        print(f"📄 Interview prep exported to: {output_file}")
    
    def export_to_json(self, result: AnalysisResult, output_file: str = "interview_prep.json"):
        """Export analysis to JSON file"""
        
        # Convert dataclasses to dict
        result_dict = {
            'video_title': result.video_title,
            'main_topics': result.main_topics,
            'key_concepts': result.key_concepts,
            'interview_questions': [asdict(q) for q in result.interview_questions],
            'use_cases': [asdict(uc) for uc in result.use_cases],
            'summary': result.summary
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result_dict, f, indent=2, ensure_ascii=False)
        
        print(f"📄 Interview prep exported to: {output_file}")


def main():
    """Example usage"""
    
    # Example: Analyze a Python tutorial video
    video_url = input("Enter YouTube video URL: ")
    video_title = input("Enter video title (optional): ")
    
    try:
        # Initialize analyzer
        analyzer = YouTubeAnalyzer()
        
        # Analyze video
        result = analyzer.analyze_youtube_video(video_url, video_title)
        
        # Export results
        analyzer.export_to_markdown(result, "interview_prep.md")
        analyzer.export_to_json(result, "interview_prep.json")
        
        print("\n" + "="*50)
        print("📊 ANALYSIS SUMMARY")
        print("="*50)
        print(f"\n{result.summary}\n")
        print(f"📌 Main Topics: {len(result.main_topics)}")
        print(f"💡 Key Concepts: {len(result.key_concepts)}")
        print(f"❓ Interview Questions: {len(result.interview_questions)}")
        print(f"🏢 Use Cases: {len(result.use_cases)}")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")


if __name__ == "__main__":
    main()