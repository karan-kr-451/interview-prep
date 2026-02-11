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
    implementation: str
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
            return ChatOllama(model=model,validate_model_on_init=True, reasoning=True, base_url=base_url)
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
    
    def _clean_and_parse_json(self, response_text: str) -> Dict:
        """
        Robustly clean and parse JSON from model response.
        Handles markdown blocks, control characters, unescaped quotes in code,
        and other common LLM JSON errors.
        """
        from json_repair import repair_json
        
        # 1. Extract JSON block if present
        json_match = re.search(r"```json\s*(.*?)```", response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1).strip()
        else:
            # Fallback: find first { and last }
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                json_str = response_text

        # Remove trailing commas before closing braces/brackets
        json_str = re.sub(r',\s*([\]}])', r'\1', json_str)
        
        # Attempt 1: Try strict=False (handles raw newlines/tabs in strings)
        try:
            return json.loads(json_str, strict=False)
        except json.JSONDecodeError:
            pass
        
        # Attempt 2: Use json_repair to fix malformed JSON
        # This handles unescaped quotes, missing commas, trailing commas,
        # and other issues common in LLM-generated JSON with code
        try:
            repaired = repair_json(json_str, return_objects=True)
            if isinstance(repaired, dict):
                return repaired
        except Exception:
            pass
        
        # Attempt 3: Last resort — raise with helpful error
        raise ValueError(f"Failed to parse JSON response after all repair attempts.\nSnippet: {json_str[:300]}...")

    def _chunk_transcript(self, transcript: str, max_chunk_size: int = 100000) -> List[str]:
        """
        Intelligently chunk very long transcripts for processing.
        Tries to split at sentence boundaries to preserve context.
        """
        if len(transcript) <= max_chunk_size:
            return [transcript]
        
        chunks = []
        current_chunk = ""
        
        # Split by sentences (rough approximation)
        sentences = transcript.replace('. ', '.|').replace('? ', '?|').replace('! ', '!|').split('|')
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= max_chunk_size:
                current_chunk += sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _merge_analysis_results(self, results: List[AnalysisResult], video_title: str) -> AnalysisResult:
        """Merge multiple analysis results from chunked processing"""
        all_topics = []
        all_concepts = []
        all_questions = []
        all_use_cases = []
        summaries = []
        
        for result in results:
            all_topics.extend(result.main_topics)
            all_concepts.extend(result.key_concepts)
            all_questions.extend(result.interview_questions)
            all_use_cases.extend(result.use_cases)
            summaries.append(result.summary)
        
        # Deduplicate topics and concepts while preserving order
        unique_topics = []
        seen_topics = set()
        for topic in all_topics:
            topic_lower = topic.lower()
            if topic_lower not in seen_topics:
                unique_topics.append(topic)
                seen_topics.add(topic_lower)
        
        unique_concepts = []
        seen_concepts = set()
        for concept in all_concepts:
            concept_lower = concept.lower()
            if concept_lower not in seen_concepts:
                unique_concepts.append(concept)
                seen_concepts.add(concept_lower)
        
        # Combine summaries
        combined_summary = " ".join(summaries)
        
        return AnalysisResult(
            video_title=video_title,
            main_topics=unique_topics,
            key_concepts=unique_concepts,
            interview_questions=all_questions,
            use_cases=all_use_cases,
            summary=combined_summary
        )
    
    def analyze_content(self, transcript: str, video_title: str = "", model: Optional[str] = None, provider: Optional[str] = None, chunk_size: Optional[int] = None) -> AnalysisResult:
        """
        Analyze transcript and generate interview prep content.
        
        For very long videos (30+ hours), automatically chunks the content and processes
        in segments, then combines the results.
        
        Args:
            transcript: Full video transcript
            video_title: Title of the video
            model: Model to use for analysis
            provider: Provider to use (anthropic, google, ollama)
            chunk_size: Max characters per chunk (default: 200000 for very long videos)
        """
        
        # Use instance provider if not specified
        provider = provider or self.provider
        
        # Determine chunk size based on provider capabilities
        if chunk_size is None:
            if provider == "anthropic":
                chunk_size = 200000  # Claude has large context window
            elif provider == "google":
                chunk_size = 300000  # Gemini has very large context
            elif provider == "ollama":
                chunk_size = 50000   # Local models typically have smaller context
        
        transcript_length = len(transcript)
        
        # For extremely long videos, process in chunks
        if transcript_length > chunk_size:
            print(f"⚠️  Large video detected ({transcript_length} chars)")
            print(f"📦 Processing in chunks of {chunk_size} characters...")
            
            chunks = self._chunk_transcript(transcript, chunk_size)
            print(f"📊 Split into {len(chunks)} chunks")
            
            chunk_results = []
            for i, chunk in enumerate(chunks, 1):
                print(f"🔄 Processing chunk {i}/{len(chunks)}...")
                chunk_result = self._analyze_single_chunk(
                    chunk, 
                    f"{video_title} (Part {i}/{len(chunks)})",
                    model,
                    provider
                )
                chunk_results.append(chunk_result)
            
            print("🔗 Merging results from all chunks...")
            final_result = self._merge_analysis_results(chunk_results, video_title)
            print(f"✅ Combined analysis: {len(final_result.interview_questions)} total questions")
            return final_result
        
        # For normal-sized videos, process directly
        return self._analyze_single_chunk(transcript, video_title, model, provider)
    
    def _analyze_single_chunk(self, transcript: str, video_title: str, model: Optional[str] = None, provider: Optional[str] = None) -> AnalysisResult:
        """Analyze a single chunk of transcript"""
        
        # Calculate dynamic question counts based on transcript length
        # For every 5000 chars, add 2-3 questions (scales with content)
        transcript_length = len(transcript)
        base_questions = 10
        additional_questions = (transcript_length // 5000) * 2
        total_questions = base_questions + additional_questions  # No cap! Scale with content
        
        # Dynamic distribution: ~30% easy, ~40% medium, ~30% hard
        easy_count = max(3, int(total_questions * 0.3))
        hard_count = max(3, int(total_questions * 0.3))
        medium_count = total_questions - easy_count - hard_count
        
        # Dynamic use cases: 1 per 10 minutes of content (estimate: 150 words/min)
        estimated_duration_minutes = transcript_length / (150 * 5)  # Rough estimate
        use_case_count = max(5, int(estimated_duration_minutes / 10))
        # No cap on use cases! Scale with content
        
        # Dynamic topics and concepts based on content length - NO LIMITS
        topic_count = max(5, transcript_length // 10000)  # More content = more topics
        concept_count = max(10, transcript_length // 5000)  # More content = more concepts
        
        analysis_prompt = f"""You are an expert technical interviewer and career coach. Analyze this educational content from a YouTube video and create comprehensive interview preparation materials.

Video Title: {video_title}
Content Length: {transcript_length} characters (~{estimated_duration_minutes:.0f} minutes estimated)

Full Transcript:
{transcript}

Based on the ENTIRE content above, please provide a comprehensive analysis:

1. Main Topics: List {topic_count} main technical topics covered (scale based on content depth)
2. Key Concepts: List {concept_count} key concepts or terms explained throughout the content
3. Interview Questions: Generate {total_questions} interview questions that interviewers commonly ask about these topics:
   - {easy_count} EASY questions (fundamental understanding, definitions, basic concepts)
   - {medium_count} MEDIUM questions (practical application, how-to, implementation)
   - {hard_count} HARD questions (advanced concepts, system design, optimization, trade-offs)
   
   For EACH question provide:
   - The question itself
   - Difficulty level (Easy/Medium/Hard)
   - Specific topic it relates to
   - Comprehensive detailed answer (2-4 paragraphs)
   - 2-3 companies known to ask similar questions in interviews
   - Implementation: working code example that demonstrates the answer

4. Real-World Use Cases: Identify {use_case_count} real-world scenarios where companies use these technologies/concepts:
   - Company name (use real companies like Google, Netflix, Amazon, Meta, Microsoft, MNCs, Startups etc.)
   - Specific scenario/problem they solved
   - Technology/concept applied from the video
   - Brief implementation description (how they used it)

5. Summary: A comprehensive paragraph summary covering the main themes and key takeaways

IMPORTANT: 
- Generate questions based on ALL topics covered, not just the first few
- Ensure questions span the entire content of the video
- Don't limit yourself - if the content covers more topics, generate more questions
- Make questions specific and practical, based on actual content discussed
- Ensure use cases are relevant to the specific technologies/concepts covered
- Provide working code examples in the implementation field for each question

CRITICAL JSON FORMATTING RULES:
- In the implementation field, write code as a single line using \\n for newlines between code lines
- Escape all double quotes inside strings with backslash (\\")
- Example implementation value: "def example():\\n    x = 10\\n    return x"
- The response MUST be valid JSON. Test it mentally before outputting.


Format your response as JSON with this structure:
{{
  "main_topics": ["topic1", "topic2", ...],
  "key_concepts": ["concept1", "concept2", ...],
  "interview_questions": [
    {{
      "question": "specific question text",
      "difficulty": "Easy|Medium|Hard",
      "topic": "specific topic name",
      "sample_answer": "comprehensive answer",
      "companies": ["Company1", "Company2", "Company3"],
      "implementation": "def example():\\n    x = 10\\n    print(x)\\n    return x"
    }},
    ... (all {total_questions} questions)
  ],
  "use_cases": [
    {{
      "company": "Company Name",
      "scenario": "specific problem/scenario",
      "technology": "specific tech from video",
      "implementation": "how they implemented it"
    }},
    ... (all {use_case_count} use cases)
  ],
  "summary": "comprehensive summary of content Point to Point and Point to Interview and Point to Job and Point-wise summary"
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
            # Calculate dynamic max_tokens based on expected output size
            # Rough estimate: each question ~150 tokens, each use case ~100 tokens
            estimated_output_tokens = (total_questions * 150) + (use_case_count * 100) + 1000
            max_output_tokens = min(estimated_output_tokens, 16000)  # Cap at model limits
            
            if provider == "anthropic":
                response = client.messages.create(
                    model=model,
                    max_tokens=max_output_tokens,
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
            
            # Clean and parse JSON
            analysis_data = self._clean_and_parse_json(response_text)
            
            # Convert to dataclass structure
            # Filter to only known fields to handle models returning extra keys
            question_fields = {'question', 'difficulty', 'topic', 'sample_answer', 'implementation', 'companies'}
            use_case_fields = {'company', 'scenario', 'technology', 'implementation'}
            
            interview_questions = []
            for q in analysis_data.get('interview_questions', []):
                # Ensure implementation field exists even if model missed it
                if 'implementation' not in q:
                    q['implementation'] = "Not provided"
                # Only keep known fields
                filtered_q = {k: v for k, v in q.items() if k in question_fields}
                interview_questions.append(InterviewQuestion(**filtered_q))
            
            use_cases = []
            for u in analysis_data.get('use_cases', []):
                filtered_u = {k: v for k, v in u.items() if k in use_case_fields}
                use_cases.append(UseCase(**filtered_u))
            
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
        
        # Calculate estimated video duration
        transcript_length = len(transcript)
        estimated_duration_minutes = transcript_length / (150 * 5)  # Rough estimate
        estimated_hours = estimated_duration_minutes / 60
        
        print(f"✅ Transcript retrieved: {len(transcript):,} characters")
        print(f"⏱️  Estimated video duration: ~{estimated_duration_minutes:.0f} minutes ({estimated_hours:.1f} hours)")
        
        if estimated_hours > 10:
            print(f"🎬 Long video detected! This will be processed in chunks.")
            print(f"⏳ Processing time estimate: {estimated_hours * 2:.0f}-{estimated_hours * 3:.0f} minutes with {provider}")
        
        print(f"🤖 Analyzing content with {provider} ({model})...")
        
        result = self.analyze_content(transcript, video_title, model=model, provider=provider)
        
        print("✅ Analysis complete!")
        print(f"📊 Generated: {len(result.interview_questions)} questions, {len(result.use_cases)} use cases")
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
                    if q.implementation and q.implementation != "Not provided":
                        md_content += f"*Implementation:*\n```python\n{q.implementation}\n```\n\n"
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