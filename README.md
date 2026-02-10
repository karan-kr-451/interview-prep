# YouTube Interview Prep Generator 🎓

An intelligent Python application that analyzes educational YouTube videos and automatically generates comprehensive interview preparation materials, including:

- ❓ Common interview questions (Easy, Medium, Hard)
- 🏢 Real-world use cases from major tech companies
- 📚 Key concepts and technical topics
- 💡 Sample answers and company-specific insights

## Features

✨ **Automatic Transcript Extraction**: Fetches video transcripts directly from YouTube
🤖 **AI-Powered Analysis**: Uses Claude AI to intelligently analyze educational content
📊 **Structured Output**: Generates organized interview prep in Markdown and JSON formats
🎯 **Difficulty Levels**: Questions categorized by Easy, Medium, and Hard
🏢 **Company Insights**: Includes which companies ask similar questions
💼 **Real-World Use Cases**: Shows how top companies use these technologies
🖥️ **Web Interface**: Easy-to-use Streamlit web application

## Installation

### Prerequisites

- Python 3.8 or higher
- Anthropic API key ([Get one here](https://console.anthropic.com/))

### Setup

1. **Clone or download this repository**

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Set up your API key:**

Option A - Environment variable (recommended):
```bash
export ANTHROPIC_API_KEY='your-api-key-here'
```

Option B - Pass directly to the script (see usage examples)

## Usage

### Method 1: Command Line Interface

Run the main script:
```bash
python youtube_interview_prep.py
```

You'll be prompted to enter:
- YouTube video URL
- Video title (optional)

The script will generate:
- `interview_prep.md` - Formatted markdown file
- `interview_prep.json` - Structured JSON data

### Method 2: Web Interface (Recommended)

Launch the Streamlit web app:
```bash
streamlit run app.py
```

This opens a user-friendly web interface where you can:
- Enter YouTube URLs
- View results interactively
- Filter questions by difficulty
- Download prep materials

### Method 3: Python Script Integration

```python
from youtube_interview_prep import YouTubeAnalyzer

# Initialize analyzer
analyzer = YouTubeAnalyzer(api_key='your-api-key')

# Analyze a video
result = analyzer.analyze_youtube_video(
    video_url='https://www.youtube.com/watch?v=...',
    video_title='Python Tutorial'
)

# Access the results
print(result.summary)
print(f"Questions: {len(result.interview_questions)}")
print(f"Use Cases: {len(result.use_cases)}")

# Export to files
analyzer.export_to_markdown(result, 'my_prep.md')
analyzer.export_to_json(result, 'my_prep.json')
```

## Example Output

### Generated Interview Questions

**Easy Question:**
> What is a Python dictionary and when would you use it?

*Asked by: Google, Amazon, Microsoft*

**Medium Question:**
> How would you implement a caching mechanism using Python decorators?

*Asked by: Netflix, Facebook, Uber*

**Hard Question:**
> Design a distributed rate limiter using Python. How would you handle race conditions?

*Asked by: Google, Amazon, Twitter*

### Real-World Use Cases

**Netflix - Content Recommendation**
- **Technology:** Python with collaborative filtering
- **Implementation:** Uses pandas and NumPy to analyze viewing patterns and generate personalized recommendations for 200M+ users

**Google - Search Indexing**
- **Technology:** Python with distributed computing
- **Implementation:** Processes billions of web pages using MapReduce-style operations

## Output Format

### Markdown File Structure
```
# Interview Prep: [Video Title]

## Summary
[AI-generated summary]

## Main Topics Covered
1. Topic 1
2. Topic 2

## Key Concepts
- Concept 1
- Concept 2

## Interview Questions
### Easy Questions
[Questions with sample answers]

### Medium Questions
[Questions with sample answers]

### Hard Questions
[Questions with sample answers]

## Real-World Use Cases
[Company scenarios and implementations]
```

### JSON File Structure
```json
{
  "video_title": "...",
  "main_topics": [...],
  "key_concepts": [...],
  "interview_questions": [
    {
      "question": "...",
      "difficulty": "Easy|Medium|Hard",
      "topic": "...",
      "sample_answer": "...",
      "companies": ["Company1", "Company2"]
    }
  ],
  "use_cases": [
    {
      "company": "...",
      "scenario": "...",
      "technology": "...",
      "implementation": "..."
    }
  ],
  "summary": "..."
}
```

## Use Cases

Perfect for:
- 🎓 **Students**: Preparing for technical interviews
- 💼 **Job Seekers**: Converting learning into interview-ready knowledge
- 👨‍💻 **Developers**: Quick prep before interviews
- 📚 **Educators**: Creating study materials from video content
- 🏢 **Bootcamps**: Generating practice questions from course videos

## Supported Video Types

Works best with:
- Programming tutorials (Python, JavaScript, Java, etc.)
- Data structures and algorithms courses
- System design videos
- Technology deep-dives
- Computer science concepts
- Framework tutorials (React, Django, etc.)

## Limitations

- Requires videos to have transcripts/captions available
- Best results with educational content (not vlogs or general videos)
- Analysis quality depends on transcript quality
- Limited to ~15,000 characters of transcript for analysis

## API Costs

This tool uses Claude AI (Sonnet 4):
- Typical cost per video: $0.05 - $0.15
- Depends on transcript length and analysis depth
- Check current pricing: https://www.anthropic.com/pricing

## Troubleshooting

**"No transcript available"**
- Video doesn't have captions enabled
- Try a different video with auto-generated or manual captions

**"Invalid YouTube URL"**
- Use the full URL: `https://www.youtube.com/watch?v=VIDEO_ID`
- Or short format: `https://youtu.be/VIDEO_ID`

**"API key error"**
- Verify your Anthropic API key is correct
- Check if it's properly set in environment variables

## Advanced Usage

### Batch Processing Multiple Videos

```python
from youtube_interview_prep import YouTubeAnalyzer

analyzer = YouTubeAnalyzer()

videos = [
    ('https://youtube.com/watch?v=1', 'Python Basics'),
    ('https://youtube.com/watch?v=2', 'Data Structures'),
    ('https://youtube.com/watch?v=3', 'Algorithms'),
]

for url, title in videos:
    result = analyzer.analyze_youtube_video(url, title)
    analyzer.export_to_markdown(result, f"{title.replace(' ', '_')}.md")
```

### Custom Analysis Prompts

You can modify the `analyze_content()` method to customize:
- Number of questions generated
- Difficulty distribution
- Focus areas (e.g., more use cases, fewer questions)
- Company preferences

## Contributing

Feel free to submit issues or pull requests for:
- Bug fixes
- Feature enhancements
- Documentation improvements
- Additional export formats

## License

MIT License - feel free to use for personal or commercial projects

## Acknowledgments

- Built with [Anthropic Claude AI](https://www.anthropic.com/)
- YouTube transcript extraction via [youtube-transcript-api](https://github.com/jdepoix/youtube-transcript-api)
- Web interface powered by [Streamlit](https://streamlit.io/)

---

**Happy Interview Prep! 🚀**

Questions? Open an issue or reach out!
