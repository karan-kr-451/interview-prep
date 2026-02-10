"""
Example usage of YouTube Interview Prep Generator
This script shows different ways to use the analyzer
"""

from youtube_interview_prep import YouTubeAnalyzer
import os


def example_single_video():
    """Example: Analyze a single video"""
    print("=" * 60)
    print("EXAMPLE 1: Single Video Analysis")
    print("=" * 60)
    
    # Initialize analyzer (make sure ANTHROPIC_API_KEY is set)
    analyzer = YouTubeAnalyzer()
    
    # Example educational video URLs (replace with actual videos)
    video_url = "https://www.youtube.com/watch?v=8ext9G7xspg"  # Example Python tutorial
    video_title = "Python Tutorial for Beginners"
    
    try:
        # Analyze the video
        result = analyzer.analyze_youtube_video(video_url, video_title)
        
        # Display summary
        print(f"\n📊 Analysis Complete!")
        print(f"Summary: {result.summary}")
        print(f"\nGenerated {len(result.interview_questions)} interview questions")
        print(f"Identified {len(result.use_cases)} real-world use cases")
        
        # Export results
        analyzer.export_to_markdown(result, "example_prep.md")
        analyzer.export_to_json(result, "example_prep.json")
        
        print("\n✅ Files exported: example_prep.md and example_prep.json")
        
    except Exception as e:
        print(f"❌ Error: {e}")


def example_batch_processing():
    """Example: Process multiple videos"""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Batch Processing Multiple Videos")
    print("=" * 60)
    
    analyzer = YouTubeAnalyzer()
    
    # List of videos to analyze
    videos = [
        {
            'url': 'https://www.youtube.com/watch?v=EXAMPLE1',
            'title': 'Python Data Structures',
            'output': 'python_ds_prep'
        },
        {
            'url': 'https://www.youtube.com/watch?v=EXAMPLE2',
            'title': 'JavaScript Async Programming',
            'output': 'js_async_prep'
        },
        # Add more videos as needed
    ]
    
    print(f"\nProcessing {len(videos)} videos...")
    
    for i, video in enumerate(videos, 1):
        print(f"\n[{i}/{len(videos)}] Processing: {video['title']}")
        
        try:
            result = analyzer.analyze_youtube_video(video['url'], video['title'])
            analyzer.export_to_markdown(result, f"{video['output']}.md")
            print(f"  ✅ Complete - {len(result.interview_questions)} questions generated")
        
        except Exception as e:
            print(f"  ❌ Failed: {e}")


def example_custom_usage():
    """Example: Access specific parts of the analysis"""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Custom Data Access")
    print("=" * 60)
    
    analyzer = YouTubeAnalyzer()
    
    # Get transcript only
    video_url = "https://www.youtube.com/watch?v=EXAMPLE"
    
    try:
        transcript = analyzer.get_transcript(video_url)
        print(f"\nTranscript length: {len(transcript)} characters")
        print(f"First 200 chars: {transcript[:200]}...")
        
        # Analyze it
        result = analyzer.analyze_content(transcript, "Custom Analysis")
        
        # Access specific data
        print("\n📌 Main Topics:")
        for topic in result.main_topics:
            print(f"  - {topic}")
        
        print("\n❓ Hard Interview Questions:")
        hard_questions = [q for q in result.interview_questions if q.difficulty == "Hard"]
        for q in hard_questions:
            print(f"  - {q.question}")
            print(f"    Companies: {', '.join(q.companies)}")
        
        print("\n🏢 Company Use Cases:")
        for uc in result.use_cases:
            print(f"  - {uc.company}: {uc.technology}")
    
    except Exception as e:
        print(f"❌ Error: {e}")


def example_filter_questions():
    """Example: Filter and work with specific question types"""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Question Filtering")
    print("=" * 60)
    
    analyzer = YouTubeAnalyzer()
    
    video_url = "https://www.youtube.com/watch?v=EXAMPLE"
    
    try:
        result = analyzer.analyze_youtube_video(video_url)
        
        # Group by difficulty
        easy = [q for q in result.interview_questions if q.difficulty == "Easy"]
        medium = [q for q in result.interview_questions if q.difficulty == "Medium"]
        hard = [q for q in result.interview_questions if q.difficulty == "Hard"]
        
        print(f"\nQuestion Distribution:")
        print(f"  Easy: {len(easy)}")
        print(f"  Medium: {len(medium)}")
        print(f"  Hard: {len(hard)}")
        
        # Filter by company
        target_company = "Google"
        google_questions = [
            q for q in result.interview_questions 
            if target_company in q.companies
        ]
        
        print(f"\nQuestions asked by {target_company}: {len(google_questions)}")
        for q in google_questions:
            print(f"  [{q.difficulty}] {q.question}")
    
    except Exception as e:
        print(f"❌ Error: {e}")


def print_menu():
    """Display menu of examples"""
    print("\n" + "=" * 60)
    print("YouTube Interview Prep Generator - Examples")
    print("=" * 60)
    print("\n1. Analyze single video")
    print("2. Batch process multiple videos")
    print("3. Custom data access")
    print("4. Filter questions by criteria")
    print("5. Run all examples")
    print("0. Exit")


def main():
    """Main menu for examples"""
    
    # Check for API key
    if not os.environ.get('ANTHROPIC_API_KEY'):
        print("⚠️  Warning: ANTHROPIC_API_KEY not found in environment variables")
        print("Please set it with: export ANTHROPIC_API_KEY='your-key-here'")
        return
    
    while True:
        print_menu()
        choice = input("\nSelect an example (0-5): ").strip()
        
        if choice == '0':
            print("Goodbye!")
            break
        elif choice == '1':
            example_single_video()
        elif choice == '2':
            example_batch_processing()
        elif choice == '3':
            example_custom_usage()
        elif choice == '4':
            example_filter_questions()
        elif choice == '5':
            example_single_video()
            example_batch_processing()
            example_custom_usage()
            example_filter_questions()
        else:
            print("Invalid choice. Please select 0-5.")
        
        input("\nPress Enter to continue...")


if __name__ == "__main__":
    # You can also run individual examples directly
    # example_single_video()
    
    # Or run the interactive menu
    main()
