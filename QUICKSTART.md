# Quick Start Guide 🚀

Get started with the YouTube Interview Prep Generator in 5 minutes!

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 2: Get Your API Key

1. Go to https://console.anthropic.com/
2. Sign up or log in
3. Navigate to API Keys
4. Create a new key
5. Copy it

## Step 3: Set Your API Key

**On macOS/Linux:**
```bash
export ANTHROPIC_API_KEY='your-api-key-here'
```

**On Windows (Command Prompt):**
```cmd
set ANTHROPIC_API_KEY=your-api-key-here
```

**On Windows (PowerShell):**
```powershell
$env:ANTHROPIC_API_KEY="your-api-key-here"
```

## Step 4: Run the App

### Option A: Web Interface (Easiest!)

```bash
streamlit run app.py
```

Then:
1. Open your browser to http://localhost:8501
2. Paste a YouTube URL (any educational video)
3. Click "Analyze Video"
4. Download your interview prep!

### Option B: Command Line

```bash
python youtube_interview_prep.py
```

Then enter:
- YouTube URL when prompted
- Video title (optional)

Your files will be saved as:
- `interview_prep.md` - Beautiful formatted guide
- `interview_prep.json` - Structured data

## Example Video URLs to Try

Try these educational videos:

**Python:**
- Python tutorials on data structures
- Python interview prep videos
- Django or Flask tutorials

**JavaScript:**
- React or Vue tutorials
- JavaScript concepts videos
- Node.js tutorials

**General CS:**
- Algorithm explanations
- System design videos
- Computer science fundamentals

## What You'll Get

For each video, you get:

✅ **10 Interview Questions**
- 3 Easy questions (fundamentals)
- 4 Medium questions (practical)
- 3 Hard questions (advanced)
- Sample answers for each
- Companies that ask similar questions

✅ **5 Real-World Use Cases**
- How companies use these concepts
- Specific implementations
- Technologies involved

✅ **Key Concepts & Topics**
- Main topics covered
- Important terms and concepts
- Summary of the content

## Sample Output

```markdown
## Interview Questions

### Medium Questions

**Q1. How would you implement a LRU cache in Python?**

*Topic:* Data Structures
*Asked by:* Google, Amazon, Facebook

*Sample Answer:*
An LRU (Least Recently Used) cache can be implemented using...
```

## Troubleshooting

**"No transcript available"**
- Video needs captions enabled
- Try another video with auto-captions

**"API Error"**
- Check your API key is correct
- Verify you have API credits

**"Module not found"**
- Run `pip install -r requirements.txt`

## Next Steps

- Read the full [README.md](README.md) for advanced features
- Check [examples.py](examples.py) for code samples
- Customize the prompts for your needs

## Tips for Best Results

1. **Choose quality videos**: Educational content works best
2. **Longer videos = more content**: 10-30 min videos ideal
3. **Technical topics**: Works great for programming, CS, tech
4. **Videos with captions**: Auto-generated or manual captions needed

## Cost Estimate

- Typical video analysis: $0.05 - $0.15
- Based on transcript length
- Claude Sonnet 4 pricing

---

**Need Help?**

Open an issue or check the README for detailed documentation!

Happy learning! 🎓
