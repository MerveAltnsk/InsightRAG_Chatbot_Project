# 🚀 Hugging Face Spaces Deployment Guide

## Quick Deployment Steps

### 1. Create a New Space

- Go to [Hugging Face Spaces](https://huggingface.co/spaces)
- Click "Create new Space"
- Choose "Streamlit" as the SDK
- Set visibility (Public/Private)

### 2. Upload Files

Upload these files to your Space:

- `app.py` (main Streamlit application)
- `requirements.txt` (dependencies)
- `README.md` (documentation)

### 3. Set Environment Variables

- Go to Settings → Secrets
- Add `GOOGLE_API_KEY` with your Gemini API key
- The app will automatically use this environment variable

### 4. Deploy

- Push your code to the Space
- The app will automatically build and deploy
- Wait for the build to complete (usually 2-3 minutes)

### 5. Test Your App

- Open your Space URL
- Enter your Gemini API key in the sidebar
- Click "Initialize RAG System"
- Start chatting!

## Important Notes

- **API Key**: Make sure to set `GOOGLE_API_KEY` in Space secrets
- **Memory**: The app will create a Chroma database in memory
- **Performance**: First initialization may take a few minutes
- **Limits**: Hugging Face Spaces have resource limits

## Troubleshooting

### Build Fails

- Check `requirements.txt` for correct package versions
- Ensure all imports are available

### Runtime Errors

- Verify API key is set correctly
- Check logs in the Space interface
- Ensure all dependencies are installed

### Performance Issues

- Reduce the number of documents processed
- Use smaller embedding models
- Optimize the RAG pipeline

## Customization

You can customize the app by:

- Modifying the UI in `app.py`
- Changing the embedding model
- Adjusting the RAG pipeline parameters
- Adding new features

Happy deploying! 🎉
