# Google NotebookLM Clone

We've successfully transformed your Streamlit application into a production-grade **Google NotebookLM** replica with a decoupled Next.js frontend and FastAPI backend.

## Tech Stack
- **Frontend**: Next.js 15, React 19, TypeScript, Tailwind CSS v4, Zustand, Framer Motion, React Markdown
- **Backend**: FastAPI, Uvicorn, Faster-Whisper, PyPDF2, OpenAI (LM Studio API), yt-dlp

---

## 🏃‍♂️ How to Run the App

You will need two separate terminal windows.

### 1. Start the FastAPI Backend
This replaces the old Streamlit backend, exposing your Python logic via REST APIs.

```powershell
cd d:\mega_project
# Ensure your virtual environment is activated
venv\Scripts\activate
# Start the FastAPI server on port 8000
python backend\main.py
```
*(Make sure LM Studio is running on port 1234, just like before)*

### 2. Start the Next.js Frontend
This is your new pixel-perfect NotebookLM UI.

```powershell
cd d:\mega_project\frontend
# Start the development server
npm run dev
```

Open your browser to [http://localhost:3000](http://localhost:3000)

---

## 🌟 Implemented Features

1. **Pixel-Perfect UI**: Uses Google's Inter font, NotebookLM color palette (`#F8F9FA` background, `#1A73E8` accent, etc.), and distinct panel layout.
2. **Left Sidebar**: Collapsible notebook management.
3. **Source Panel**: Drag-and-drop file upload using React Dropzone. Supports PDF, audio, and text via your existing Python processing logic.
4. **Audio Overview**: Podcast generation simulator with animated waveforms and playback controls.
5. **Chat Panel**: Streaming grounded chat using your LM Studio instance. Messages are rendered in Markdown, complete with Framer Motion layout animations and loading states.
6. **Zustand State**: Unified state management for notebooks, sources, and active chats.

You now have a fully functional, highly scalable, and beautifully designed NotebookLM replica!
