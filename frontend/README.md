--WHAT# RAG System Frontend

Simple Next.js chat interface for the RAG system.

## Quick Start

```bash
# Install dependencies
npm install

# Development (with local API)
npm run dev

# Production build
npm run build
npm start
```

## Configuration

Create `.env.local`:
```
API_URL=http://localhost:8000
```

For production:
```
API_URL=https://your-api.onrender.com
```

## Deploy to Vercel (FREE)

```bash
# Install Vercel CLI
npm i -g vercel

# Deploy
vercel

# Set environment variable
vercel env add API_URL
# Enter: https://your-api.onrender.com

# Deploy production
vercel --prod
```

Done! Your frontend will be live at `https://your-app.vercel.app`

## Features

- Real-time chat interface
- Message history
- Loading states
- Error handling
- Responsive design
- Dark theme

## Usage

1. Start your backend API (local or deployed)
2. Update `API_URL` in `.env.local`
3. Run `npm run dev`
4. Open http://localhost:3000
5. Start asking questions!
