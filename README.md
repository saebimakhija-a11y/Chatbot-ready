# Veterinary Clinic Node/Express API (Gemini + LangChain)

This project implements a Node/Express JSON API chatbot and booking engine for a veterinary clinic sterilisation flow, using your provided document as the only context source.

## Features

- Express JSON API backend
- Booking decision engine with:
  - Monday-Thursday operating-day validation
  - Daily 240-minute quota
  - Dog cap of max 2/day
  - Species/sex/weight-based service durations
  - Dog-in-heat rejection rule
  - Blood-test-required rule for age > 6
  - Clear confirmation/rejection payloads
- Short-term memory (session-based recent chat turns)
- Semantic retrieval using embeddings over source-of-truth document context
- LangChain prompt chain with Gemini model
- Vercel-ready configuration (`vercel.json`)

## Setup

1. Install dependencies:

```bash
npm install
```

2. Configure environment variables:

```bash
copy .env.example .env
```

Then edit `.env` and set your real `GEMINI_API_KEY`.

Optional:
- `SOURCE_DOC_PATH` lets you point to a custom source document file.
- Default is `./data/source-of-truth.txt`.

## Run

```bash
npm run dev
```

Server starts at `http://localhost:3000` by default.

## API Endpoints

- `GET /api/health` - Health check
- `POST /api/book` - Validate and create booking
- `POST /api/chat` - Ask policy/chat questions (with short-term memory)
- `POST /api/memory/clear` - Clear short-term memory
- `POST /api/source/refresh` - Clear semantic cache after updating source document

## Important Notes

- Current storage is in-memory only; bookings and memory reset on server restart.
- For per-user memory, provide `sessionId` in request body or `x-session-id` header.
- Client chooses day only (not surgery time), per policy.
- Source-of-truth defaults to `data/source-of-truth.txt`; if missing, embedded fallback text is used.
