import "dotenv/config";
import express from "express";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { z } from "zod";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { ChatPromptTemplate, MessagesPlaceholder } from "@langchain/core/prompts";
import { AIMessage, HumanMessage } from "@langchain/core/messages";

import { clearSemanticStoreCache, retrievePolicyContext } from "./semantic.js";

const app = express();
app.use(express.json());

const PORT = Number(process.env.PORT || 3000);
const MAX_MEMORY_TURNS = 6;
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// In-memory stores. Use DB/Redis in production.
const chatMemoryBySession = new Map();

const chatSchema = z.object({
  message: z.string().min(1),
  sessionId: z.string().min(1).optional()
});

function getSessionId(req) {
  return (
    req.headers["x-session-id"] ||
    req.body?.sessionId ||
    req.query?.sessionId ||
    "default-session"
  );
}

function getMemory(sessionId) {
  return chatMemoryBySession.get(sessionId) || [];
}

function pushMemory(sessionId, role, content) {
  const mem = getMemory(sessionId);
  mem.push({ role, content });
  chatMemoryBySession.set(sessionId, mem.slice(-MAX_MEMORY_TURNS));
}

function toLangChainHistory(memory) {
  return memory.map((m) =>
    m.role === "user" ? new HumanMessage(m.content) : new AIMessage(m.content)
  );
}

app.use(express.static(path.resolve(__dirname, "../")));

app.get("/api/health", (_req, res) => {
  res.json({ ok: true, service: "vet-clinic-chatbot-node" });
});

app.post("/api/chat", async (req, res) => {
  if (!process.env.GEMINI_API_KEY) {
    return res.status(500).json({
      error: "GEMINI_API_KEY is missing. Set it in environment variables."
    });
  }

  const parsed = chatSchema.safeParse(req.body);
  if (!parsed.success) {
    return res.status(400).json({ error: "Invalid payload.", issues: parsed.error.issues });
  }

  const { message } = parsed.data;
  const sessionId = String(getSessionId(req));

  try {
    const llm = new ChatGoogleGenerativeAI({
      model: "gemini-1.5-flash",
      temperature: 0.2
    });

    const semanticContext = await retrievePolicyContext(message, 4);
    const memory = getMemory(sessionId);
    const history = toLangChainHistory(memory);

    const prompt = ChatPromptTemplate.fromMessages([
      [
        "system",
        "You are a veterinary clinic booking chatbot. Use ONLY the source-of-truth context provided. "
          + "If a fact is not in context, say you cannot infer it. "
          + "Never provide routine consultation or emergency-care advice. "
          + "Return STRICT JSON only (no markdown, no prose outside JSON) with this exact shape: "
          + "{\"answer\":\"string\",\"decision\":\"confirmed|rejected|needs_more_info|informational\","
          + "\"reason\":\"string|null\",\"actions\":[\"string\"],\"used_memory\":true|false}."
      ],
      ["system", "Policy context:\n{semanticContext}"],
      new MessagesPlaceholder("history"),
      ["human", "{question}"]
    ]);

    const chain = prompt.pipe(llm);
    const response = await chain.invoke({
      semanticContext,
      history,
      question: message
    });

    const rawReply =
      typeof response.content === "string" ? response.content : JSON.stringify(response.content);
    const cleaned = rawReply
      .replace(/^```json\s*/i, "")
      .replace(/^```\s*/i, "")
      .replace(/\s*```$/, "")
      .trim();
    let replyObject;
    try {
      replyObject = JSON.parse(cleaned);
    } catch {
      replyObject = {
        answer: cleaned || "I cannot infer that from the provided source document.",
        decision: "informational",
        reason: null,
        actions: [],
        used_memory: history.length > 0
      };
    }

    const reply = JSON.stringify(replyObject);
    pushMemory(sessionId, "user", message);
    pushMemory(sessionId, "assistant", reply);

    return res.json({
      reply: replyObject,
      sessionId,
      memoryTurns: getMemory(sessionId).length
    });
  } catch (err) {
    return res.status(500).json({ error: "Chat failed.", detail: err.message });
  }
});

app.post("/api/memory/clear", (req, res) => {
  const sessionId = String(getSessionId(req));
  chatMemoryBySession.delete(sessionId);
  res.json({ status: "ok", sessionId, message: "Short-term memory cleared." });
});

app.post("/api/source/refresh", (_req, res) => {
  clearSemanticStoreCache();
  res.json({
    status: "ok",
    message: "Semantic source cache cleared. The next chat request will re-index the source document."
  });
});

app.get("/", (_req, res) => {
  res.sendFile(path.resolve(__dirname, "../index.html"));
});

app.use((_req, res) => {
  res.status(404).json({ error: "Not found" });
});

app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
});
