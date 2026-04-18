import { Document } from "@langchain/core/documents";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

let cachedStore = null;
let cachedSourceText = "";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const DEFAULT_SOURCE_PATH = path.resolve(__dirname, "../data/source-of-truth.txt");

async function loadSourceOfTruthText() {
  const configuredPath = process.env.SOURCE_DOC_PATH
    ? path.resolve(process.cwd(), process.env.SOURCE_DOC_PATH)
    : DEFAULT_SOURCE_PATH;

  const content = await fs.readFile(configuredPath, "utf8");
  const normalized = content.trim();
  if (!normalized) {
    throw new Error(
      `Source-of-truth document is empty at: ${configuredPath}. Add content before using chat.`
    );
  }
  return normalized;
}

function buildPolicyDocs(sourceText) {
  return sourceText.split("\n\n")
    .map((chunk) => chunk.trim())
    .filter(Boolean)
    .map((chunk, idx) => new Document({ pageContent: chunk, metadata: { chunk: idx + 1 } }));
}

export async function getSemanticStore() {
  const sourceText = await loadSourceOfTruthText();
  if (cachedStore && sourceText === cachedSourceText) return cachedStore;

  const embeddings = new GoogleGenerativeAIEmbeddings({
    modelName: "models/embedding-001"
  });
  const docs = buildPolicyDocs(sourceText);
  cachedStore = await MemoryVectorStore.fromDocuments(docs, embeddings);
  cachedSourceText = sourceText;
  return cachedStore;
}

export function clearSemanticStoreCache() {
  cachedStore = null;
  cachedSourceText = "";
}

export async function retrievePolicyContext(query, k = 4) {
  const store = await getSemanticStore();
  const docs = await store.similaritySearch(query, k);
  return docs.map((d) => d.pageContent).join("\n\n");
}
