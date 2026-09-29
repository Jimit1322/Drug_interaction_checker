"use client";

import { useState, useRef, useEffect } from "react";
import {
  Send,
  BookOpen,
  FlaskConical,
  Shield,
  Loader2,
  ChevronDown,
} from "lucide-react";

// ── Types ──────────────────────────────────────────────────────────
interface Source {
  type: string;
  name: string;
  url: string;
  section: string;
  pmid: string;
  year: string;
  icon: string;
}

interface Message {
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
  severity?: string;
  drugs?: string[];
  loading?: boolean;
}

// ── Severity badge ─────────────────────────────────────────────────
function SeverityBadge({ severity }: { severity: string }) {
  const config: Record<string, { color: string; bg: string; icon: string }> = {
    SEVERE: {
      color: "text-red-700",
      bg: "bg-red-50 border-red-200",
      icon: "🚨",
    },
    MODERATE: {
      color: "text-orange-700",
      bg: "bg-orange-50 border-orange-200",
      icon: "⚠️",
    },
    MILD: {
      color: "text-yellow-700",
      bg: "bg-yellow-50 border-yellow-200",
      icon: "ℹ️",
    },
    UNKNOWN: {
      color: "text-gray-600",
      bg: "bg-gray-50 border-gray-200",
      icon: "❓",
    },
  };
  const c = config[severity] || config.UNKNOWN;
  return (
    <span
      className={`inline-flex items-center gap-1 px-3 py-1 rounded-full border text-xs font-semibold ${c.bg} ${c.color}`}
    >
      {c.icon} {severity}
    </span>
  );
}

// ── Source card ────────────────────────────────────────────────────
function SourceCard({ source }: { source: Source }) {
  return (
    <a
      href={source.url || "#"}
      target="_blank"
      rel="noopener noreferrer"
      className="flex items-start gap-2 p-2 rounded-lg bg-white border border-gray-100 hover:border-blue-200 hover:bg-blue-50 transition-all text-xs group"
    >
      <span className="text-base mt-0.5">{source.icon}</span>
      <div className="min-w-0">
        <p className="font-medium text-gray-700 group-hover:text-blue-700 truncate">
          {source.name}
        </p>
        <p className="text-gray-400 mt-0.5">
          {source.type}
          {source.section ? ` · ${source.section}` : ""}
          {source.year ? ` · ${source.year}` : ""}
          {source.pmid ? ` · PMID ${source.pmid}` : ""}
        </p>
      </div>
    </a>
  );
}

// ── Drug autocomplete input ────────────────────────────────────────
function DrugInput({
  value,
  onChange,
  placeholder,
}: {
  value: string;
  onChange: (v: string) => void;
  placeholder: string;
}) {
  const [suggestions, setSuggestions] = useState<string[]>([]);
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handleClick = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        setOpen(false);
      }
    };
    document.addEventListener("mousedown", handleClick);
    return () => document.removeEventListener("mousedown", handleClick);
  }, []);

  const fetchSuggestions = async (q: string) => {
    if (q.length < 2) {
      setSuggestions([]);
      setOpen(false);
      return;
    }
    try {
      const r = await fetch(
        `http://localhost:8000/drugs/search?q=${encodeURIComponent(q)}`
      );
      const data = await r.json();
      setSuggestions(data.drugs || []);
      setOpen((data.drugs || []).length > 0);
    } catch {
      setSuggestions([]);
    }
  };

  return (
    <div ref={ref} className="relative flex-1">
      <input
        type="text"
        value={value}
        onChange={(e) => {
          onChange(e.target.value);
          fetchSuggestions(e.target.value);
        }}
        placeholder={placeholder}
        className="w-full px-4 py-2.5 rounded-xl border border-gray-200 focus:outline-none focus:ring-2 focus:ring-blue-500 text-sm bg-white"
      />
      {open && suggestions.length > 0 && (
        <ul className="absolute z-20 w-full mt-1 bg-white border border-gray-200 rounded-xl shadow-lg overflow-hidden">
          {suggestions.map((s) => (
            <li
              key={s}
              onClick={() => {
                onChange(s);
                setOpen(false);
              }}
              className="px-4 py-2.5 text-sm hover:bg-blue-50 cursor-pointer text-gray-700"
            >
              {s}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

// ── Message bubble ─────────────────────────────────────────────────
function MessageBubble({ msg }: { msg: Message }) {
  const [showSources, setShowSources] = useState(false);

  if (msg.role === "user") {
    return (
      <div className="flex justify-end">
        <div className="bg-blue-600 text-white px-4 py-3 rounded-2xl rounded-tr-sm max-w-[80%] text-sm leading-relaxed">
          {msg.content}
        </div>
      </div>
    );
  }

  if (msg.loading) {
    return (
      <div className="flex justify-start">
        <div className="bg-white border border-gray-100 px-5 py-4 rounded-2xl rounded-tl-sm shadow-sm">
          <div className="flex items-center gap-3">
            <Loader2 className="w-4 h-4 animate-spin text-blue-500 shrink-0" />
            <div>
              <p className="text-sm text-gray-600">Searching medical databases...</p>
              <p className="text-xs text-gray-400 mt-0.5">
                Checking FDA labels + PubMed research
              </p>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex justify-start max-w-[90%]">
      <div className="bg-white border border-gray-100 px-5 py-4 rounded-2xl rounded-tl-sm shadow-sm w-full">
        {msg.severity && msg.severity !== "UNKNOWN" && (
          <div className="mb-3">
            <SeverityBadge severity={msg.severity} />
          </div>
        )}

        <div className="text-sm text-gray-700 leading-relaxed whitespace-pre-wrap">
          {msg.content
            .replace(/⚠️\s*SEVERITY:\s*(SEVERE|MODERATE|MILD|UNKNOWN)\n*/i, "")
            .trim()}
        </div>

        {msg.sources && msg.sources.length > 0 && (
          <div className="mt-4">
            <button
              onClick={() => setShowSources(!showSources)}
              className="flex items-center gap-1.5 text-xs text-blue-600 hover:text-blue-800 font-medium"
            >
              <BookOpen className="w-3.5 h-3.5" />
              {showSources ? "Hide" : "Show"} {msg.sources.length} sources
              <ChevronDown
                className={`w-3 h-3 transition-transform ${
                  showSources ? "rotate-180" : ""
                }`}
              />
            </button>

            {showSources && (
              <div className="mt-2 grid gap-1.5">
                {Array.from(
                  new Map(msg.sources.map((s) => [s.name, s])).values()
                ).map((s, i) => (
                  <SourceCard key={i} source={s} />
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

// ── Main page ──────────────────────────────────────────────────────
export default function Home() {
  const [drug1, setDrug1] = useState("");
  const [drug2, setDrug2] = useState("");
  const [context, setContext] = useState("");
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState(false);
  const [stats, setStats] = useState<{ total_chunks: number } | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    fetch("http://localhost:8000/stats")
      .then((r) => r.json())
      .then(setStats)
      .catch(() => {});
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const buildQuestion = () => {
    if (drug1 && drug2) {
      return `Can I take ${drug1} and ${drug2} together?${
        context ? ` Patient context: ${context}` : ""
      }`;
    }
    return question;
  };

  const handleSubmit = async () => {
    const q = buildQuestion();
    if (!q.trim() || loading) return;

    const userMsg: Message = { role: "user", content: q };
    const loadingMsg: Message = { role: "assistant", content: "", loading: true };

    setMessages((prev) => [...prev, userMsg, loadingMsg]);
    setLoading(true);
    setDrug1("");
    setDrug2("");
    setContext("");
    setQuestion("");

    try {
      const r = await fetch("http://localhost:8000/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: q,
          patient_context: context,
        }),
      });
      const data = await r.json();

      setMessages((prev) => [
        ...prev.slice(0, -1),
        {
          role: "assistant",
          content: data.answer,
          sources: data.sources,
          severity: data.severity,
          drugs: data.drugs_found,
        },
      ]);
    } catch {
      setMessages((prev) => [
        ...prev.slice(0, -1),
        {
          role: "assistant",
          content:
            "Error connecting to server. Make sure the backend is running on port 8000.",
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const EXAMPLES = [
    "Can I take Warfarin and Ibuprofen together?",
    "Is Amoxicillin safe with Cyclosporine?",
    "What are the risks of Lidocaine with Bupropion?",
  ];

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col">
      {/* Header */}
      <header className="bg-white border-b border-gray-200 px-6 py-4 flex items-center justify-between sticky top-0 z-10">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl bg-blue-600 flex items-center justify-center">
            <FlaskConical className="w-5 h-5 text-white" />
          </div>
          <div>
            <h1 className="font-semibold text-gray-900 text-sm">
              Drug Interaction Checker
            </h1>
            <p className="text-xs text-gray-400">
              Powered by FDA DailyMed + PubMed + Gemini AI
            </p>
          </div>
        </div>
        {stats && (
          <div className="hidden sm:flex items-center gap-1.5 text-xs text-gray-400 bg-gray-50 px-3 py-1.5 rounded-full border">
            <Shield className="w-3 h-3 text-green-500" />
            {stats.total_chunks.toLocaleString()} medical documents indexed
          </div>
        )}
      </header>

      {/* Chat area */}
      <main className="flex-1 overflow-y-auto px-4 py-6 max-w-3xl mx-auto w-full">
        {messages.length === 0 && (
          <div className="text-center py-16">
            <div className="w-16 h-16 rounded-2xl bg-blue-50 flex items-center justify-center mx-auto mb-4">
              <FlaskConical className="w-8 h-8 text-blue-500" />
            </div>
            <h2 className="text-xl font-semibold text-gray-800 mb-2">
              Check Drug Interactions
            </h2>
            <p className="text-gray-500 text-sm max-w-md mx-auto mb-8 leading-relaxed">
              Ask about any drug combination. Answers are grounded in FDA drug
              labels and peer-reviewed research with citations.
            </p>
            <div className="flex flex-col gap-2 max-w-sm mx-auto">
              {EXAMPLES.map((ex) => (
                <button
                  key={ex}
                  onClick={() => setQuestion(ex)}
                  className="text-left px-4 py-3 rounded-xl border border-gray-200 hover:border-blue-300 hover:bg-blue-50 text-sm text-gray-600 hover:text-blue-700 transition-all"
                >
                  {ex}
                </button>
              ))}
            </div>
          </div>
        )}

        <div className="space-y-4">
          {messages.map((msg, i) => (
            <MessageBubble key={i} msg={msg} />
          ))}
        </div>
        <div ref={bottomRef} />
      </main>

      {/* Input area */}
      <div className="border-t border-gray-200 bg-white px-4 py-4">
        <div className="max-w-3xl mx-auto space-y-3">
          <div className="flex gap-2">
            <DrugInput
              value={drug1}
              onChange={setDrug1}
              placeholder="Drug 1 (e.g. Warfarin)"
            />
            <span className="self-center text-gray-400 font-medium text-sm">
              +
            </span>
            <DrugInput
              value={drug2}
              onChange={setDrug2}
              placeholder="Drug 2 (e.g. Ibuprofen)"
            />
          </div>

          <input
            type="text"
            value={context}
            onChange={(e) => setContext(e.target.value)}
            placeholder="Patient context (optional) — e.g. 65 year old with diabetes"
            className="w-full px-4 py-2.5 rounded-xl border border-gray-200 focus:outline-none focus:ring-2 focus:ring-blue-500 text-sm bg-white"
          />

          <div className="flex gap-2">
            <input
              type="text"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Or ask freely: 'Is it safe to take aspirin with metformin?'"
              className="flex-1 px-4 py-2.5 rounded-xl border border-gray-200 focus:outline-none focus:ring-2 focus:ring-blue-500 text-sm bg-white"
            />
            <button
              onClick={handleSubmit}
              disabled={loading || (!drug1 && !drug2 && !question.trim())}
              className="px-4 py-2.5 bg-blue-600 text-white rounded-xl hover:bg-blue-700 disabled:opacity-40 disabled:cursor-not-allowed transition-colors flex items-center gap-2"
            >
              {loading ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <Send className="w-4 h-4" />
              )}
              <span className="text-sm font-medium">Ask</span>
            </button>
          </div>

          <p className="text-center text-xs text-gray-400">
            ⚕️ For educational purposes only. Always consult a pharmacist or
            doctor.
          </p>
        </div>
      </div>
    </div>
  );
}