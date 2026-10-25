import React, { useEffect, useRef, useState } from "react";
import "./App.css";
import { CollectPanel } from "./CollectPanel";
import { useSignRecognition } from "./useSignRecognition";

interface HealthData {
  status: string;
  version: string;
  environment: string;
  recognizer: string;
  llm_configured: boolean;
  primary_model: string;
}

interface TokenChip {
  id: string;
  token: string;
  confidence: number;
}

interface SpokenSentence {
  id: string;
  sentence: string;
  intent?: string;
  tokens?: TokenChip[];
  timestamp: string;
}

const COMMON_SIGNS = [
  "HELP", "I", "NEED", "DOCTOR", "PAIN", "WHERE", "TOILET", "YES", "NO", "THANK_YOU",
  "WATER", "FOOD", "BUS", "SHOP", "HOSPITAL", "OFFICE", "RESTAURANT",
];

const SCENARIOS = [
  { value: "general", label: "General" },
  { value: "shop", label: "Shop" },
  { value: "restaurant", label: "Restaurant" },
  { value: "bus", label: "Bus / Travel" },
  { value: "hospital_reception", label: "Hospital Reception" },
  { value: "office", label: "Office" },
];

const TONES = [
  { value: "polite", label: "Polite" },
  { value: "casual", label: "Casual" },
];

export const App: React.FC = () => {
  const [health, setHealth] = useState<HealthData | null>(null);
  const [loadingHealth, setLoadingHealth] = useState<boolean>(true);
  const [mockMode] = useState<boolean>(true);  const [typedInput, setTypedInput] = useState<string>("");
  const [tokens, setTokens] = useState<TokenChip[]>([]);

  // Compose output (one-way signer sentence)
  const [composedSentence, setComposedSentence] = useState<string>("");
  const [alternatives, setAlternatives] = useState<string[]>([]);
  const [composeIntent, setComposeIntent] = useState<string>("");
  const [ambiguityNote, setAmbiguityNote] = useState<string | null>(null);

  // Confirm-before-speak: ON by default; auto-speak OFF by default
  const [confirmPending, setConfirmPending] = useState<boolean>(false);
  const [autoSpeak, setAutoSpeak] = useState<boolean>(false);
  const [isSpeaking, setIsSpeaking] = useState<boolean>(false);
  const [ttsEnabled, setTtsEnabled] = useState<boolean>(true);
  const [ttsVoice, setTtsVoice] = useState<string>("");
  const [ttsRate, setTtsRate] = useState<number>(1.0);
  const [ttsPitch, setTtsPitch] = useState<number>(1.0);

  // Scenario + signer profile (Part A)
  const [scenario, setScenario] = useState<string>("general");
  const [signerName, setSignerName] = useState<string>("");
  const [signerTone, setSignerTone] = useState<string>("polite");

  // Spoken history (signer output only)
  const [spokenHistory, setSpokenHistory] = useState<SpokenSentence[]>([]);

  // UI state
  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const [showListenerView, setShowListenerView] = useState<boolean>(false);

  const [recognitionStatus, setRecognitionStatus] = useState<string>("Idle");
  const [recognitionError, setRecognitionError] = useState<string | null>(null);

  const {
    isActive: recognitionActive,
    ready: recognitionReady,
    error: recognitionHookError,
    fps: recognitionFps,
    videoRef: recognitionVideoRef,
    frameRef: recognitionCanvasRef,
    toggle: toggleRecognition,
  } = useSignRecognition({
    sessionId,
    onToken: (token) => {
      setTokens((prev) => [
        ...prev,
        {
          id: `recog-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
          token: token.token,
          confidence: token.confidence,
        },
      ]);
      setRecognitionStatus(`Recognized: ${token.token} (${(token.confidence * 100).toFixed(0)}%)`);
    },
    onStatus: setRecognitionStatus,
  });

  const [sessionId] = useState<string>(() => "session-" + Math.random().toString(36).substring(2, 9));
  const [view, setView] = useState<"chat" | "collect">("chat");

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const utteranceRef = useRef<SpeechSynthesisUtterance | null>(null);

  // Poll backend health
  const checkHealth = async () => {
    setLoadingHealth(true);
    try {
      const response = await fetch("http://127.0.0.1:8000/health");
      if (response.ok) {
        const data: HealthData = await response.json();
        setHealth(data);
      } else {
        setHealth(null);
      }
    } catch {
      setHealth(null);
    } finally {
      setLoadingHealth(false);
    }
  };

  useEffect(() => {
    checkHealth();
    const interval = setInterval(checkHealth, 15000);
    return () => clearInterval(interval);
  }, []);

  // Surface hook-level errors to local state
  useEffect(() => {
    setRecognitionError(recognitionHookError || null);
  }, [recognitionHookError]);

  // TTS voices
  useEffect(() => {
    if (!window.speechSynthesis) return;
    const populate = () => {
      const voices = window.speechSynthesis.getVoices();
      if (voices.length > 0 && !ttsVoice) {
        const enVoice = voices.find((v) => v.lang.startsWith("en")) || voices[0];
        setTtsVoice(enVoice?.voiceURI || enVoice?.name || "");
      }
    };
    populate();
    window.speechSynthesis.onvoiceschanged = populate;
  }, [ttsVoice]);

  // Stop speaking when utterance ends
  useEffect(() => {
    const onEnd = () => setIsSpeaking(false);
    window.speechSynthesis.addEventListener("end", onEnd);
    return () => window.speechSynthesis.removeEventListener("end", onEnd);
  }, []);

  // Scroll to bottom of spoken history
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [spokenHistory]);

  // Convert typed text into editable chips
  const handleAddTypedTokens = () => {
    if (!typedInput.trim()) return;
    const words = typedInput.trim().toUpperCase().split(/\s+/);
    const newChips: TokenChip[] = words.map((w, idx) => ({
      id: `${Date.now()}-${idx}-${w}`,
      token: w,
      confidence: 0.95,
    }));
    setTokens((prev) => [...prev, ...newChips]);
    setTypedInput("");
    inputRef.current?.focus();
  };

  const handleAddQuickSign = (sign: string) => {
    const newChip: TokenChip = {
      id: `${Date.now()}-${sign}`,
      token: sign,
      confidence: 0.95,
    };
    setTokens((prev) => [...prev, newChip]);
  };

  const handleRemoveToken = (id: string) => {
    setTokens((prev) => prev.filter((t) => t.id !== id));
  };

  const handleEditToken = (id: string, newToken: string) => {
    const clean = newToken.trim().toUpperCase();
    if (!clean) {
      handleRemoveToken(id);
      return;
    }
    setTokens((prev) =>
      prev.map((t) => (t.id === id ? { ...t, token: clean } : t))
    );
  };

  const handleConfidenceChange = (id: string, newConf: number) => {
    setTokens((prev) =>
      prev.map((t) => (t.id === id ? { ...t, confidence: Math.max(0.1, Math.min(1.0, newConf)) } : t))
    );
  };

  // Single Compose call: glosses -> first-person sentence spoken as the signer
  const handleCompose = async () => {
    let currentTokens = [...tokens];
    if (typedInput.trim()) {
      const words = typedInput.trim().toUpperCase().split(/\s+/);
      const newChips: TokenChip[] = words.map((w, idx) => ({
        id: `${Date.now()}-${idx}-${w}`,
        token: w,
        confidence: 0.95,
      }));
      currentTokens = [...currentTokens, ...newChips];
      setTokens(currentTokens);
      setTypedInput("");
    }

    if (currentTokens.length === 0) return;

    setIsProcessing(true);

    const payloadTokens = currentTokens.map((t) => ({
      token: t.token,
      confidence: t.confidence,
    }));

    const signerProfile = {
      name: signerName,
      tone: signerTone,
    };

    try {
      const response = await fetch("http://127.0.0.1:8000/compose", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          tokens: payloadTokens,
          session_id: sessionId,
          scenario,
          signer_profile: signerProfile,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();

      setComposedSentence(data.sentence || "");
      setAlternatives(data.alternatives || []);
      setComposeIntent(data.intent || "");
      setAmbiguityNote(data.ambiguity_note || null);

      // Clear active token chips once composed
      setTokens([]);

      // Confirm-before-speak: ON by default
      if (data.needs_confirmation) {
        setConfirmPending(true);
        setRecognitionStatus("Sentence composed — please confirm before speaking.");
      } else if (autoSpeak) {
        speakSentence(data.sentence);
        addSpokenHistory(data.sentence, data.intent, currentTokens);
      } else {
        setConfirmPending(true);
        setRecognitionStatus("Sentence composed — please confirm before speaking.");
      }
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Failed to compose sentence";
      setRecognitionStatus(`Compose error: ${msg}`);
    } finally {
      setIsProcessing(false);
    }
  };

  // Confirm the composed sentence and speak it
  const handleConfirmAndSpeak = (sentence: string) => {
    setComposedSentence(sentence);
    setConfirmPending(false);
    speakSentence(sentence);
    addSpokenHistory(sentence, composeIntent, tokens);
    setRecognitionStatus("Speaking...");
  };

  // Cancel composition
  const handleCancelCompose = () => {
    setComposedSentence("");
    setAlternatives([]);
    setConfirmPending(false);
    setRecognitionStatus("Composition cancelled.");
  };

  const addSpokenHistory = (sentence: string, intent?: string, toks?: TokenChip[]) => {
    const entry: SpokenSentence = {
      id: `spoken-${Date.now()}`,
      sentence,
      intent,
      tokens: toks,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    };
    setSpokenHistory((prev) => [entry, ...prev].slice(0, 50));
  };

  const speakSentence = (text: string) => {
    if (!ttsEnabled || !window.speechSynthesis || !text.trim()) return;
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = ttsRate;
    utterance.pitch = ttsPitch;
    if (ttsVoice) {
      const voices = window.speechSynthesis.getVoices();
      const match = voices.find((v) => v.voiceURI === ttsVoice || v.name === ttsVoice);
      if (match) utterance.voice = match;
    }
    utteranceRef.current = utterance;
    setIsSpeaking(true);
    window.speechSynthesis.speak(utterance);
  };

  const handleSpeak = () => {
    if (!composedSentence.trim()) return;
    speakSentence(composedSentence);
    addSpokenHistory(composedSentence, composeIntent, tokens);
    setRecognitionStatus("Speaking...");
  };

  const handleAlternativeSelect = (alt: string) => {
    handleConfirmAndSpeak(alt);
  };

  const handleNewConversation = async () => {
    try {
      await fetch(`http://127.0.0.1:8000/compose/reset?session_id=${sessionId}`, {
        method: "POST",
      });
    } catch {
      // Ignore network errors on reset
    }
    setSpokenHistory([]);
    setTokens([]);
    setTypedInput("");
    setComposedSentence("");
    setAlternatives([]);
    setConfirmPending(false);
    setIsSpeaking(false);
    window.speechSynthesis.cancel();
    setRecognitionStatus("New conversation started.");
  };

  const openListenerView = (text: string) => {
    // Full-screen high-contrast view for the hearing person
    const win = window.open("", "_blank", "width=400,height=300");
    if (!win) return;
    const doc = win.document;
    doc.open();
    doc.write(`
      <!DOCTYPE html>
      <html>
      <head>
        <meta charset="utf-8">
        <title>SignBridge — Show to Listener</title>
        <style>
          body { margin:0; background:#000; color:#0f0; font-family:monospace; display:flex; align-items:center; justify-content:center; height:100vh; overflow:hidden; }
          .text { font-size: clamp(3rem, 15vw, 8rem); text-align:center; line-height:1.2; white-space:pre-wrap; }
        </style>
      </head>
      <body>
        <div class="text">${escapeHtml(text)}</div>
      </body>
      </html>
    `);
    doc.close();
  };

  const escapeHtml = (text: string) => text.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c] || c);

  const isConnected = health?.status === "healthy";

  // Listener full-screen overlay (rendered inline instead of popup when possible)
  return (
    <div className="app-container">
      {showListenerView && (
        <div className="listener-overlay" onClick={() => setShowListenerView(false)}>
          <div className="listener-text">{composedSentence}</div>
          <div className="listener-hint">Tap anywhere to close</div>
        </div>
      )}

      {/* Top Header */}
      <header className="app-header">
        <div className="brand-group">
          <div className="brand-logo" aria-hidden="true">🤟</div>
          <div>
            <h1 className="brand-title">SignBridge AI</h1>
            <p className="brand-subtitle">One-Way ISL → Spoken English for the Hearing Person</p>
          </div>
        </div>

        {/* Connection Status Pill */}
        <div
          id="backend-connection-badge"
          className={`status-pill ${isConnected ? "status-connected" : "status-disconnected"}`}
          aria-live="polite"
        >
          <span className="status-dot" aria-hidden="true"></span>
          <span className="status-text">
            {loadingHealth && !health
              ? "Backend: connecting..."
              : isConnected
              ? `Backend: connected${health.llm_configured ? " · LLM live" : " · mock mode"}`
              : "Backend: offline"}
          </span>
        </div>

        {/* View toggle: Compose <-> Collect */}
        <button
          id="view-toggle-btn"
          className="control-btn"
          onClick={() => setView(view === "chat" ? "collect" : "chat")}
          title={`Switch to ${view === "chat" ? "Data Collection" : "Compose"} view`}
        >
          {view === "chat" ? "📊 Collect Data" : "💬 Compose Sentence"}
        </button>
      </header>

      {view === "chat" ? (
        <>
          {/* Recognition status banner */}
          {recognitionError && <div className="error-banner">{recognitionError}</div>}
          <div className={`status-line ${recognitionActive ? "status-active" : "status-idle"}`}>
            {recognitionStatus}
          </div>

          {/* Main One-Way Layout */}
          <main className="compose-layout">
            {/* Left Column: Sign Input & Token Chips */}
            <section className="input-column">
              <div className="dashboard-card input-panel">
                <div className="panel-header">
                  <h2 className="panel-title">Sign Input → Compose</h2>
                  <span className="mode-badge">{mockMode ? "Mock Mode" : "Camera Mode"}</span>
                </div>

                {/* Scenario selector */}
                <div className="form-row">
                  <label className="form-label">Scenario</label>
                  <select
                    className="form-select"
                    value={scenario}
                    onChange={(e) => setScenario(e.target.value)}
                  >
                    {SCENARIOS.map((s) => (
                      <option key={s.value} value={s.value}>
                        {s.label}
                      </option>
                    ))}
                  </select>
                </div>

                {/* Signer profile */}
                <div className="form-row">
                  <label className="form-label">Signer name</label>
                  <input
                    className="form-input"
                    type="text"
                    value={signerName}
                    onChange={(e) => setSignerName(e.target.value)}
                    placeholder="e.g. Ravi"
                  />
                </div>
                <div className="form-row">
                  <label className="form-label">Tone</label>
                  <select
                    className="form-select"
                    value={signerTone}
                    onChange={(e) => setSignerTone(e.target.value)}
                  >
                    {TONES.map((t) => (
                      <option key={t.value} value={t.value}>
                        {t.label}
                      </option>
                    ))}
                  </select>
                </div>

                {/* Quick Sign Shortcut Buttons */}
                <div className="quick-signs-row" aria-label="Common ISL Signs">
                  <span className="quick-signs-label">Quick Signs:</span>
                  <div className="quick-signs-chips">
                    {COMMON_SIGNS.map((sign) => (
                      <button
                        key={sign}
                        className="quick-sign-btn"
                        onClick={() => handleAddQuickSign(sign)}
                        title={`Add ${sign} token`}
                      >
                        +{sign}
                      </button>
                    ))}
                  </div>
                </div>

                {/* Typed Gloss Input Bar */}
                <form
                  className="typed-input-form"
                  onSubmit={(e) => {
                    e.preventDefault();
                    handleAddTypedTokens();
                  }}
                >
                  <input
                    ref={inputRef}
                    id="typed-gloss-input"
                    className="gloss-text-input"
                    type="text"
                    value={typedInput}
                    onChange={(e) => setTypedInput(e.target.value)}
                    placeholder="Type ISL glosses (e.g. WATER WANT)..."
                    aria-label="Type sign glosses"
                  />
                  <button type="submit" className="add-tokens-btn">
                    + Add
                  </button>
                </form>

                {/* Recognition camera controls (Phase 4) */}
                <div className="recognition-controls">
                  <button
                    id="recognition-camera-btn"
                    className={`action-btn ${recognitionActive ? "btn-stop" : recognitionReady ? "btn-start" : "btn-start"}`}
                    onClick={toggleRecognition}
                    disabled={!recognitionReady}
                    title="Start camera-based sign recognition"
                  >
                    {recognitionActive ? "⏹ Stop Recognition" : recognitionReady ? "▶ Start Recognition Camera" : "Loading models…"}
                  </button>
                  <span className={`recognition-status ${recognitionError ? "status-error" : "status-ok"}`}>
                    {recognitionError || recognitionStatus}
                  </span>
                  <span className={`recognition-status ${recognitionActive ? "status-ok" : "status-muted"}`}>
                    {recognitionActive ? `FPS: ${recognitionFps}` : ""}
                  </span>
                </div>

                {/* Camera preview */}
                {recognitionActive && recognitionVideoRef && (
                  <div className="recognition-preview">
                    <video
                      ref={recognitionVideoRef}
                      autoPlay
                      muted
                      playsInline
                      className="recognition-video"
                    />
                    <canvas ref={recognitionCanvasRef} className="recognition-overlay" />
                    <p className="recognition-hint">
                      Hold signs in frame. Recognized tokens appear as editable chips — review, edit,
                      then tap <strong>Compose & Speak</strong>.
                    </p>
                  </div>
                )}

                {/* Token chips (editable, confidence-shaded) */}
                <div className="chip-tray">
                  <h3 className="tray-title">Recognized Tokens ({tokens.length})</h3>
                  {tokens.length === 0 ? (
                    <p className="empty-tray">No tokens yet. Type or sign above.</p>
                  ) : (
                    <div className="chip-list">
                      {tokens.map((t) => (
                        <div key={t.id} className="chip-item">
                          <span
                            className={`chip-token ${t.confidence < 0.5 ? "chip-low-conf" : ""}`}
                            title={`Confidence: ${(t.confidence * 100).toFixed(0)}%`}
                          >
                            {t.token}
                          </span>
                          <span className="chip-conf">
                            {(t.confidence * 100).toFixed(0)}%
                          </span>
                          <input
                            className="chip-edit"
                            value={t.token}
                            onChange={(e) => handleEditToken(t.id, e.target.value)}
                            aria-label={`Edit token ${t.token}`}
                          />
                          <button
                            className="chip-conf-slider"
                            type="range"
                            min="0"
                            max="100"
                            value={Math.round(t.confidence * 100)}
                            onChange={(e) => handleConfidenceChange(t.id, parseInt(e.target.value) / 100)}
                            aria-label={`Confidence for ${t.token}`}
                          />
                          <button
                            className="chip-delete"
                            onClick={() => handleRemoveToken(t.id)}
                            aria-label={`Remove ${t.token}`}
                          >
                            ✕
                          </button>
                        </div>
                      ))}
                    </div>
                  )}
                </div>

                {/* Compose & Speak */}
                <div className="compose-actions">
                  <button
                    id="compose-speak-btn"
                    className={`action-btn ${isProcessing ? "btn-processing" : "btn-compose"}`}
                    onClick={handleCompose}
                    disabled={
                      isProcessing ||
                      tokens.length === 0 ||
                      (!signerName && false)
                    }
                    title="Compose sentence and prepare to speak"
                  >
                    {isProcessing ? "Composing…" : "✍ Compose Sentence"}
                  </button>

                  {confirmPending && composedSentence && (
                    <div className="confirm-panel">
                      <p className="confirm-text">
                        <strong>Composed sentence (speak as signer):</strong> "{composedSentence}"
                      </p>
                      {ambiguityNote && (
                        <p className="confirm-note">{ambiguityNote}</p>
                      )}
                      {alternatives.length > 0 && (
                        <div className="alternatives">
                          <span className="alt-label">Choose an alternative:</span>
                          {alternatives.map((alt, i) => (
                            <button
                              key={i}
                              className="action-btn btn-outline alt-btn"
                              onClick={() => handleAlternativeSelect(alt)}
                            >
                              "{alt}"
                            </button>
                          ))}
                          <button
                            className="action-btn btn-outline alt-btn"
                            onClick={() => handleConfirmAndSpeak(composedSentence)}
                          >
                            Use composed
                          </button>
                        </div>
                      )}
                      <div className="confirm-buttons">
                        <button
                          className="action-btn btn-confirm"
                          onClick={() => handleConfirmAndSpeak(composedSentence)}
                          title="Confirm and speak aloud"
                        >
                          ✅ Confirm & Speak
                        </button>
                        <button
                          className="action-btn btn-cancel"
                          onClick={handleCancelCompose}
                          title="Cancel composition"
                        >
                          ❌ Cancel
                        </button>
                      </div>
                      <div className="auto-speak-row">
                        <label className="checkbox-label">
                          <input
                            type="checkbox"
                            checked={autoSpeak}
                            onChange={(e) => setAutoSpeak(e.target.checked)}
                          />
                          Auto-speak after compose (OFF by default)
                        </label>
                      </div>
                    </div>
                  )}

                  {!confirmPending && composedSentence && !isProcessing && (
                    <div className="speak-panel">
                      <p className="speak-text">{composedSentence}</p>
                      <div className="speak-controls">
                        <button
                          className="action-btn btn-speak"
                          onClick={handleSpeak}
                          disabled={isSpeaking}
                          title="Speak the composed sentence aloud"
                        >
                          {isSpeaking ? "⏹ Stop Speaking" : "🔊 Speak"}
                        </button>
                        <button
                          className="action-btn btn-listener"
                          onClick={() => {
                            setShowListenerView(true);
                            openListenerView(composedSentence);
                          }}
                          title="Show large text to the hearing person"
                        >
                          👀 Show to Listener
                        </button>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </section>

            {/* Right Column: Spoken History & TTS Settings */}
            <section className="history-column">
              <div className="dashboard-card history-panel">
                <div className="panel-header">
                  <h2 className="panel-title">Spoken History</h2>
                  <button className="action-btn btn-outline" onClick={handleNewConversation}>
                    New Conversation
                  </button>
                </div>

                {/* TTS Settings */}
                <div className="tts-settings">
                  <h3 className="tts-title">Text-to-Speech</h3>
                  <div className="form-row">
                    <label className="form-label">Enabled</label>
                    <label className="checkbox-label">
                      <input
                        type="checkbox"
                        checked={ttsEnabled}
                        onChange={(e) => setTtsEnabled(e.target.checked)}
                      />
                      Speak aloud
                    </label>
                  </div>
                  <div className="form-row">
                    <label className="form-label">Voice</label>
                    <select className="form-select" value={ttsVoice} onChange={(e) => setTtsVoice(e.target.value)}>
                      <option value="">Default</option>
                    </select>
                  </div>
                  <div className="form-row">
                    <label className="form-label">Rate</label>
                    <input
                      className="form-input"
                      type="range"
                      min="0.5"
                      max="2"
                      step="0.1"
                      value={ttsRate}
                      onChange={(e) => setTtsRate(parseFloat(e.target.value))}
                    />
                    <span className="range-value">{ttsRate.toFixed(1)}</span>
                  </div>
                  <div className="form-row">
                    <label className="form-label">Pitch</label>
                    <input
                      className="form-input"
                      type="range"
                      min="0"
                      max="2"
                      step="0.1"
                      value={ttsPitch}
                      onChange={(e) => setTtsPitch(parseFloat(e.target.value))}
                    />
                    <span className="range-value">{ttsPitch.toFixed(1)}</span>
                  </div>
                  {isSpeaking && (
                    <div className="speaking-indicator">🔊 Speaking…</div>
                  )}
                </div>

                {/* Spoken history list */}
                <div className="history-list">
                  <h3 className="history-title">Spoken Sentences</h3>
                  {spokenHistory.length === 0 ? (
                    <p className="empty-history">No sentences spoken yet.</p>
                  ) : (
                    <ul>
                      {spokenHistory.map((entry) => (
                        <li key={entry.id} className="history-item">
                          <div className="history-sentence">"{entry.sentence}"</div>
                          <div className="history-meta">
                            {entry.intent ? `[${entry.intent}] · ` : ""}
                            {entry.timestamp}
                          </div>
                          <button
                            className="action-btn btn-outline history-replay"
                            onClick={() => speakSentence(entry.sentence)}
                            title="Replay this sentence"
                          >
                            🔊 Replay
                          </button>
                          <button
                            className="action-btn btn-outline history-show"
                            onClick={() => {
                              setShowListenerView(true);
                              openListenerView(entry.sentence);
                            }}
                            title="Show to listener"
                          >
                            👀 Show
                          </button>
                        </li>
                      ))}
                    </ul>
                  )}
                </div>
              </div>
            </section>
          </main>
        </>
      ) : (
        <CollectPanel />
      )}

      {/* Footer */}
      <footer className="app-footer">
        SignBridge AI · One-way ISL → spoken English · {health?.recognizer || "mock"} recognizer
      </footer>
    </div>
  );
};
