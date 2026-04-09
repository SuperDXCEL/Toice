from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch, json, re, time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

MODEL_ID = "Qwen/Qwen2.5-32B-Instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="cuda",
)
model.eval()
print("Model ready.")

SYSTEM_PROMPT = """Eres un sistema experto en detección de fraude telefónico y vishing en tiempo real.
Recibes fragmentos de conversación secuenciales de una llamada en curso.
Tienes memoria de los fragmentos anteriores de la misma llamada.

Detecta:
- Suplantación de identidad (banco, Agencia Tributaria, policía, soporte técnico)
- Urgencia o miedo ("su cuenta será bloqueada", "será arrestado", "actúe ahora")
- Solicitud de datos sensibles (contraseñas, PIN, OTP, DNI, tarjetas)
- Pagos inusuales (tarjetas regalo, bizum urgente, crypto, transferencia inmediata)
- Aislamiento ("no se lo diga a nadie", "no cuelgue", "no llame al banco")
- Pretexting (escenarios fabricados para ganar confianza)
- Escalada gradual de manipulación a lo largo de la llamada

Responde ÚNICAMENTE con JSON válido, sin texto adicional:
{
  "risk_level": "low|medium|high|critical",
  "tactics_detected": [],
  "confidence": 0.0,
  "summary": "explicación breve en español",
  "flagged_phrases": [],
  "escalating": false,
  "requires_immediate_action": false
}"""


# ── Memory Manager ────────────────────────────────────────────────────────────
@dataclass
class CallSession:
    call_id: str
    chunks: deque = field(default_factory=lambda: deque(maxlen=20))  # last ~2 mins at 5s chunks
    risk_history: list = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    chunk_count: int = 0

    def add_chunk(self, text: str):
        self.chunks.append(text)
        self.chunk_count += 1

    def get_context(self) -> str:
        if not self.chunks:
            return ""
        return "\n".join(self.chunks)

    def is_escalating(self) -> bool:
        """Check if risk level is trending upward across recent chunks."""
        risk_map = {"low": 0, "medium": 1, "high": 2, "critical": 3, "unknown": -1}
        recent = [risk_map.get(r, -1) for r in self.risk_history[-4:]]
        recent = [r for r in recent if r >= 0]
        return len(recent) >= 2 and recent[-1] > recent[0]

    def summary_stats(self) -> dict:
        return {
            "call_id": self.call_id,
            "duration_s": round(time.time() - self.start_time, 1),
            "chunks_analyzed": self.chunk_count,
            "risk_trend": self.risk_history[-5:],
            "escalating": self.is_escalating(),
        }


class SessionManager:
    def __init__(self):
        self.sessions: dict[str, CallSession] = {}

    def new_call(self, call_id: str) -> CallSession:
        """Start a new call session, resetting any existing one with the same ID."""
        if call_id in self.sessions:
            print(f"🔄 Resetting session for call {call_id}")
        self.sessions[call_id] = CallSession(call_id=call_id)
        return self.sessions[call_id]

    def get(self, call_id: str) -> Optional[CallSession]:
        return self.sessions.get(call_id)

    def end_call(self, call_id: str) -> Optional[dict]:
        """End call and return final summary stats."""
        session = self.sessions.pop(call_id, None)
        if session:
            return session.summary_stats()
        return None

    def reset_all(self):
        self.sessions.clear()
        print("🔄 All sessions reset.")


# ── Inference ─────────────────────────────────────────────────────────────────
def analyze_chunk(call_id: str, new_chunk: str, session_manager: SessionManager) -> dict:
    session = session_manager.get(call_id)
    if not session:
        print(f"⚠️  No session for {call_id}, creating one.")
        session = session_manager.new_call(call_id)

    # Add chunk to memory BEFORE analysis so model sees it in context
    session.add_chunk(new_chunk)
    context = session.get_context()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Historial de la llamada (fragmentos anteriores + actual):\n"
                f"---\n{context}\n---\n\n"
                f"Fragmento más reciente: {new_chunk}\n\n"
                f"Chunk #{session.chunk_count} | "
                f"Duración: {round(time.time() - session.start_time)}s | "
                f"Riesgo previo: {session.risk_history[-1] if session.risk_history else 'none'}"
            )
        }
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False
    )

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output[0][inputs["input_ids"].shape[-1]:]
    raw = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    try:
        cleaned = re.sub(r"```json|```", "", raw).strip()
        result = json.loads(cleaned)
    except json.JSONDecodeError:
        result = {
            "risk_level": "unknown",
            "tactics_detected": [],
            "confidence": 0.0,
            "summary": "Parse error",
            "flagged_phrases": [],
            "escalating": False,
            "requires_immediate_action": False,
            "raw": raw
        }

    # Store risk level in session history
    session.risk_history.append(result["risk_level"])

    # Override escalating field with our own trend detection
    result["escalating"] = session.is_escalating()
    result["chunk"] = session.chunk_count
    result["call_id"] = call_id

    return result


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    manager = SessionManager()

    # ── Simulated call 1 (vishing) ────────────────────────────────────────────
    CALL_1 = "call_001"
    manager.new_call(CALL_1)

    vishing_chunks = [
        "Hola, le llamo del departamento de seguridad de su banco BBVA.",
        "Hemos detectado una transferencia sospechosa de 3.400 euros desde su cuenta.",
        "Para bloquearla necesito verificar su identidad. ¿Puede decirme su DNI?",
        "Perfecto. Ahora necesito el código que le acaba de llegar por SMS. Es urgente.",
        "No cuelgue bajo ningún concepto o la transferencia se completará en segundos.",
    ]

    print("=" * 60)
    print(f"📞 CALL {CALL_1} STARTED")
    print("=" * 60)

    for chunk in vishing_chunks:
        print(f"\n🎙️  [{CALL_1}] {chunk}")
        start = time.time()
        result = analyze_chunk(CALL_1, chunk, manager)
        elapsed = time.time() - start

        risk_emoji = {"low": "🟢", "medium": "🟡", "high": "🟠", "critical": "🔴"}.get(result["risk_level"], "⚪")
        print(f"{risk_emoji} Risk: {result['risk_level'].upper()} | Confidence: {result['confidence']} | ⏱️ {elapsed:.2f}s")
        print(f"📋 Tactics: {result['tactics_detected']}")
        print(f"💬 {result['summary']}")
        if result["escalating"]:
            print("📈 ESCALATING — risk level rising across chunks")
        if result["requires_immediate_action"]:
            print("🚨 IMMEDIATE ACTION REQUIRED")

    final = manager.end_call(CALL_1)
    print(f"\n📊 Call ended: {final}")

    # ── Reset flag demo — same call ID reused ─────────────────────────────────
    print("\n" + "=" * 60)
    print(f"🔄 NEW CALL — resetting {CALL_1}")
    print("=" * 60)
    manager.new_call(CALL_1)  # reset flag — clears memory for this call_id

    result = analyze_chunk(CALL_1, "Hola, ¿sigue en garantía mi lavadora?", manager)
    print(f"\n🎙️  Post-reset chunk")
    print(f"Risk: {result['risk_level']} | {result['summary']}")
