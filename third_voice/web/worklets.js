// Mic: collect 20 ms frames (480 samples at 24 kHz) as Int16 and post them to the main thread.
class MicProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.buf = new Int16Array(480);
    this.n = 0;
  }
  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (!ch) return true;
    for (let i = 0; i < ch.length; i++) {
      const s = Math.max(-1, Math.min(1, ch[i]));
      this.buf[this.n++] = s < 0 ? s * 32768 : s * 32767;
      if (this.n === 480) {
        this.port.postMessage(this.buf.buffer, [this.buf.buffer]);
        this.buf = new Int16Array(480);
        this.n = 0;
      }
    }
    return true;
  }
}

// Player: a small jitter buffer of Int16 frames from the server; "flush" drops everything queued.
class PlayerProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.q = [];
    this.off = 0;
    this.primed = false;
    this.port.onmessage = (e) => {
      if (e.data === "flush") { this.q = []; this.off = 0; this.primed = false; return; }
      this.q.push(new Int16Array(e.data));
      if (this.q.length >= 2) this.primed = true;
    };
  }
  process(_, outputs) {
    const out = outputs[0][0];
    let level = 0;
    for (let i = 0; i < out.length; i++) {
      if (!this.primed || this.q.length === 0) { out[i] = 0; continue; }
      const f = this.q[0];
      const v = f[this.off++] / 32768;
      out[i] = v;
      level = Math.max(level, Math.abs(v));
      if (this.off >= f.length) { this.q.shift(); this.off = 0; }
    }
    if (this.q.length > 25) this.q.splice(0, this.q.length - 5); // never drift more than ~100 ms behind
    for (let c = 1; c < outputs[0].length; c++) outputs[0][c].set(out);
    this.port.postMessage(level);
    return true;
  }
}

registerProcessor("mic-processor", MicProcessor);
registerProcessor("player-processor", PlayerProcessor);
