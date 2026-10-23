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

// Mixer: one jitter queue per slot (0 = the AI, 1..5 = people). Messages in:
//   ArrayBuffer (1 byte slot + 960 bytes PCM) | {flush: slot} | {gain: g, slot, ramp: seconds}
// A slot starts playing once ~60 ms is queued and drops its oldest audio past 300 ms. Posts {levels: [6 peaks]}
// about 10 times a second.
const SLOTS = 6, PRIME = 3, MAX = 15, RATE = 24000;
class MixerProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.q = Array.from({ length: SLOTS }, () => []);
    this.off = new Int32Array(SLOTS);
    this.primed = new Uint8Array(SLOTS);
    this.gain = new Float32Array(SLOTS).fill(1);
    this.target = new Float32Array(SLOTS).fill(1);
    this.step = new Float32Array(SLOTS);
    this.peak = new Float32Array(SLOTS);
    this.drops = 0; this.n = 0;
    this.port.onmessage = (e) => {
      const d = e.data;
      if (d instanceof ArrayBuffer) {
        const s = new Uint8Array(d, 0, 1)[0];
        if (s >= SLOTS) return;
        const q = this.q[s];
        q.push(new Int16Array(d.slice(1)));
        if (q.length >= PRIME) this.primed[s] = 1;
        if (q.length > MAX) { this.drops += q.length - PRIME; q.splice(0, q.length - PRIME); this.off[s] = 0; }
      } else if (d.flush !== undefined) {
        this.q[d.flush] = []; this.off[d.flush] = 0; this.primed[d.flush] = 0;
      } else if (d.gain !== undefined) {
        const s = d.slot || 0;
        this.target[s] = d.gain;
        this.step[s] = Math.abs(d.gain - this.gain[s]) / Math.max(1, (d.ramp ?? 0.03) * RATE);
      }
    };
  }
  process(_, outputs) {
    const out = outputs[0][0];
    out.fill(0);
    for (let s = 0; s < SLOTS; s++) {
      const q = this.q[s];
      if (!this.primed[s]) { this.gain[s] = this.target[s]; continue; }
      let g = this.gain[s], peak = this.peak[s];
      const tg = this.target[s], st = this.step[s];
      for (let i = 0; i < out.length; i++) {
        if (!q.length) { this.primed[s] = 0; break; } // ran dry: wait for ~60 ms again
        const f = q[0];
        if (g !== tg) g = g < tg ? Math.min(tg, g + st) : Math.max(tg, g - st);
        const v = f[this.off[s]++] / 32768;
        out[i] += v * g;
        peak = Math.max(peak, Math.abs(v * g));
        if (this.off[s] >= f.length) { q.shift(); this.off[s] = 0; }
      }
      this.gain[s] = g; this.peak[s] = peak;
    }
    for (let i = 0; i < out.length; i++) out[i] = Math.max(-1, Math.min(1, out[i]));
    for (let c = 1; c < outputs[0].length; c++) outputs[0][c].set(out);
    if ((this.n += out.length) >= RATE / 10) {
      this.port.postMessage({ levels: Array.from(this.peak), gain: this.gain[0], drops: this.drops });
      this.peak.fill(0); this.n = 0;
    }
    return true;
  }
}

registerProcessor("mic-processor", MicProcessor);
registerProcessor("mixer-processor", MixerProcessor);
