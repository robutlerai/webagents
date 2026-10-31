/**
 * A synchronous, incremental SHA-256 (FIPS 180-4). CASFS reads are verified
 * chunk by chunk, and the WASI and Pyodide mounts call the filesystem
 * synchronously, so the hash must work without `crypto.subtle` and without
 * holding a whole file. Runs in Node, browsers and Workers alike.
 */

const K = new Uint32Array([
  0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
  0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
  0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
  0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
  0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
  0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
  0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
  0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
]);

export class Sha256 {
  private h = new Uint32Array([0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19]);
  private buf = new Uint8Array(64);
  private n = 0;
  private len = 0;
  private w = new Uint32Array(64);

  private block(b: Uint8Array, o: number): void {
    const w = this.w;
    const h = this.h;
    for (let i = 0; i < 16; i++) w[i] = (b[o + 4 * i] << 24) | (b[o + 4 * i + 1] << 16) | (b[o + 4 * i + 2] << 8) | b[o + 4 * i + 3];
    for (let i = 16; i < 64; i++) {
      const x = w[i - 15];
      const y = w[i - 2];
      const s0 = ((x >>> 7) | (x << 25)) ^ ((x >>> 18) | (x << 14)) ^ (x >>> 3);
      const s1 = ((y >>> 17) | (y << 15)) ^ ((y >>> 19) | (y << 13)) ^ (y >>> 10);
      w[i] = (w[i - 16] + s0 + w[i - 7] + s1) | 0;
    }
    let a = h[0], bb = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
    for (let i = 0; i < 64; i++) {
      const S1 = ((e >>> 6) | (e << 26)) ^ ((e >>> 11) | (e << 21)) ^ ((e >>> 25) | (e << 7));
      const t1 = (hh + S1 + ((e & f) ^ (~e & g)) + K[i] + w[i]) | 0;
      const S0 = ((a >>> 2) | (a << 30)) ^ ((a >>> 13) | (a << 19)) ^ ((a >>> 22) | (a << 10));
      const t2 = (S0 + ((a & bb) ^ (a & c) ^ (bb & c))) | 0;
      hh = g; g = f; f = e; e = (d + t1) | 0; d = c; c = bb; bb = a; a = (t1 + t2) | 0;
    }
    h[0] += a; h[1] += bb; h[2] += c; h[3] += d; h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
  }

  update(data: Uint8Array | string): this {
    const u = typeof data === 'string' ? new TextEncoder().encode(data) : data;
    this.len += u.length;
    let i = 0;
    if (this.n) {
      const k = Math.min(64 - this.n, u.length);
      this.buf.set(u.subarray(0, k), this.n);
      this.n += k;
      i = k;
      if (this.n === 64) {
        this.block(this.buf, 0);
        this.n = 0;
      }
    }
    for (; i + 64 <= u.length; i += 64) this.block(u, i);
    if (i < u.length) {
      this.buf.set(u.subarray(i), 0);
      this.n = u.length - i;
    }
    return this;
  }

  digestBytes(): Uint8Array {
    const bits = this.len * 8;
    const pad = new Uint8Array((this.n < 56 ? 56 : 120) - this.n + 8);
    pad[0] = 0x80;
    const hi = Math.floor(bits / 0x100000000);
    const lo = bits >>> 0;
    const p = pad.length;
    pad[p - 8] = hi >>> 24; pad[p - 7] = hi >>> 16; pad[p - 6] = hi >>> 8; pad[p - 5] = hi;
    pad[p - 4] = lo >>> 24; pad[p - 3] = lo >>> 16; pad[p - 2] = lo >>> 8; pad[p - 1] = lo;
    this.len -= pad.length;
    this.update(pad);
    const out = new Uint8Array(32);
    for (let i = 0; i < 8; i++) {
      out[4 * i] = this.h[i] >>> 24;
      out[4 * i + 1] = this.h[i] >>> 16;
      out[4 * i + 2] = this.h[i] >>> 8;
      out[4 * i + 3] = this.h[i];
    }
    return out;
  }

  digest(): string {
    return toHex(this.digestBytes());
  }
}

export const toHex = (u8: Uint8Array): string => {
  let s = '';
  for (const b of u8) s += (b < 16 ? '0' : '') + b.toString(16);
  return s;
};
export const fromHex = (h: string): Uint8Array => {
  const u = new Uint8Array(h.length / 2);
  for (let i = 0; i < u.length; i++) u[i] = parseInt(h.substr(2 * i, 2), 16);
  return u;
};
export const sha256 = (bytes: Uint8Array | string): string => new Sha256().update(bytes).digest();
