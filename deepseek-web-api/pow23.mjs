const MASK = (1n << 64n) - 1n;
const RC = [
  0x1n, 0x8082n, 0x800000000000808an, 0x8000000080008000n,
  0x808bn, 0x80000001n, 0x8000000080008081n, 0x8000000000008009n,
  0x8an, 0x88n, 0x80008009n, 0x8000000an,
  0x8000808bn, 0x800000000000008bn, 0x8000000000008089n,
  0x8000000000008003n, 0x8000000000008002n,
  0x8000000000000080n, 0x800an, 0x800000008000000an,
  0x8000000080008081n, 0x8000000000008080n,
  0x80000001n, 0x8000000080008008n,
];
const RHO = [
  0, 1, 62, 28, 27,
  36, 44, 6, 55, 20,
  3, 10, 43, 25, 39,
  41, 45, 15, 21, 8,
  18, 2, 61, 56, 14,
];

const rot = (value, bits) => bits === 0 ? value : ((value << BigInt(bits)) | (value >> BigInt(64 - bits))) & MASK;

export function deepseekHash(input) {
  const bytes = Buffer.from(input, 'utf8');
  const padded = Buffer.alloc(Math.ceil((bytes.length + 1) / 136) * 136 || 136);
  bytes.copy(padded);
  padded[bytes.length] = 0x06;
  padded[padded.length - 1] |= 0x80;
  const a = Array(25).fill(0n);
  for (let block = 0; block < padded.length; block += 136) {
    for (let i = 0; i < 17; i++) a[i] ^= padded.readBigUInt64LE(block + i * 8);
    for (let round = 1; round < 24; round++) {
      const c = Array(5);
      const d = Array(5);
      const b = Array(25);
      for (let x = 0; x < 5; x++) c[x] = a[x] ^ a[x + 5] ^ a[x + 10] ^ a[x + 15] ^ a[x + 20];
      for (let x = 0; x < 5; x++) d[x] = c[(x + 4) % 5] ^ rot(c[(x + 1) % 5], 1);
      for (let x = 0; x < 5; x++) for (let y = 0; y < 5; y++) a[x + 5 * y] ^= d[x];
      for (let x = 0; x < 5; x++) for (let y = 0; y < 5; y++) b[y + 5 * ((2 * x + 3 * y) % 5)] = rot(a[x + 5 * y], RHO[x + 5 * y]);
      for (let x = 0; x < 5; x++) for (let y = 0; y < 5; y++) a[x + 5 * y] = b[x + 5 * y] ^ ((~b[(x + 1) % 5 + 5 * y]) & b[(x + 2) % 5 + 5 * y]);
      a[0] ^= RC[round];
    }
  }
  const out = Buffer.alloc(32);
  for (let i = 0; i < 4; i++) out.writeBigUInt64LE(a[i] & MASK, i * 8);
  return out.toString('hex');
}

if (process.argv[1]?.endsWith('pow23.mjs')) console.log(deepseekHash(process.argv[2] ?? 'abc_123_0'));
