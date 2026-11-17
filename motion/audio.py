"""영상용 배경 트랙을 합성한다 (numpy만 사용, 외부 음원 없음).
120 BPM · 40초 · 장면 전환(4/6/8/12/…/32/36초)에 맞춘 히트와 우쉬.
  python3 audio.py  →  out/track.wav
"""
import os, wave
import numpy as np

SR, DUR, BEAT = 44100, 40.0, 0.5
N = int(SR * DUR)
TAIL = SR * 3
rng = np.random.default_rng(11)

def bus():
    return np.zeros((2, N + TAIL))

drums, music, fx, send = bus(), bus(), bus(), bus()

def put(b, x, t, g=1.0, pan=0.0):
    """x: mono(1d) or stereo(2,n). pan -1..1"""
    i = int(round(t * SR))
    if i < 0:
        x = x[..., -i:]; i = 0
    if x.ndim == 1:
        l, r = np.cos((pan + 1) * np.pi / 4), np.sin((pan + 1) * np.pi / 4)
        x = np.stack([x * l, x * r]) * np.sqrt(2) / 1.0 * 0.7071
    n = min(x.shape[1], b.shape[1] - i)
    if n > 0:
        b[:, i:i + n] += x[:, :n] * g

def filt(x, lo=None, hi=None, order=2):
    X = np.fft.rfft(x); f = np.fft.rfftfreq(len(x), 1 / SR); f[0] = 1e-6
    H = np.ones_like(f)
    if hi: H *= 1 / np.sqrt(1 + (f / hi) ** (2 * order))
    if lo: H *= 1 / np.sqrt(1 + (lo / f) ** (2 * order))
    return np.fft.irfft(X * H, len(x))

def tt(d):
    return np.arange(int(d * SR)) / SR

def hz(m):
    return 440.0 * 2 ** ((m - 69) / 12)

def noise(d):
    return rng.standard_normal(int(d * SR))

def fade(x, a=0.004, r=0.02):
    n = len(x); e = np.ones(n)
    na, nr = max(1, int(a * SR)), max(1, int(r * SR))
    e[:na] = np.linspace(0, 1, na); e[-nr:] *= np.linspace(1, 0, nr)
    return x * e

def saw(f, t, ph=0.0):
    return 2 * ((f * t + ph) % 1.0) - 1

# ---------------- instruments ----------------
def kick():
    t = tt(0.5)
    f = 44 + 120 * np.exp(-t / 0.032)
    x = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.17)
    c = filt(noise(0.5) * np.exp(-t / 0.004), lo=1800) * 0.35
    return np.tanh((x + c) * 1.6) * 0.9

def clap():
    t = tt(0.4)
    env = sum(np.where(t >= d, np.exp(-(t - d) / 0.009), 0) for d in (0, 0.012, 0.024)) * 0.7
    env += np.where(t >= 0.03, np.exp(-(t - 0.03) / 0.085), 0)
    return np.stack([filt(noise(0.4), lo=900, hi=5200) * env for _ in range(2)]) * 0.5

def hat(dec=0.02):
    t = tt(dec * 7)
    return filt(noise(dec * 7), lo=7500, order=3) * np.exp(-t / dec)

def thud(f0=120):
    t = tt(0.45)
    f = f0 * 0.62 + f0 * 0.5 * np.exp(-t / 0.05)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.13)

def bass(m, d):
    t = tt(d); f = hz(m)
    x = filt(saw(f, t) * 0.55 + np.sin(2 * np.pi * f * t), hi=380)
    env = np.minimum(1, t / 0.006) * (0.72 + 0.28 * np.exp(-t / 0.08))
    return fade(x * env, 0.003, 0.035)

def pad(notes, d, cutoff=1500, att=0.3, rel=0.5):
    t = tt(d + rel); out = np.zeros((2, len(t)))
    for m in notes:
        for ch in range(2):
            v = np.zeros(len(t))
            for det in (-0.09, 0.0, 0.11):
                v += saw(hz(m + det + (0.035 if ch else -0.035)), t, rng.random())
            out[ch] += v
    out = np.stack([filt(out[c], lo=140, hi=cutoff) for c in range(2)])
    env = np.minimum(1, t / att) * np.where(t > d, np.maximum(0, 1 - (t - d) / rel), 1)
    return out * env / (len(notes) * 3)

def pluck(m, d=0.5, dec=0.10, bright=1.0):
    t = tt(d); f = hz(m)
    x = np.sin(2 * np.pi * f * t) + 0.45 * bright * np.sin(4 * np.pi * f * t) + 0.2 * bright * np.sin(6 * np.pi * f * t)
    return fade(x * np.exp(-t / dec), 0.002, 0.03) * 0.5

def bell(m, d=1.6):
    t = tt(d); f = hz(m)
    x = np.sin(2 * np.pi * f * t) + 0.5 * np.sin(2 * np.pi * f * 2.01 * t) * np.exp(-t / 0.3) + 0.25 * np.sin(2 * np.pi * f * 3.02 * t) * np.exp(-t / 0.12)
    return fade(x * np.exp(-t / 0.45), 0.002, 0.05) * 0.4

def whoosh(d=0.75, peak=0.55):
    t = tt(d); u = t / d
    env = np.where(u < peak, (u / peak) ** 2.2, ((1 - u) / (1 - peak)) ** 1.6)
    n = noise(d)
    lo, mid, hi_ = filt(n, lo=250, hi=1200), filt(n, lo=1200, hi=4500), filt(n, lo=4500, hi=12000)
    x = (lo * (1 - u) ** 1.5 * 1.4 + mid * np.sin(np.pi * u) + hi_ * u ** 1.5 * 1.2) * env
    pan = np.clip(u * 2 - 1, -1, 1)
    return np.stack([x * np.cos((pan + 1) * np.pi / 4), x * np.sin((pan + 1) * np.pi / 4)])

def riser(d=1.0):
    t = tt(d); u = t / d
    n = noise(d)
    x = (filt(n, lo=600, hi=3000) * (1 - u) + filt(n, lo=3000, hi=13000) * u) * u ** 2.2
    s = np.sin(2 * np.pi * np.cumsum(180 + 720 * u ** 2) / SR) * u ** 2 * 0.25
    return fade(x * 0.9 + s, 0.01, 0.006)

def swish(d=0.22):
    t = tt(d); u = t / d
    return filt(noise(d), lo=2200, hi=9000) * np.sin(np.pi * u) ** 2

def impact(big=1.0):
    t = tt(1.6)
    sub = np.sin(2 * np.pi * np.cumsum(46 + 34 * np.exp(-t / 0.09)) / SR) * np.exp(-t / 0.34)
    body = filt(noise(1.6), hi=1800) * np.exp(-t / 0.11) * 0.5
    return np.tanh((sub + body) * 1.4 * big) * 0.62

def crash():
    t = tt(1.8)
    return np.stack([filt(noise(1.8), lo=3800, order=2) * np.exp(-t / 0.42) for _ in range(2)])

def tick():
    t = tt(0.05)
    return (filt(noise(0.05), lo=2500) * np.exp(-t / 0.003) + np.sin(2 * np.pi * 1700 * t) * np.exp(-t / 0.012) * 0.6)

# ---------------- arrangement ----------------
CH = {  # bass midi, pad voicing
    'F': (41, [53, 57, 60, 64]), 'G': (43, [55, 59, 62, 64]),
    'Am': (45, [57, 60, 64, 67]), 'C': (48, [55, 60, 62, 64, 67]),
}
BARS = ['F', 'G', 'Am', 'C'] * 4 + ['F', 'G', 'C', 'C']
K, CL, H1, H2, OH = kick(), clap(), hat(0.018), hat(0.03), hat(0.085)
kick_times = []

for b, name in enumerate(BARS):
    t0 = b * 2.0; bm, notes = CH[name]
    # pad
    if b < 2:   put(music, pad(notes, 2.0, cutoff=700 + 500 * b, att=0.9 if b == 0 else 0.3), t0, 0.50)
    elif b == 2: put(music, pad(notes, 2.0, cutoff=1100), t0, 0.55)
    elif b == 19: pass
    elif b == 18: put(music, pad(notes, 3.6, cutoff=2800, rel=1.6), t0, 0.90)
    elif b in (16, 17): put(music, pad(notes, 2.0, cutoff=1500), t0, 0.80)
    else:       put(music, pad(notes, 2.0, cutoff=2800), t0, 0.85)
    if not (4 <= b <= 15):
        continue
    breakdown = (b == 14)                      # 28–30s: 숫자 '46' 직전 브레이크
    for s in range(16):                        # 16분음표
        t = t0 + s * 0.125
        quiet = breakdown and s < 8
        last_half = s >= 14 and b in (5, 7, 9, 11, 13, 15)   # 전환 직전 반 박은 비움
        if s % 4 == 0 and not quiet and not last_half:
            put(drums, K, t, 0.68); kick_times.append(t)
        if s % 8 == 4 and b >= 5 and not quiet:
            put(drums, CL, t, 0.70); put(send, CL, t, 0.22)
        if not quiet:
            if s % 4 == 2: put(drums, OH if (b >= 10 and s % 8 == 6) else H2, t, 0.34, pan=0.25)
            elif s % 2 == 1 and b >= 6: put(drums, H1, t, 0.16, pan=-0.3)
            elif s % 4 == 0 and b >= 6: put(drums, H1, t, 0.12, pan=-0.2)
        # bass: 8분음표 오프비트
        if s % 2 == 0 and not quiet and not last_half:
            oct_ = 12 if s in (6, 14) else 0
            put(music, bass(bm + oct_, 0.21), t, 0.40 if s % 4 else 0.27)
        # pluck arp
        if b >= 6 and not quiet:
            seq = [0, 1, 2, 3, 2, 1, 2, 3, 0, 2, 1, 3, 2, 3, 1, 2]
            m = notes[seq[s] % len(notes)] + 12 + (12 if s in (3, 7, 11, 15) else 0)
            p = pluck(m, 0.4, 0.085)
            pn = -0.55 if s % 2 else 0.55
            put(music, p, t, 0.27, pan=pn); put(send, p, t, 0.15, pan=-pn)
            put(music, p, t + 0.375, 0.10, pan=-pn)              # 점8분 딜레이

# --- S1 (0–4): 문서 하이라이트 스윕
for t, m in ((0.5, 69), (1.0, 72), (1.5, 76), (2.5, 79)):
    put(fx, swish(), t - 0.03, 0.30, pan=-0.3); put(music, pluck(m, 0.7, 0.16), t, 0.30); put(send, pluck(m, 0.7, 0.16), t, 0.22)
put(fx, riser(1.0), 3.0, 0.34)
put(fx, whoosh(0.9, 0.7), 3.15, 0.30)
# --- S1b (4–6): 단어 3개
for i, t in enumerate((4.0, 4.5, 5.0)):
    put(drums, thud(92 + i * 10), t, 0.75); put(drums, H2, t, 0.14)
    put(music, pluck(69 + (0, 3, 7)[i], 0.7, 0.14), t, 0.30); put(send, pluck(69 + (0, 3, 7)[i], 0.7, 0.14), t, 0.2)
put(fx, riser(0.55), 5.45, 0.30)
# --- S2 (6–8): 로고
put(drums, impact(), 6.0, 0.95); put(send, crash(), 6.0, 0.22); put(fx, crash(), 6.0, 0.10)
put(fx, tick(), 6.38, 0.5)
for i, m in enumerate((84, 88, 91)):
    put(music, bell(m), 6.5 + i * 0.03, 0.30); put(send, bell(m), 6.5 + i * 0.03, 0.30)
for i, m in enumerate((72, 76, 79, 84, 86, 88, 91)):
    put(music, pluck(m, 0.4, 0.09), 7.0 + i * 0.0625, 0.22, pan=(i - 3) / 4); put(send, pluck(m, 0.4, 0.09), 7.0 + i * 0.0625, 0.16)
# --- 전환 우쉬 + 히트
for T in (8, 12, 16, 20, 24, 28, 32):
    put(fx, whoosh(0.78, 0.5), T - 0.39, 0.36)
    put(drums, impact(0.7), T, 0.42); put(send, crash(), T, 0.10)
for t in (8.5, 9.5, 10.5):                       # 시간표 블록 팝
    put(music, bell(88 if t < 10 else 91, 0.9), t, 0.16); put(send, bell(88, 0.9), t, 0.14)
put(fx, riser(0.9), 18.0, 0.16)                  # 수강생 카운트업
put(fx, riser(0.9), 21.0, 0.14)                  # 피드백 카운트업
put(music, bell(91, 0.9), 22.0, 0.15); put(send, bell(91, 0.9), 22.0, 0.13)
put(fx, riser(1.0), 28.0, 0.30)                  # '46' 직전
put(drums, impact(), 29.0, 0.9); put(send, crash(), 29.0, 0.2); kick_times.append(29.0)
for i, t in enumerate((29.5, 30.0, 30.5)):
    put(music, bell(86 + i * 2, 0.8), t, 0.15); put(send, bell(86 + i * 2, 0.8), t, 0.12)
# --- S8a (32–36): 세 줄 + '연결되었습니다'
for i, t in enumerate((32.14, 33.0, 34.0)):
    put(drums, thud(100 + i * 12), t, 0.8); put(drums, H2, t, 0.14); kick_times.append(t)
    for j, m in enumerate((72 + (0, 4, 7)[i], 79 + (0, 4, 5)[i])):
        put(music, pluck(m, 0.9, 0.2), t + j * 0.04, 0.26); put(send, pluck(m, 0.9, 0.2), t + j * 0.04, 0.2)
    put(fx, riser(0.6), t + 0.02, 0.10)
put(fx, riser(0.6), 34.42, 0.26)
put(drums, impact(), 35.0, 0.9); put(send, crash(), 35.0, 0.22); kick_times.append(35.0)
for i, m in enumerate((79, 84, 88, 91)):
    put(music, bell(m, 1.4), 35.0 + i * 0.03, 0.22); put(send, bell(m, 1.4), 35.0 + i * 0.03, 0.25)
# --- S8 (36–40): 엔드 카드
put(fx, whoosh(0.7, 0.55), 35.62, 0.34)
put(drums, impact(), 36.0, 0.95); put(send, crash(), 36.0, 0.24)
put(fx, tick(), 36.42, 0.5)
for i, m in enumerate((84, 88, 91, 96)):
    put(music, bell(m, 2.2), 36.5 + i * 0.035, 0.26); put(send, bell(m, 2.2), 36.5 + i * 0.035, 0.3)
for i, m in enumerate((72, 76, 79, 84, 79, 76)):
    put(music, pluck(m, 0.8, 0.18), 37.0 + i * 0.25, 0.20 - i * 0.02, pan=(-1) ** i * 0.4); put(send, pluck(m, 0.8, 0.18), 37.0 + i * 0.25, 0.16)
put(music, bell(96, 2.0), 38.0, 0.16); put(send, bell(96, 2.0), 38.0, 0.2)

# ---------------- mix ----------------
t_all = np.arange(N + TAIL) / SR
duck = np.ones(N + TAIL)
for k in kick_times:
    i = int(k * SR); n = int(0.4 * SR)
    duck[i:i + n] = np.minimum(duck[i:i + n], 1 - 0.55 * np.exp(-np.arange(n) / SR / 0.11))
music *= duck

# reverb (지수 감쇠 노이즈 IR)
ir_t = tt(1.9)
ir = np.stack([filt(noise(1.9), lo=350, hi=6500) * np.exp(-ir_t / 0.5) for _ in range(2)])
ir /= np.sqrt((ir ** 2).sum(axis=1, keepdims=True))
def conv(x, h):
    n = len(x) + len(h) - 1; m = 1 << (n - 1).bit_length()
    return np.fft.irfft(np.fft.rfft(x, m) * np.fft.rfft(h, m), m)[:len(x)]
wet = np.stack([conv(send[c], ir[c]) for c in range(2)])

mix = drums * 1.0 + music * 1.0 + fx * 1.0 + wet * 0.55
mix = mix[:, :N]
e = np.ones(N); nf = int(1.6 * SR); e[-nf:] = np.cos(np.linspace(0, np.pi / 2, nf)) ** 2   # 페이드아웃
mix *= e
mix = np.stack([filt(mix[c], lo=40, order=3) for c in range(2)])
mix = np.tanh(mix * 1.35) / np.tanh(1.35)
mix *= 0.61 / np.abs(mix).max()          # ≈ -14 LUFS, 트루피크 여유 확보

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out', 'track.wav')
os.makedirs(os.path.dirname(out), exist_ok=True)
with wave.open(out, 'wb') as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
    w.writeframes((mix.T * 32767).astype('<i2').tobytes())
print('wrote', out, 'peak', float(np.abs(mix).max()))
