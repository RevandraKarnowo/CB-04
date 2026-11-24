import numpy as np

# ---------------------------------------------------------------
# Model objek (danau) pada citra satelit, resolusi 30 m/piksel.
# Lebar objek w(x) (m) dan garis tengahnya yc(x) (m), x = 0..360 m.
# ---------------------------------------------------------------
PIX = 30.0          # ukuran piksel (m)
LX = 360.0          # panjang objek arah x (m)

def lebar(x):
    u = np.pi * np.asarray(x, float) / LX
    return 140*np.sin(u) + 25*np.sin(3*u) + 10*np.sin(7*u)

def dlebar(x):      # turunan eksak w'(x), untuk uji galat pangkas
    u = np.pi * np.asarray(x, float) / LX
    return (np.pi/LX) * (140*np.cos(u) + 75*np.cos(3*u) + 70*np.cos(7*u))

def pusat(x):
    return 150 + 20*np.sin(2*np.pi*np.asarray(x, float)/LX)

LUAS_EKSAK = (2*LX/np.pi) * (140 + 25/3 + 10/7)   # integral w(x) dx secara analitik

# ---------------------------------------------------------------
# Citra sampel 10 baris x 12 kolom (intensitas 0-255)
# ---------------------------------------------------------------
def buat_citra(seed=4):
    rng = np.random.default_rng(seed)
    baris, kolom = 10, 12
    cit = np.zeros((baris, kolom)); cov = np.zeros((baris, kolom))
    s = 10                                   # supersampling 10x10 per piksel
    off = (np.arange(s) + 0.5) / s
    for r in range(baris):
        for c in range(kolom):
            xs = (c + off) * PIX
            ys = (r + off) * PIX
            XX, YY = np.meshgrid(xs, ys)
            dalam = np.abs(YY - pusat(XX)) < lebar(XX) / 2
            cov[r, c] = dalam.mean()
    cit = 175 + (45 - 175) * cov + rng.normal(0, 3, cov.shape)
    return np.clip(np.round(cit), 0, 255), cov

# ---------------------------------------------------------------
# Beda terpusat (central difference): f'(x) ~ (f(x+h) - f(x-h)) / 2h
# Di tepi citra dipakai beda maju / mundur.
# ---------------------------------------------------------------
def gradien(I, h=1.0):
    gx = np.zeros_like(I); gy = np.zeros_like(I)
    gx[:, 1:-1] = (I[:, 2:] - I[:, :-2]) / (2*h)
    gx[:, 0] = (I[:, 1] - I[:, 0]) / h
    gx[:, -1] = (I[:, -1] - I[:, -2]) / h
    gy[1:-1, :] = (I[2:, :] - I[:-2, :]) / (2*h)
    gy[0, :] = (I[1, :] - I[0, :]) / h
    gy[-1, :] = (I[-1, :] - I[-2, :]) / h
    return gx, gy, np.hypot(gx, gy)

def beda_terpusat(f, x, h):  return (f(x + h) - f(x - h)) / (2*h)
def beda_maju(f, x, h):      return (f(x + h) - f(x)) / h

# ---------------------------------------------------------------
# Integrasi numerik
# ---------------------------------------------------------------
def trapesium(y, h):
    return h * (y[0]/2 + y[1:-1].sum() + y[-1]/2)

def simpson13(y, h):        # n genap
    n = len(y) - 1
    assert n % 2 == 0
    return h/3 * (y[0] + 4*y[1:-1:2].sum() + 2*y[2:-1:2].sum() + y[-1])

def simpson38(y, h):        # n kelipatan 3
    n = len(y) - 1
    assert n % 3 == 0
    jumlah = y[0] + y[-1]
    for i in range(1, n):
        jumlah += 2*y[i] if i % 3 == 0 else 3*y[i]
    return 3*h/8 * jumlah

def simpson_gabungan(y, h):  # n ganjil: 1/3 di (n-3) interval pertama + 3/8 di 3 interval terakhir
    n = len(y) - 1
    if n % 2 == 0:
        return simpson13(y, h)
    return simpson13(y[:n-2], h) + simpson38(y[n-3:], h)

def sampel_lebar(n, bulat=False):
    x = np.linspace(0, LX, n + 1); y = lebar(x)
    return x, (np.round(y) if bulat else y), LX / n

# ---------------------------------------------------------------
def hitung():
    R = {}
    # --- Bagian A: citra + gradien ---
    I, cov = buat_citra()
    gx, gy, G = gradien(I)
    R["I"], R["cov"], R["gx"], R["gy"], R["G"] = I, cov, gx, gy, G
    R["ambang"] = 0.4 * G.max()
    R["tepi"] = G >= R["ambang"]
    R["batas_nyata"] = (cov > 0.02) & (cov < 0.98)      # piksel yang benar-benar dilewati batas objek
    tp = (R["tepi"] & R["batas_nyata"]).sum()
    R["presisi"] = tp / R["tepi"].sum()
    R["recall"] = tp / R["batas_nyata"].sum()
    R["seg"] = I < (45 + 175) / 2                        # segmentasi ambang tengah
    R["luas_piksel"] = R["seg"].sum() * PIX**2
    # --- uji galat pangkas turunan ---
    xs = np.linspace(30, 330, 301)
    R["turunan"] = []
    for h in (30.0, 15.0, 7.5, 3.75):
        e_c = np.max(np.abs(beda_terpusat(lebar, xs, h) - dlebar(xs)))
        e_m = np.max(np.abs(beda_maju(lebar, xs, h) - dlebar(xs)))
        R["turunan"].append((h, e_m, e_c))
    # --- Bagian B: luas ---
    x, w, h = sampel_lebar(12, bulat=True)
    R["x12"], R["w12"], R["h12"] = x, w, h
    R["T12"], R["S13"], R["S38"] = trapesium(w, h), simpson13(w, h), simpson38(w, h)
    # --- konvergensi (tanpa pembulatan) ---
    R["konv"] = []
    for n in (6, 12, 24, 48):
        _, y, hh = sampel_lebar(n)
        R["konv"].append((n, hh, trapesium(y, hh), simpson13(y, hh), simpson38(y, hh)))
    R["gab"] = []
    for n in (9, 15, 21):
        _, y, hh = sampel_lebar(n)
        R["gab"].append((n, hh, simpson_gabungan(y, hh)))
    return R

def grafik(R, simpan=True):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(10, 3.3))
    for a, d, jd in zip(ax, (R["I"], R["G"], R["tepi"].astype(float)),
                        ("Citra sampel", "Magnitudo gradien", "Peta tepi")):
        im = a.imshow(d, cmap="gray"); a.set_title(jd, fontsize=9)
        a.set_xticks(range(0, 12, 2)); a.set_yticks(range(0, 10, 2))
    ax[0].set_ylabel("baris"); ax[1].set_xlabel("kolom")
    fig.tight_layout()
    if simpan: fig.savefig("l_gambar1.png", dpi=200)

    xx = np.linspace(0, LX, 721)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.plot(xx, pusat(xx) + lebar(xx)/2, "k-", lw=1.2, label="kontur objek")
    ax.plot(xx, pusat(xx) - lebar(xx)/2, "k-", lw=1.2)
    ax.fill_between(xx, pusat(xx) - lebar(xx)/2, pusat(xx) + lebar(xx)/2, color="#9ecae1", alpha=.6)
    for xi in R["x12"]:
        ax.plot([xi, xi], [pusat(xi) - lebar(xi)/2, pusat(xi) + lebar(xi)/2], "r-", lw=.8)
    ax.plot([], [], "r-", lw=.8, label="lebar w(x) di 13 titik")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_aspect("equal"); ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    if simpan: fig.savefig("l_gambar2.png", dpi=200)

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    hh = [t[0] for t in R["turunan"]]
    ax[0].loglog(hh, [t[1] for t in R["turunan"]], "ro-", label="beda maju  O(h)")
    ax[0].loglog(hh, [t[2] for t in R["turunan"]], "b^-", label="beda terpusat  O(h²)")
    ax[0].set_xlabel("h (m)"); ax[0].set_ylabel("galat maks. turunan"); ax[0].grid(alpha=.3, which="both"); ax[0].legend(fontsize=8)
    hk = [t[1] for t in R["konv"]]
    ax[1].loglog(hk, [abs(t[2]-LUAS_EKSAK) for t in R["konv"]], "ro-", label="Trapesium")
    ax[1].loglog(hk, [abs(t[3]-LUAS_EKSAK) for t in R["konv"]], "b^-", label="Simpson 1/3")
    ax[1].loglog(hk, [abs(t[4]-LUAS_EKSAK) for t in R["konv"]], "gs--", label="Simpson 3/8")
    ax[1].set_xlabel("h (m)"); ax[1].set_ylabel("galat luas (m²)"); ax[1].grid(alpha=.3, which="both"); ax[1].legend(fontsize=8)
    fig.tight_layout()
    if simpan: fig.savefig("l_gambar3.png", dpi=200)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    np.set_printoptions(linewidth=200, suppress=True)
    R = hitung()
    print("Luas eksak =", LUAS_EKSAK)
    print("Citra:\n", R["I"].astype(int))
    print("Gx (baris 5):", R["gx"][5])
    print("Ambang =", R["ambang"], " presisi = %.2f recall = %.2f" % (R["presisi"], R["recall"]))
    print("Luas hitung piksel =", R["luas_piksel"])
    for t in R["turunan"]: print("h=%.2f maju=%.4f terpusat=%.5f" % t)
    print("Trapesium =", R["T12"], " S1/3 =", R["S13"], " S3/8 =", R["S38"])
    for t in R["konv"]: print("n=%d h=%.2f T=%.2f S13=%.2f S38=%.2f" % t)
    for t in R["gab"]: print("gabungan n=%d h=%.2f A=%.2f" % t)
    grafik(R); plt.show()
