from flask import Flask, request, jsonify, render_template
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import pickle, os, re
import pandas as pd
from collections import Counter
import re

app = Flask(__name__)

# ── Load 6 Model ──────────────────────────────────────────────
factory = StemmerFactory()
stemmer = factory.create_stemmer()

nama_aspek = {
    'aspek_1': 'Fitur Medis Aplikasi',
    'aspek_2': 'Pelayanan & Kepuasan',
    'aspek_3': 'Teknis Aplikasi',
}

pipeline = {}
for key in nama_aspek.keys():
    with open(f'models/{key}_relevansi.pkl', 'rb') as f:
        rel = pickle.load(f)
    with open(f'models/{key}_sentimen.pkl', 'rb') as f:
        sen = pickle.load(f)
    pipeline[key] = {
        'nama'     : nama_aspek[key],
        'relevansi': rel,
        'sentimen' : sen,
    }

print('✅ 6 Model berhasil dimuat!')

# ── Fungsi Pembantu ───────────────────────────────────────────
def split_kalimat(teks):
    # Pecah berdasarkan tanda baca DAN kata penghubung
    # yang biasa memisahkan 2 opini berbeda
    kata_penghubung = r'\b(tapi|tetapi|namun|akan tetapi|sedangkan|meskipun|walaupun|padahal|dan|serta)\b'
    
    # Gabung pola tanda baca + kata penghubung
    pola = r'[.!?,;]+|' + kata_penghubung
    
    bagian = re.split(pola, teks, flags=re.IGNORECASE)
    
    # Filter hasil yang kosong atau terlalu pendek
    hasil = []
    for b in bagian:
        if b and b.strip() and len(b.strip()) > 3:
            # Buang jika hanya kata penghubung saja
            if not re.fullmatch(kata_penghubung, b.strip(), flags=re.IGNORECASE):
                hasil.append(b.strip())
    
    return hasil

def prediksi_satu(teks, ulasan_asli=''):
    teks_stem = stemmer.stem(teks.lower())
    hasil = []

    for key, res in pipeline.items():
        X_rel     = res['relevansi']['tfidf'].transform([teks_stem])
        relevansi = res['relevansi']['model'].predict(X_rel)[0]

        if relevansi == 'Relevan':
            X_sen    = res['sentimen']['tfidf'].transform([teks_stem])
            sentimen = res['sentimen']['model'].predict(X_sen)[0]
            hasil.append({
                'ulasan_asli': ulasan_asli,  # ← tambah ini
                'kalimat'    : teks,
                'aspek'      : res['nama'],
                'sentimen'   : sentimen,
            })

    return hasil

def buat_kesimpulan(rows):
    if not rows:
        return "Tidak ada aspek yang terdeteksi dari ulasan ini."

    aspek_sentimen = {}
    for r in rows:
        aspek = r['aspek']
        if aspek not in aspek_sentimen:
            aspek_sentimen[aspek] = []
        aspek_sentimen[aspek].append(r['sentimen'])

    positif_list = []
    negatif_list = []

    for aspek, sentimens in aspek_sentimen.items():
        dominant = Counter(sentimens).most_common(1)[0][0]
        if dominant == 'Positif':
            positif_list.append(aspek)
        elif dominant == 'Negatif':
            negatif_list.append(aspek)

    kesimpulan = "Berdasarkan hasil analisis multi-aspek, "
    if positif_list:
        kesimpulan += f"mayoritas pengguna menunjukkan sentimen positif terhadap aspek {', '.join(positif_list)}"
    if negatif_list:
        if positif_list:
            kesimpulan += f", sedangkan aspek {', '.join(negatif_list)} mendapat opini negatif"
        else:
            kesimpulan += f"aspek {', '.join(negatif_list)} mendapat opini negatif"
    kesimpulan += "."
    return kesimpulan

# ── Stopword Bahasa Indonesia ─────────────────────────────────
STOPWORDS = set([
    'yang','dan','di','ke','dari','untuk','dengan','ini','itu','tidak',
    'ada','adalah','pada','dalam','akan','juga','saya','kami','kita',
    'mereka','anda','bisa','sudah','sudah','atau','jika','karena','tapi',
    'tetapi','namun','sangat','lebih','sudah','belum','sering','selalu',
    'masih','agar','hingga','oleh','setelah','sebelum','antara','seperti',
    'kalau','maka','harus','dapat','nya','lah','kah','pun','lagi','saja',
    'app','aplikasi','mysiloam','siloam','ya','yg','sy','ga','gak','ngga',
    'tidak','bisa','tolong','mohon','minta','please','banget','sekali',
    'nya','ku','mu','si','pak','bu','mas','mbak','dok','dokter',
])

def ekstrak_kata_negatif(rows, aspek_nama):
    """Ekstrak kata paling sering muncul di ulasan negatif per aspek"""
    kata_counter = {}

    for nama in aspek_nama.values():
        kata_counter[nama] = Counter()

    for r in rows:
        if r.get('sentimen') == 'Negatif':
            aspek = r['aspek']
            teks  = r['kalimat'].lower()
            # Bersihkan teks
            teks  = re.sub(r'[^a-zA-Z\s]', ' ', teks)
            kata  = [k for k in teks.split()
                     if k not in STOPWORDS and len(k) > 3]
            kata_counter[aspek].update(kata)

    # Ambil top 50 per aspek
    hasil = {}
    for aspek, counter in kata_counter.items():
        hasil[aspek] = [
            {'text': k, 'value': v}
            for k, v in counter.most_common(50)
            if v > 0
        ]
    return hasil

# ── Routes ────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict/text', methods=['POST'])
def predict_text():
    data = request.get_json()
    teks = data.get('teks', '').strip()
    if not teks:
        return jsonify({'error': 'Teks kosong'}), 400

    kalimat_list = split_kalimat(teks)
    rows = []
    for k in kalimat_list:
        rows.extend(prediksi_satu(k, ulasan_asli=teks))

    return jsonify({
        'rows'      : rows,
        'kesimpulan': buat_kesimpulan(rows),
    })

@app.route('/predict/csv', methods=['POST'])
def predict_csv():
    try:
        file = request.files.get('file')
        if not file:
            return jsonify({'error': 'File tidak ditemukan'}), 400

        df = pd.read_csv(file)
        print(f'Kolom CSV    : {df.columns.tolist()}')
        print(f'Jumlah baris : {len(df)}')

        # Deteksi kolom teks otomatis
        col_teks = None
        for col in df.columns:
            if any(k in col.lower() for k in ['ulasan', 'komentar', 'review', 'teks', 'text', 'content']):
                col_teks = col
                break
        if col_teks is None:
            col_teks = df.columns[0]

        print(f'Kolom dipakai: {col_teks}')

        rows = []
        for teks in df[col_teks].dropna():
            kalimat_list = split_kalimat(str(teks))
            for k in kalimat_list:
                rows.extend(prediksi_satu(k, ulasan_asli=str(teks)))

        print(f'Total hasil  : {len(rows)} baris')

        sentimen_count = Counter(r['sentimen'] for r in rows)
        aspek_count    = {}
        for r in rows:
            aspek = r['aspek']
            sen   = r['sentimen']
            if aspek not in aspek_count:
                aspek_count[aspek] = Counter()
            aspek_count[aspek][sen] += 1

        # Tambahkan word cloud data
        wordcloud_data = ekstrak_kata_negatif(rows, nama_aspek)

        return jsonify({
            'rows'          : rows,
            'sentimen_count': dict(sentimen_count),
            'aspek_count'   : {k: dict(v) for k, v in aspek_count.items()},
            'kesimpulan'    : buat_kesimpulan(rows),
            'wordcloud'     : wordcloud_data,   # ← tambah ini
        })

    except Exception as e:
        print(f'ERROR: {e}')
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
    
# ← tambahkan di sini, paling bawah
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)