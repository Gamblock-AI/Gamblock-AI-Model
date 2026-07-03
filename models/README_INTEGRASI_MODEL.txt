
PANDUAN INTEGRASI MODEL AI GAMBLOCK-AI

File yang digunakan developer:
1. gamblock_logistic_regression.onnx
   Model Logistic Regression untuk menghasilkan probabilitas situs judi.

2. gambling_keywords.json
   Daftar keyword untuk Rule-Based System.

3. gamblock_hybrid_metadata.json
   Metadata model berisi bobot hybrid, threshold, fitur URL, ukuran model, dan metrik evaluasi.

Alur integrasi:
1. Aplikasi mengambil URL dan konten halaman/DOM.
2. Teks dari title, heading, anchor text, atau content dibersihkan.
3. Aplikasi mengekstrak fitur URL sesuai daftar URL_FEATURE_COLUMNS:
   ['url_length', 'url_digit_count', 'url_dot_count', 'url_slash_count', 'url_hyphen_count', 'url_question_count', 'url_equal_count', 'url_keyword_count', 'url_has_number', 'url_has_https', 'url_is_valid', 'domain_length', 'subdomain_length', 'suffix_length']

4. Model ONNX menghasilkan probabilitas judi.
5. Rule-Based System menghasilkan rule_score.
6. Aplikasi menghitung hybrid_score:

   hybrid_score = (0.75 * ml_probability) + (0.25 * rule_score)

7. Jika hybrid_score >= 0.4, situs diklasifikasikan sebagai judi.
8. Jika situs terdeteksi judi, aplikasi menjalankan mekanisme blokir dan Pattern Interrupt.

Catatan:
- ONNX digunakan untuk bagian Logistic Regression.
- Rule-Based System tetap dihitung terpisah menggunakan gambling_keywords.json.
- Metadata hybrid digunakan agar developer memakai bobot dan threshold yang sama dengan hasil training.
