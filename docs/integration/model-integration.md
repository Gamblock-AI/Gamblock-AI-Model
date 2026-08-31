# Panduan Integrasi Model AI Gamblock-AI

Dokumen ini menjelaskan artifact yang dipakai oleh authority protection Android
dan Windows. Semua pemrosesan harus tetap lokal di perangkat.

## Deployment files

Gunakan file berikut dari `models/`:

| File | Kegunaan |
|---|---|
| `gamblock_logistic_regression.onnx` | Logistic Regression untuk inferensi on-device |
| `gambling_keywords.json` | Ruleset keyword eksplisit |
| `gamblock_hybrid_metadata.json` | Contract bobot, threshold, URL features, dan metrik |
| `gamblock_training_metadata.json` | Metadata hasil training model |

Nama file ONNX, ruleset, dan metadata hybrid adalah deployment contract. Jangan
mengganti nama atau formatnya tanpa memperbarui seluruh client authority.

## Local inference flow

1. Ambil URL dan konten halaman yang sudah committed secara lokal.
2. Bersihkan teks dari title, heading, anchor text, atau DOM/content.
3. Hitung 14 URL features berikut:

   `url_length`, `url_digit_count`, `url_dot_count`, `url_slash_count`,
   `url_hyphen_count`, `url_question_count`, `url_equal_count`,
   `url_keyword_count`, `url_has_number`, `url_has_https`, `url_is_valid`,
   `domain_length`, `subdomain_length`, `suffix_length`.

4. Jalankan model ONNX untuk memperoleh `ml_probability`.
5. Jalankan ruleset secara terpisah untuk memperoleh `rule_score`.
6. Hitung artifact score:

   ```text
   hybrid_score = (0.80 * ml_probability) + (0.20 * rule_score)
   ```

7. Gunakan threshold canonical `0.45` bersama evidence policy berikut:
   - rule eksplisit pada URL atau konten tetap dapat menjadi bukti blocking;
   - model-only blocking memerlukan konten halaman yang committed;
   - untuk model-only blocking, skor teks DOM tanpa URL features juga harus
     mencapai threshold;
   - fitur bentuk URL saja tidak boleh memblokir opaque short link atau
     redirect netral.

8. Jika keputusan final adalah `block`, authority lokal menjalankan blocking dan
   Pattern Interrupt.

## Privacy requirements

- URL, domain, DOM, screenshot, dan browsing history tidak boleh dikirim ke
  backend, website, extension relay, atau cloud API.
- Website dan backend hanya menerima aggregate protection events sesuai
  contract produk.
- Jangan menambahkan field browsing baru ke payload remote.
