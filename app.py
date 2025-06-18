import streamlit as st
import pandas as pd
import joblib # Untuk memuat model .pkl
import os # Untuk memeriksa keberadaan file model

# --- Konfigurasi Halaman Streamlit ---
st.set_page_config(
    page_title="Prediksi Kelulusan Mahasiswa",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="auto"
)

# --- Judul dan Deskripsi Aplikasi ---
st.title("🎓 Prediksi Kategori Waktu Lulus Mahasiswa")
st.write("Aplikasi ini memprediksi kategori waktu kelulusan (Tepat Waktu/Terlambat) berdasarkan beberapa faktor.")
st.markdown("---")

# --- Muat Model ---
@st.cache_resource # Cache resource untuk menghindari pemuatan ulang model setiap kali aplikasi refresh
def load_model():
    model_path = 'model_graduation.pkl'
    if not os.path.exists(model_path):
        st.error(f"Error: Model '{model_path}' tidak ditemukan. Pastikan file model ada di direktori yang sama.")
        st.stop() # Hentikan aplikasi jika model tidak ditemukan
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Error saat memuat model: {e}")
        st.stop()

# Coba muat model
nb_model = load_model()

# --- Input Data Baru dari Pengguna ---
st.header("Masukkan Data Mahasiswa Baru:")

# Menggunakan kolom untuk tata letak yang lebih baik
col1, col2 = st.columns(2)

with col1:
    new_ACT = st.number_input("Nilai ACT Composite Score:", min_value=1.0, max_value=36.0, value=25.0, step=0.1)
    new_SAT = st.number_input("Nilai SAT Total Score:", min_value=400.0, max_value=1600.0, value=1200.0, step=1.0)
    new_GPA = st.number_input("Nilai Rata-rata SMA (GPA):", min_value=0.0, max_value=4.0, value=3.0, step=0.01)

with col2:
    new_income = st.number_input("Pendapatan Orang Tua (IDR):", min_value=0.0, value=5000000.0, step=100000.0)
    new_education = st.number_input("Tingkat Pendidikan Orang Tua (Angka):", min_value=0, max_value=20, value=12) # Contoh: 12 untuk SMA, 16 untuk S1

# --- Tombol Prediksi ---
st.markdown("---")
if st.button("Prediksi Kategori Waktu Lulus"):
    if nb_model is not None:
        try:
            # Buat DataFrame dari input baru
            new_data_df = pd.DataFrame(
                [[new_ACT, new_SAT, new_GPA, new_income, new_education]],
                columns=['ACT composite score', 'SAT total score', 'high school gpa', 'parental income', 'parent_edu_numerical']
            )

            # Lakukan prediksi
            predicted_code = nb_model.predict(new_data_df)[0]
            
            # Konversi hasil prediksi ke label asli
            label_mapping = {1: 'Tepat Waktu', 0: 'Terlambat'}
            predicted_label = label_mapping.get(predicted_code, 'Tidak diketahui')

            st.subheader("Hasil Prediksi:")
            if predicted_label == 'Tepat Waktu':
                st.success(f"Mahasiswa ini diprediksi akan lulus: **{predicted_label}**")
            else:
                st.warning(f"Mahasiswa ini diprediksi akan lulus: **{predicted_label}**")

            st.markdown("---")
            st.write("Catatan: Prediksi ini didasarkan pada model yang dilatih sebelumnya.")

        except Exception as e:
            st.error(f"Terjadi kesalahan saat memproses prediksi: {e}")
    else:
        st.warning("Model belum dimuat. Pastikan file 'model_graduation.pkl' ada.")

st.markdown("---")
st.caption("Dibuat oleh Data Scientist dengan Streamlit untuk memprediksi kelulusan.")