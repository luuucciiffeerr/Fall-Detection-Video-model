import streamlit as st
import os

st.set_page_config(page_title="FolioFinder", layout="wide")

st.title("📁 FolioFinder")
st.caption("سیستم هوشمند تحلیل اسناد فارسی – پروژه دانشجویی")

DATA_DIR = "data/raw"
os.makedirs(DATA_DIR, exist_ok=True)

# Sidebar
st.sidebar.header("📂 آپلود اسناد")

uploaded_files = st.sidebar.file_uploader(
    "فایل‌های PDF یا Word را انتخاب کنید",
    type=["pdf", "docx"],
    accept_multiple_files=True
)

if uploaded_files:
    for file in uploaded_files:
        path = os.path.join(DATA_DIR, file.name)
        with open(path, "wb") as f:
            f.write(file.read())
    st.sidebar.success("فایل‌ها ذخیره شدند ✅")

# Main
st.subheader("📑 اسناد بارگذاری‌شده")

files = os.listdir(DATA_DIR)
if files:
    for f in files:
        st.write(f"• {f}")
else:
    st.info("هنوز سندی بارگذاری نشده است.")
