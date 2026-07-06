import streamlit as st
import torch
import timm
import json
import os
from datetime import datetime
from torchvision import transforms
from PIL import Image

# Настройка страницы
st.set_page_config(page_title="Диагностика болезней растений", layout="centered")

st.title("🌿 Диагностика болезней растений")
st.markdown("**Система компьютерного зрения для анализа состояния листьев**")

# Предупреждение
st.warning("⚠️ **Внимание:** Модель является учебным прототипом. Не используйте для реальной агрономической диагностики!")

# === ЗАГРУЗКА КЛАССОВ ===
CLASSES_FILE = 'classes.json'
if not os.path.exists(CLASSES_FILE):
    st.error(f"❌ Файл {CLASSES_FILE} не найден. Положите его рядом с app.py")
    st.stop()

with open(CLASSES_FILE, 'r', encoding='utf-8') as f:
    CLASSES = json.load(f)

# === ЗАГРУЗКА МОДЕЛИ ===
MODEL_PATH = 'best_efficientnet_b0.pth'

@st.cache_resource
def load_model():
    model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=len(CLASSES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()
    return model

try:
    model = load_model()
    st.success(f"✅ Модель загружена: EfficientNet-B0 ({len(CLASSES)} классов)")
except Exception as e:
    st.error(f"❌ Ошибка загрузки модели: {e}")
    st.stop()

# Трансформация
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# === БОКОВАЯ ПАНЕЛЬ ===
st.sidebar.header("📊 Статистика")
HISTORY_FILE = "history.json"

if os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        history = json.load(f)
    st.sidebar.metric("Всего проверок", len(history))
else:
    history = []

# === ЗАГРУЗКА ИЗОБРАЖЕНИЯ ===
st.subheader("📷 Загрузка изображения")
uploaded_file = st.file_uploader("Загрузите фотографию листа", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns(2)
    
    with col1:
        img = Image.open(uploaded_file).convert('RGB')
        st.image(img, caption="Загруженное изображение", use_container_width=True)
    
    if st.button("🔍 Провести диагностику", type="primary"):
        with st.spinner("Анализ изображения..."):
            inputs = transform(img).unsqueeze(0)
            with torch.no_grad():
                outputs = model(inputs)
                probs = torch.nn.functional.softmax(outputs[0], dim=0)
            
            top_probs, top_classes = probs.topk(3)
            
            with col2:
                st.subheader("📋 Результаты диагностики")
                
                top_class_name = CLASSES[top_classes[0].item()]
                top_confidence = top_probs[0].item() * 100
                
                st.success(f"**Диагноз:** {top_class_name}")
                st.metric("Уверенность модели", f"{top_confidence:.2f}%")
                
                if 'healthy' in top_class_name.lower():
                    st.info("✅ Растение здорово")
                else:
                    disease_name = top_class_name.split('___')[-1].replace('_', ' ')
                    st.error(f"⚠️ Обнаружено заболевание: **{disease_name}**")
                
                st.markdown("**Топ-3 предсказания:**")
                for i in range(3):
                    cls_name = CLASSES[top_classes[i].item()]
                    conf = top_probs[i].item() * 100
                    st.text(f"{i+1}. {cls_name} — {conf:.2f}%")
            
            # Сохранение в историю
            result_record = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "file": uploaded_file.name,
                "result": CLASSES[top_classes[0].item()],
                "confidence": f"{top_probs[0].item()*100:.2f}%"
            }
            
            history.append(result_record)
            with open(HISTORY_FILE, "w", encoding="utf-8") as f:
                json.dump(history, f, ensure_ascii=False, indent=2)
            
            # Экспорт в CSV
            st.markdown("---")
            st.subheader("📥 Экспорт результатов")
            
            csv_data = "Дата,Файл,Результат,Уверенность\n"
            for h in history:
                csv_data += f"{h['timestamp']},{h['file']},{h['result']},{h['confidence']}\n"
            
            st.download_button(
                label="📥 Скачать историю (CSV)",
                data=csv_data,
                file_name=f"diagnosis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
            
            # Ограничения модели
            with st.expander("⚠️ Ограничения модели"):
                st.markdown("""
                **Модель может ошибаться в следующих случаях:**
                - Плохое освещение (темные или пересвеченные фото)
                - Размытые изображения
                - Частично закрытые листья
                - Сложный фон
                - Похожими заболеваниями разных культур
                
                **Рекомендации:**
                - Делайте фото при хорошем освещении
                - Фокусируйтесь на листе
                - Используйте простой фон
                """)

else:
    st.info("👆 Загрузите изображение для начала диагностики")

# Footer
st.markdown("---")
st.markdown("**Разработано в рамках учебной практики по компьютерному зрению** | Благовисная В.О. | МТУСИ, 2026")