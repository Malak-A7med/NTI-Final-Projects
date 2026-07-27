# Hotel Cancellation Predictor — Streamlit App

## هيا تشغّليه ازاي

1. شغّلي آخر خلية في النوتبوك (`Save The Model`) عشان يتحفظلك:
   - `hotel_model.pkl`
   - `preprocessor.pkl`
   - `label_encoder.pkl`

2. حطي الـ 3 ملفات دول في نفس الفولدر اللي فيه `app.py`.

3. ثبّتي المكتبات المطلوبة:
   ```bash
   pip install -r requirements.txt
   ```

4. شغّلي التطبيق:
   ```bash
   streamlit run app.py
   ```

5. هتفتح تلقائي في المتصفح على `http://localhost:8501`.

## الفكرة

- بتدخّلي بيانات الحجز من الشريط الجانبي (عدد الضيوف، نوع الغرفة، السعر، الـ lead time...).
- الموديل (XGBoost المُدرّب في النوتبوك) بيتنبأ باحتمالية الإلغاء.
- فيه Gauge تفاعلي بيوضح النسبة، وتفسير SHAP بيوضح ليه الموديل وصل للقرار ده.
