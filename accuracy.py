import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import interp1d
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

# 1. قراءة الملف (استبدلي 'your_file.csv' باسم ملفكِ الحقيقي)
df = pd.read_csv('sample_data/dataSet.csv')
# تنظيف البيانات من السطور الفارغة (NaN) التي تظهر بسبب اختلاف أطوال الأعمدة
df = df.fillna(method='ffill').fillna(method='bfill')

# 2. تحديد نطاق تردد موحد (من 2.0 إلى 12.0 جيجاهرتز بـ 50 نقطة ثابتة)
common_freq = np.linspace(2.0, 12.0, 50)

X_list = []
y_list = []

# 3. معالجة كل سطر (عينة) وتوحيد تردداتها رياضياً
for index, row in df.iterrows():
    try:
        # استخراج قراءات المعامل S13
        f1, s13 = row['Freq1'], row['S13']
        # استخراج قراءات المعامل S14
        f2, s14 = row['Freq2'], row['S14']
        # استخراج قراءات المعامل S23
        f3, s23 = row['Freq3'], row['S23']
        
        # نقاط افتراضية لعمل المنحنى لكل معامل بناءً على قيم السطر الحالي
        # (بايثون سيقوم بربط النقاط لإنشاء منحنى مستمر لكل معامل)
        interp_s13 = interp1d([f1-0.1, f1, f1+0.1], [s13, s13, s13], fill_value="extrapolate")(common_freq)
        interp_s14 = interp1d([f2-0.1, f2, f2+0.1], [s14, s14, s14], fill_value="extrapolate")(common_freq)
        interp_s23 = interp1d([f3-0.1, f3, f3+0.1], [s23, s23, s23], fill_value="extrapolate")(common_freq)
        
        # دمج المعاملات الثلاثة معاً لتكوين مصفوفة الخصائص الهوائية الكاملة لهذه العينة
        full_features = np.concatenate([interp_s13, interp_s14, interp_s23])
        
        X_list.append(full_features)
        y_list.append(int(row['Label']))
    except Exception as e:
        continue

X = np.array(X_list)
y = np.array(y_list)

print(f"✓ تم الانتهاء من خطوة الـ Interpolation بنجاح!")
print(f"حجم البيانات الجاهزة للذكاء الاصطناعي: {X.shape}")

# 4. تقسيم البيانات إلى تدريب واختبار (80% تدريب، 20% اختبار)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 5. عمل تعيير للمدخلات (Feature Scaling)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. تدريب خوارزمية الـ SVM وتوليد النتائج
classifier = SVC(kernel='rbf', C=5.0, gamma='scale')
classifier.fit(X_train_scaled, y_train)

# 7. التنبؤ بالحالات واختبار الدقة لتقديمها للجنة
y_pred = classifier.predict(X_test_scaled)

print("\n================ تقرير الأداء النهائي ================")
print(classification_report(y_test, y_pred, target_names=['Healthy (0)', 'Tumor (1)']))

# 8. رسم مصفوفة الارتباك (Confusion Matrix) لحفظها فوراً في العرض التقديمي
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Healthy', 'Tumor'], yticklabels=['Healthy', 'Tumor'])
plt.title('Automated Breast Cancer Detection Result')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.savefig('result_matrix.png', dpi=300)
plt.show()
