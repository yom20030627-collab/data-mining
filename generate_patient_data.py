import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# 設置隨機種子以便重現
np.random.seed(42)
random.seed(42)

class PatientDataGenerator:
    def __init__(self, num_patients=100, days_per_patient=7):
        self.num_patients = num_patients
        self.days_per_patient = days_per_patient
        self.diseases = ['糖尿病', '高血壓', '冠心病', '肺炎', '敗血症', '健康']
        self.genders = ['男', '女']
        exit
    def generate_patients(self):
        """生成患者的基本信息"""
        patients = []
        
        for patient_id in range(1, self.num_patients + 1):
            age = np.random.randint(18, 95)
            gender = random.choice(self.genders)
            
            # 根據年齡分配疾病
            if age > 70:
                disease = random.choices(
                    self.diseases,
                    weights=[0.25, 0.25, 0.15, 0.10, 0.10, 0.15]
                )[0]
            elif age > 50:
                disease = random.choices(
                    self.diseases,
                    weights=[0.15, 0.20, 0.10, 0.10, 0.05, 0.40]
                )[0]
            else:
                disease = random.choices(
                    self.diseases,
                    weights=[0.05, 0.05, 0.02, 0.05, 0.03, 0.80]
                )[0]
            
            patients.append({
                'patient_id': patient_id,
                'age': age,
                'gender': gender,
                'disease': disease,
                'admission_date': datetime.now() - timedelta(days=random.randint(1, 30))
            })
        
        return pd.DataFrame(patients)
    
    def generate_vital_signs(self, patient_df):
        """根據疾病生成生命體徵數據"""
        vital_signs_data = []
        
        for _, patient in patient_df.iterrows():
            patient_id = patient['patient_id']
            disease = patient['disease']
            age = patient['age']
            
            # 根據疾病設定基線值
            baseline = self._get_baseline_values(disease, age)
            
            # 為每個患者生成多天的數據
            for day in range(self.days_per_patient):
                for hour in range(24):
                    timestamp = patient['admission_date'] + timedelta(days=day, hours=hour)
                    
                    # 生成含日變化和隨機波動的生命體徵
                    vital_signs = self._generate_daily_vitals(baseline, hour, day)
                    
                    vital_signs_data.append({
                        'patient_id': patient_id,
                        'age': age,
                        'gender': patient['gender'],
                        'disease': disease,
                        'timestamp': timestamp,
                        'date': timestamp.date(),
                        'hour': hour,
                        'day': day,
                        **vital_signs
                    })
        
        return pd.DataFrame(vital_signs_data)
    
    def _get_baseline_values(self, disease, age):
        """根據疾病類型返回基線生命體徵值"""
        baselines = {
            '糖尿病': {
                'heart_rate': 75 + age/20,
                'sbp': 135 + age/10,
                'dbp': 85 + age/15,
                'resp_rate': 16,
                'spo2': 97,
                'temperature': 36.8,
                'blood_glucose': 150
            },
            '高血壓': {
                'heart_rate': 80 + age/25,
                'sbp': 150 + age/5,
                'dbp': 95 + age/10,
                'resp_rate': 16,
                'spo2': 98,
                'temperature': 36.7,
                'blood_glucose': 100
            },
            '冠心病': {
                'heart_rate': 85 + age/20,
                'sbp': 140 + age/10,
                'dbp': 88 + age/12,
                'resp_rate': 17,
                'spo2': 96,
                'temperature': 36.8,
                'blood_glucose': 105
            },
            '肺炎': {
                'heart_rate': 95 + age/15,
                'sbp': 125 + age/15,
                'dbp': 80 + age/20,
                'resp_rate': 22,
                'spo2': 93,
                'temperature': 38.5,
                'blood_glucose': 110
            },
            '敗血症': {
                'heart_rate': 110 + age/10,
                'sbp': 110 + age/20,
                'dbp': 70 + age/25,
                'resp_rate': 24,
                'spo2': 91,
                'temperature': 39.2,
                'blood_glucose': 160
            },
            '健康': {
                'heart_rate': 70,
                'sbp': 120,
                'dbp': 80,
                'resp_rate': 15,
                'spo2': 98,
                'temperature': 36.5,
                'blood_glucose': 90
            }
        }
        return baselines.get(disease, baselines['健康'])
    
    def _generate_daily_vitals(self, baseline, hour, day):
        """生成每小時的生命體徵數據，包含日變化"""
        # 日變化模式（清晨低，午間高）
        circadian_factor = 1 + 0.1 * np.sin((hour - 6) * np.pi / 12)
        
        # 隨機波動
        noise = np.random.normal(0, 0.08)
        
        # 日趨勢（前3天逐漸改善或惡化）
        trend_factor = 1 - (day * 0.02) if day < 3 else 1
        
        return {
            'heart_rate': max(40, baseline['heart_rate'] * circadian_factor * trend_factor + np.random.normal(0, 5)),
            'sbp': max(80, baseline['sbp'] * circadian_factor * trend_factor + np.random.normal(0, 8)),
            'dbp': max(50, baseline['dbp'] * circadian_factor * trend_factor + np.random.normal(0, 5)),
            'resp_rate': max(10, baseline['resp_rate'] + np.random.normal(0, 2)),
            'spo2': min(100, max(85, baseline['spo2'] - (np.random.normal(0, 2) if day > 2 else 0))),
            'temperature': baseline['temperature'] + np.random.normal(0, 0.3),
            'blood_glucose': max(60, baseline['blood_glucose'] + np.random.normal(0, 15))
        }

def main():
    """主函數 - 生成數據並保存到CSV文件"""
    print("開始生成模擬病患數據...")
    
    # 創建生成器
    generator = PatientDataGenerator(num_patients=100, days_per_patient=7)
    
    # 生成患者基本信息
    print("生成患者基本信息...")
    patients_df = generator.generate_patients()
    
    # 生成生命體徵數據
    print("生成生命體徵數據...")
    vitals_df = generator.generate_vital_signs(patients_df)
    
    # 保存為CSV文件
    patients_df.to_csv('patients_info.csv', index=False, encoding='utf-8-sig')
    vitals_df.to_csv('patients_vitals.csv', index=False, encoding='utf-8-sig')
    print("CSV 已成功輸出到Downloads 資料夾")
    
    print("\n✅ 數據生成完成！")
    print(f"患者數量: {len(patients_df)}")
    print(f"生命體徵記錄數: {len(vitals_df)}")
    print(f"\n文件已保存:")
    print("  - patients_info.csv (患者基本信息)")
    print("  - patients_vitals.csv (患者生命體徵)")
    
    # 顯示統計摘要
    print("\n" + "="*60)
    print("患者疾病分佈：")
    print(patients_df['disease'].value_counts())
    
    print("\n患者年齡統計：")
    print(f"平均年齡: {patients_df['age'].mean():.1f}")
    print(f"年齡範圍: {patients_df['age'].min()}-{patients_df['age'].max()}")
    
    print("\n性別分佈：")
    print(patients_df['gender'].value_counts())
    
    print("\n生命體徵統計摘要：")
    print(vitals_df[['heart_rate', 'sbp', 'dbp', 'resp_rate', 'spo2', 'temperature', 'blood_glucose']].describe())
    
    return patients_df, vitals_df

if __name__ == "__main__":
    patients_df, vitals_df = main()

    import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    roc_curve, precision_recall_curve, f1_score, accuracy_score
)
import warnings
warnings.filterwarnings('ignore')

class DiseasePredictor:
    """疾病預測模型類"""
    
    def __init__(self, data_path='patient_monitoring_data.csv'):
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        self.load_data(data_path)
    
    def load_data(self, data_path):
        """加載數據"""
        print("📂 加載數據...")
        try:
            self.df = pd.read_csv(data_path)
            print(f"✅ 成功加載 {len(self.df)} 條記錄")
        except FileNotFoundError:
            print(f"❌ 找不到文件: {data_path}")
            raise
    
    def preprocess_data(self):
        """數據預處理"""
        print("\n🔄 開始數據預處理...")
        
        # 刪除缺失值
        self.df = self.df.dropna()
        
        # 轉換timestamp為datetime
        if 'timestamp' in self.df.columns:
            self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        
        print(f"✅ 預處理後數據量: {len(self.df)}")
    
    def engineer_features(self):
        """特徵工程"""
        print("\n⚙️  進行特徵工程...")
        
        df = self.df.copy()
        
        # 1. 生命體徵特徵
        print("  • 計算生命體徵特徵...")
        
        # 脈搏壓 (Pulse Pressure)
        df['pulse_pressure'] = df['systolic_bp'] - df['diastolic_bp']
        
        # 平均動脈壓 (Mean Arterial Pressure)
        df['map'] = (df['diastolic_bp'] + (df['systolic_bp'] - df['diastolic_bp']) / 3)
        
        # 心率血壓乘積 (Heart Rate × Systolic BP)
        df['hr_sbp_product'] = df['heart_rate'] * df['systolic_bp']
        
        # 脈搏氧合指數 (SpO2 to HR Ratio)
        df['spo2_hr_ratio'] = df['oxygen_saturation'] / (df['heart_rate'] + 1)
        
        # 2. 按患者計算統計特徵
        print("  • 計算患者級別特徵...")
        
        patient_stats = df.groupby('patient_id').agg({
            'heart_rate': ['mean', 'std', 'min', 'max'],
            'systolic_bp': ['mean', 'std', 'min', 'max'],
            'diastolic_bp': ['mean', 'std'],
            'body_temperature': ['mean', 'std'],
            'blood_glucose': ['mean', 'std'],
            'oxygen_saturation': ['mean', 'min'],
            'respiratory_rate': ['mean', 'std'],
            'lactate': ['mean', 'max'],
            'creatinine': ['mean', 'max']
        }).round(2)
        
        # 展平列名
        patient_stats.columns = ['_'.join(col).strip() for col in patient_stats.columns.values]
        
        # 3. 危險指標
        print("  • 計算危險指標...")
        
        # SIRS 標準 (Systemic Inflammatory Response Syndrome)
        df['sirs_score'] = 0
        df.loc[df['body_temperature'] > 38.5, 'sirs_score'] += 1
        df.loc[df['body_temperature'] < 36, 'sirs_score'] += 1
        df.loc[df['heart_rate'] > 90, 'sirs_score'] += 1
        df.loc[df['respiratory_rate'] > 20, 'sirs_score'] += 1
        
        # SOFA 評分簡化版
        df['sofa_simplified'] = 0
        df.loc[df['creatinine'] > 1.2, 'sofa_simplified'] += 1
        df.loc[df['oxygen_saturation'] < 93, 'sofa_simplified'] += 1
        df.loc[df['gcs_score'] < 15, 'sofa_simplified'] += 1
        df.loc[df['systolic_bp'] < 100, 'sofa_simplified'] += 1
        
        # 4. 時間特徵
        print("  • 計算時間特徵...")
        df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)
        
        self.df = df
        self.feature_names = self._get_feature_columns()
        
        print(f"✅ 生成 {len(self.feature_names)} 個特徵")
        return patient_stats
    
    def _get_feature_columns(self):
        """獲取所有特徵列"""
        exclude_cols = ['patient_id', 'disease', 'gender', 'admission_datetime', 
                       'timestamp', 'day_of_week', 'age', 'bmi']
        feature_cols = [col for col in self.df.columns if col not in exclude_cols]
        return feature_cols
    
    def prepare_data_for_modeling(self):
        """準備建模數據"""
        print("\n🎯 準備建模數據...")
        
        df = self.df.copy()
        
        # 編碼目標變量
        df['disease_encoded'] = self.label_encoder.fit_transform(df['disease'])
        
        # 性別編碼
        gender_mapping = {'男': 0, '女': 1}
        df['gender_encoded'] = df['gender'].map(gender_mapping)
        
        # 獲取特徵和目標
        X = df[self.feature_names]
        y = df['disease_encoded']
        
        # 標準化特徵
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=self.feature_names)
        
        # 分割訓練和測試集
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"✅ 訓練集大小: {len(self.X_train)}")
        print(f"✅ 測試集大小: {len(self.X_test)}")
        print(f"✅ 類別分佈: {dict(zip(self.label_encoder.classes_, np.bincount(y)))}")
    
    def build_models(self):
        """構建多個預測模型"""
        print("\n🤖 構建預測模型...")
        
        model_configs = {
            '邏輯回歸': LogisticRegression(max_iter=1000, random_state=42),
            '隨機森林': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            '梯度提升': GradientBoostingClassifier(n_estimators=100, random_state=42),
            '支持向量機': SVC(kernel='rbf', probability=True, random_state=42)
        }
        
        for model_name, model in model_configs.items():
            print(f"  • 訓練 {model_name}...")
            model.fit(self.X_train, self.y_train)
            self.models[model_name] = model
        
        print(f"✅ 完成 {len(self.models)} 個模型的訓練")
    
    def evaluate_models(self):
        """評估模型性能"""
        print("\n📊 模型評估結果:")
        print("=" * 80)
        
        results = {}
        
        for model_name, model in self.models.items():
            print(f"\n【{model_name}】")
            
            # 預測
            y_pred = model.predict(self.X_test)
            y_pred_proba = model.predict_proba(self.X_test)
            
            # 計算指標
            accuracy = accuracy_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred, average='weighted')
            
            print(f"  • 準確率: {accuracy:.4f}")
            print(f"  • F1分數: {f1:.4f}")
            
            # 交叉驗證
            cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5)
            print(f"  • 交叉驗證分數: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
            
            results[model_name] = {
                'accuracy': accuracy,
                'f1': f1,
                'cv_scores': cv_scores,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
            
            # 詳細分類報告
            print(f"\n  分類報告:")
            class_names = self.label_encoder.classes_
            report = classification_report(
                self.y_test, y_pred, 
                target_names=class_names,
                digits=4
            )
            print(f"  {report}")
        
        print("=" * 80)
        return results
    
    def feature_importance(self):
        """獲取特徵重要性"""
        print("\n🔍 特徵重要性分析:")
        print("=" * 80)
        
        importances_dict = {}
        
        for model_name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                feature_importance_df = pd.DataFrame({
                    'feature': self.feature_names,
                    'importance': importances
                }).sort_values('importance', ascending=False)
                
                importances_dict[model_name] = feature_importance_df
                
                print(f"\n【{model_name}】 Top 15 重要特徵:")
                print(feature_importance_df.head(15).to_string(index=False))
        
        return importances_dict
    
    def predict_disease(self, patient_data):
        """預測單個患者的疾病"""
        print("\n🔮 疾病預測:")
        
        # 標準化輸入數據
        patient_scaled = self.scaler.transform([patient_data])
        
        predictions = {}
        for model_name, model in self.models.items():
            pred_class = model.predict(patient_scaled)[0]
            pred_proba = model.predict_proba(patient_scaled)[0]
            
            disease_name = self.label_encoder.inverse_transform([pred_class])[0]
            
            predictions[model_name] = {
                'disease': disease_name,
                'confidence': pred_proba[pred_class],
                'probabilities': dict(zip(self.label_encoder.classes_, pred_proba))
            }
        
        return predictions
    
    def visualize_results(self, results):
        """可視化結果"""
        print("\n📈 生成可視化圖表...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('疾病預測模型性能評估', fontsize=16, fontweight='bold')
        
        # 1. 準確率比較
        ax1 = axes[0, 0]
        model_names = list(results.keys())
        accuracies = [results[m]['accuracy'] for m in model_names]
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
        ax1.bar(model_names, accuracies, color=colors)
        ax1.set_ylabel('準確率')
        ax1.set_title('模型準確率比較')
        ax1.set_ylim([0, 1])
        for i, v in enumerate(accuracies):
            ax1.text(i, v + 0.02, f'{v:.3f}', ha='center')
        
        # 2. 混淆矩陣 (使用最佳模型)
        best_model = max(results.items(), key=lambda x: x[1]['accuracy'])[0]
        ax2 = axes[0, 1]
        cm = confusion_matrix(self.y_test, results[best_model]['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2)
        ax2.set_title(f'混淆矩陣 ({best_model})')
        ax2.set_xlabel('預測')
        ax2.set_ylabel('真實')
        
        # 3. F1 分數
        ax3 = axes[1, 0]
        f1_scores = [results[m]['f1'] for m in model_names]
        ax3.bar(model_names, f1_scores, color=colors)
        ax3.set_ylabel('F1 分數')
        ax3.set_title('F1 分數比較')
        ax3.set_ylim([0, 1])
        for i, v in enumerate(f1_scores):
            ax3.text(i, v + 0.02, f'{v:.3f}', ha='center')
        
        # 4. 交叉驗證分數
        ax4 = axes[1, 1]
        cv_data = [results[m]['cv_scores'] for m in model_names]
        bp = ax4.boxplot(cv_data, labels=model_names, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        ax4.set_ylabel('交叉驗證分數')
        ax4.set_title('模型穩定性評估')
        
        plt.tight_layout()
        plt.savefig('disease_prediction_results.png', dpi=300, bbox_inches='tight')
        print("✅ 圖表已保存: disease_prediction_results.png")
        plt.show()

def main():
    """主函數"""
    print("="*80)
    print("疾病預測模型構建系統")
    print("="*80)
    
    try:
        # 1. 初始化和加載數據
        predictor = DiseasePredictor('patient_monitoring_data.csv')
        
        # 2. 數據預處理
        predictor.preprocess_data()
        
        # 3. 特徵工程
        patient_stats = predictor.engineer_features()
        
        # 4. 準備建模數據
        predictor.prepare_data_for_modeling()
        
        # 5. 構建模型
        predictor.build_models()
        
        # 6. 評估模型
        results = predictor.evaluate_models()
        
        # 7. 特徵重要性
        importances = predictor.feature_importance()
        
        # 8. 可視化
        predictor.visualize_results(results)
        
        # 9. 示例預測
        print("\n" + "="*80)
        print("示例預測")
        print("="*80)
        
        # 使用測試集中的第一個樣本進行預測
        sample_patient = predictor.X_test.iloc[0].values
        predictions = predictor.predict_disease(sample_patient)
        
        print(f"\n患者預測結果:")
        for model_name, pred in predictions.items():
            print(f"\n  【{model_name}】")
            print(f"    預測疾病: {pred['disease']}")
            print(f"    信心度: {pred['confidence']:.2%}")
            print(f"    各疾病概率:")
            for disease, prob in pred['probabilities'].items():
                print(f"      - {disease}: {prob:.2%}")
        
        print("\n" + "="*80)
        print("✅ 模型構建完成！")
        print("="*80)
        
        return predictor, results
        
    except Exception as e:
        print(f"\n❌ 出錯: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    predictor, results = main()