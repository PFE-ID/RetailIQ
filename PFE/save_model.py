import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder

# ── تحميل البيانات ──────────────────────────
df = pd.read_csv("cleaned_data.csv")
df["Date"] = pd.to_datetime(df["Date"])

# ── تدريب الـ Encoders ───────────────────────
le_region   = LabelEncoder().fit(df["Region"])
le_segment  = LabelEncoder().fit(df["Segment"])
le_category = LabelEncoder().fit(df["Category"])
le_subcat   = LabelEncoder().fit(df["Sub_Category"])

# ── تحويل البيانات ───────────────────────────
df_ml = df.copy()
df_ml["Region"]       = le_region.transform(df["Region"])
df_ml["Segment"]      = le_segment.transform(df["Segment"])
df_ml["Category"]     = le_category.transform(df["Category"])
df_ml["Sub_Category"] = le_subcat.transform(df["Sub_Category"])

# ── تدريب النموذج ────────────────────────────
features = ["Region","Segment","Category","Sub_Category",
            "Year","Month","Quantity","Unit_Cost","Unit_Price","Discount"]
X = df_ml[features]
y = df_ml["Sales"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)
model.fit(X_train, y_train)

# ── حفظ الملفين ✅ ───────────────────────────
with open("model_gbr_sales.pkl", "wb") as f:
    pickle.dump(model, f)

with open("encoders.pkl", "wb") as f:
    pickle.dump({
        "region":   le_region,
        "segment":  le_segment,
        "category": le_category,
        "subcat":   le_subcat
    }, f)

print("✅ تم إنشاء الملفين بنجاح!")
print("📁 model_gbr_sales.pkl")
print("📁 encoders.pkl")