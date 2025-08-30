import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ===============================
# 1. Load dataset
# ===============================
df = pd.read_csv("Procurement KPI Analysis Dataset.csv")

# ===============================
# 2. Data Cleaning
# ===============================
# Convert date
df["Order_Date"] = pd.to_datetime(df["Order_Date"])
df["Delivery_Date"] = pd.to_datetime(df["Delivery_Date"], errors="coerce")
# Save to CSV File
df.to_csv("procurement_kpi_cleaned.csv", index=False)


# Create Lead Time Column (if Delivery_Date is available)
df["Lead_Time_Days"] = (df["Delivery_Date"] - df["Order_Date"]).dt.days

# Create Total Cost Column (Quantity × Unit Price)
df["Total_Cost"] = df["Quantity"] * df["Unit_Price"]

# Create Savings Column (difference between Unit Price and Negotiated Price)
df["Cost_Savings"] = (df["Unit_Price"] - df["Negotiated_Price"]) * df["Quantity"]

# Create Defect Rate Column
df["Defect_Rate"] = df["Defective_Units"] / df["Quantity"]

# On-Time Delivery Flag (consider late if Lead Time > median of all suppliers)
median_lead = df["Lead_Time_Days"].median()
df["On_Time_Delivery"] = df["Lead_Time_Days"].apply(lambda x: "Yes" if pd.notnull(x) and x <= median_lead else "No")

# ===============================
# 3. Exploratory Data Analysis (EDA)
# ===============================

## a) Distribusi Lead Time
plt.figure(figsize=(8,5))
sns.histplot(df["Lead_Time_Days"].dropna(), bins=20, kde=True)
plt.title("Delivery Lead Time Distribution")
plt.xlabel("Lead Time (days)")
plt.show()

## b) Monthly Purchase Order Trend
df["Month"] = df["Order_Date"].dt.to_period("M")
orders_per_month = df.groupby("Month")["PO_ID"].nunique()

plt.figure(figsize=(10,5))
orders_per_month.plot(kind="line", marker="o")
plt.title("Monthly Purchase Order Trend")
plt.xlabel("Month")
plt.ylabel("Total PO")
plt.show()

## c) Top 10 Suppliers by Number of Purchase Orders (POs)
top_suppliers = df["Supplier"].value_counts().head(10)

plt.figure(figsize=(10,5))
sns.barplot(x=top_suppliers.index, y=top_suppliers.values)
plt.xticks(rotation=45)
plt.title("Top 10 Suppliers by Number of Purchase Orders (POs)")
plt.ylabel("Total PO")
plt.show()

## d) On-Time Delivery Rate per Supplier
on_time = df.groupby("Supplier")["On_Time_Delivery"].apply(lambda x: (x=="Yes").mean()*100).sort_values(ascending=False).head(10)

plt.figure(figsize=(10,5))
sns.barplot(x=on_time.index, y=on_time.values)
plt.xticks(rotation=45)
plt.title("Top 10 Suppliers with the Highest On-Time Delivery Rate")
plt.ylabel("On-Time Delivery Rate (%)")
plt.show()

## e) Defect Rate per Item Category
defect_rate = df.groupby("Item_Category")["Defect_Rate"].mean().sort_values(ascending=False)

plt.figure(figsize=(8,5))
sns.barplot(x=defect_rate.index, y=defect_rate.values)
plt.xticks(rotation=45)
plt.title("Defect Rate per Item Category")
plt.ylabel("Defect Rate (%)")
plt.show()

## f) Numerical Correlation
plt.figure(figsize=(8,6))
sns.heatmap(df[["Quantity","Unit_Price","Negotiated_Price","Total_Cost","Cost_Savings","Defect_Rate","Lead_Time_Days"]].corr(), annot=True, cmap="coolwarm")
plt.title("Numerical Variable Correlation Heatmap")
plt.show()
