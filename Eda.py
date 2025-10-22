import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Loading the dataset
df = sns.load_dataset('titanic')

print("Titanic Dataset Analysis\n")

# Dataset Information
print("\n1. Dataset Overview\n")
print("Shape:", df.shape)
print("\nFirst few rows:")
print(df.head())

print("\nColumn info:")
print(df.info())

print("\nMissing values:")
print(df.isnull().sum())

# Summary Statistics
print("\n2. Summary Statistics\n")

print(df.describe())

# Basic stats for key columns
print("\nAge stats:")
print("Mean:", df['age'].mean())
print("Median:", df['age'].median())
print("Std:", df['age'].std())

print("\nFare stats:")
print("Mean:", df['fare'].mean())
print("Median:", df['fare'].median())
print("Std:", df['fare'].std())

# Visualizations - Distributions

# Age distribution
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.hist(df['age'].dropna(), bins=20, color='skyblue', edgecolor='black')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Count')

# Fare distribution
plt.subplot(1, 2, 2)
plt.hist(df['fare'], bins=20, color='lightgreen', edgecolor='black')
plt.title('Fare Distribution')
plt.xlabel('Fare')
plt.ylabel('Count')

plt.tight_layout()
plt.savefig('distributions.png')
print("\nSaved: distributions.png")
plt.show()

# Survival count
plt.figure(figsize=(6, 4))
survival_counts = df['survived'].value_counts()
plt.bar(['Died', 'Survived'], survival_counts, color=['red', 'green'])
plt.title('Survival Count')
plt.ylabel('Number of Passengers')
plt.savefig('survival_count.png')
print("Saved: survival_count.png")
plt.show()

# Box plots for outliers
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.boxplot(df['age'].dropna())
plt.title('Age Boxplot')
plt.ylabel('Age')

plt.subplot(1, 3, 2)
plt.boxplot(df['fare'])
plt.title('Fare Boxplot')
plt.ylabel('Fare')

plt.subplot(1, 3, 3)
plt.boxplot(df['sibsp'])
plt.title('Siblings/Spouse Boxplot')
plt.ylabel('Count')

plt.tight_layout()
plt.savefig('boxplots.png')
print("Saved: boxplots.png")
plt.show()

# Survival Analysis
# Survival by gender
print("\n3. Survival Analysis\n")


survival_gender = df.groupby('sex')['survived'].mean() * 100
print("\nSurvival rate by gender:")
print(survival_gender)

plt.figure(figsize=(6, 4))
survival_gender.plot(kind='bar', color=['pink', 'lightblue'])
plt.title('Survival Rate by Gender')
plt.ylabel('Survival Rate (%)')
plt.xlabel('Gender')
plt.xticks(rotation=0)
plt.savefig('survival_gender.png')
print("Saved: survival_gender.png")
plt.show()

# Survival by class
survival_class = df.groupby('pclass')['survived'].mean() * 100
print("\nSurvival rate by class:")
print(survival_class)

plt.figure(figsize=(6, 4))
survival_class.plot(kind='bar', color='orange')
plt.title('Survival Rate by Class')
plt.ylabel('Survival Rate (%)')
plt.xlabel('Passenger Class')
plt.xticks(rotation=0)
plt.savefig('survival_class.png')
print("Saved: survival_class.png")
plt.show()

# Correlation Analysis
print("\n4. Correlation Analysis")

# Select numeric columns only
numeric_cols = df.select_dtypes(include=['float64', 'int64'])
correlation = numeric_cols.corr()

print("\nCorrelation with survival:")
print(correlation['survived'].sort_values(ascending=False))

# Correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap')
plt.savefig('correlation.png')
print("Saved: correlation.png")
plt.show()

# Pairplot for relationships
print("\n5. Feature Relationships\n")

# Select important columns for pairplot
cols_to_plot = ['survived', 'pclass', 'age', 'fare']
df_subset = df[cols_to_plot].dropna()

sns.pairplot(df_subset, hue='survived', palette={0: 'red', 1: 'green'})
plt.savefig('pairplot.png')
print("Saved: pairplot.png")
plt.show()

# Additional Analysis - Survival by Age groups
df['age_group'] = pd.cut(df['age'], bins=[0, 18, 35, 60, 100], 
                          labels=['Child', 'Young Adult', 'Adult', 'Senior'])

survival_age = df.groupby('age_group')['survived'].mean() * 100
print("\nSurvival rate by age group:")
print(survival_age)

plt.figure(figsize=(8, 5))
survival_age.plot(kind='bar', color='purple')
plt.title('Survival Rate by Age Group')
plt.ylabel('Survival Rate (%)')
plt.xlabel('Age Group')
plt.xticks(rotation=45)
plt.savefig('survival_age.png')
print("Saved: survival_age.png")
plt.show()

# Key Findings
print("\n6. Key Insights")
print("-"*50)

total_passengers = len(df)
survived = df['survived'].sum()
survival_rate = (survived / total_passengers) * 100

print(f"\nTotal passengers: {total_passengers}")
print(f"Survived: {survived}")
print(f"Overall survival rate: {survival_rate:.2f}%")

print("\nMain observations:")
print("- Women had much higher survival rate than men")
print("- 1st class passengers survived more than 3rd class")
print("- Children had better survival chances")
print("- Higher fare correlates with better survival")


print("\nAnalysis Complete!")
