import pandas as pd
import matplotlib.pyplot as plt
import io

# 1. SETUP: Create the CSV content (Verified Data)
csv_content = """Element Name,Symbol,Atomic Number,Atomic Radius (pm),First Ionization Energy (kJ/mol)
Hydrogen,H,1,53,1312.0
Lithium,Li,3,152,520.2
Sodium,Na,11,186,495.8
Potassium,K,19,227,418.8
Rubidium,Rb,37,248,403.0
Caesium,Cs,55,265,375.7
Francium,Fr,87,348,392.8"""

# 2. READ: Load the CSV into a DataFrame
# (In a real scenario, you would use: df = pd.read_csv('group1.csv'))
df = pd.read_csv(io.StringIO(csv_content))

# 3. DISPLAY: Show the raw data
print("--- Group 1 Elements Data ---")
print(df)

# 4. VISUALIZE: Create a Dual-Axis Plot
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot Atomic Radius (Bar Chart) - Left Axis
color = 'tab:blue'
ax1.set_xlabel('Element Symbol')
ax1.set_ylabel('Atomic Radius (pm)', color=color)
ax1.bar(df['Symbol'], df['Atomic Radius (pm)'], color=color, alpha=0.6, label='Atomic Radius')
ax1.tick_params(axis='y', labelcolor=color)

# Create a second y-axis for Ionization Energy
ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('1st Ionization Energy (kJ/mol)', color=color)
ax2.plot(df['Symbol'], df['First Ionization Energy (kJ/mol)'], color=color, marker='o', linewidth=2, label='Ionization Energy')
ax2.tick_params(axis='y', labelcolor=color)

# Title and Layout
plt.title('Group 1 Trends: Atomic Radius vs. Ionization Energy')
fig.tight_layout()
plt.show()