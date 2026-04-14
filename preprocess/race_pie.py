import matplotlib.pyplot as plt

# Data
labels = ['01 White', '02 Black']
sizes = [60.37735849, 39.1509434]
colors = ['#ff9999','#66b3ff']  # similar to your example (red & blue)

# Plot
plt.figure(figsize=(5,5))
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)

# Equal aspect ratio ensures the pie chart is circular
plt.axis('equal')  
plt.show()
