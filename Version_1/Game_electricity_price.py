import numpy as np
import cv2
import matplotlib.pyplot as plt

#Load the day-ahead prices for January 1st
prices_day_1 = np.loadtxt('France_price_2022_F.csv', delimiter=',', usecols=1, skiprows=0, max_rows=24)
print(prices_day_1.shape)
time = range(0, 24)

# Create array of battery state of charge (SoC) for the day
SoC = [50 for i in range(len(time))]

# Create array of profits for the day
profits = [0 for i in range(len(time))]

# Plot prices as a function of time
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 5), dpi=150)
ax1.plot(time, prices_day_1, 'o-', color='black')
ax1.set_xlabel('')
ax1.set_ylabel('Price ($)', fontsize=7)
ax1.set_title('Prices on Day 1', fontsize=11)

# Plot battery state of charge (SoC) as a function of time
ax2.plot(time, SoC, 'o-', color='black')
ax2.set_xlabel('')
ax2.set_ylabel('Battery state of charge (SoC)', fontsize=7)
ax2.set_title('', fontsize=11)

# Plot profits as a function of time
ax3.plot(time, profits, 'o-', color='black')
ax3.set_xlabel('Time (hours)', fontsize=7)
ax3.set_ylabel('Profit ($)', fontsize=7)
ax3.set_title('', fontsize=11)

# Convert plots to image
fig.canvas.draw()
img = np.array(fig.canvas.renderer.buffer_rgba())

# Display image and color points on key press
i = 0
while i <= len(time):
    # Display image
    cv2.imshow('a', img)
    key = cv2.waitKey(0)
    
    # Compute new battery state of charge based on battery behavior
    if key == ord('d') and SoC[i] > 0:
        # Set battery behavior to discharge
        SoC[i+1] = SoC[i] - 25
        profits[i+1] = profits[i] + prices_day_1[i]
        ax1.plot(time[i], prices_day_1[i], 'o', color='blue')
        i += 1
    elif key == ord('c') and SoC[i] < 100:
        # Set battery behavior to charge
        SoC[i+1] = SoC[i] + 25
        profits[i+1] = profits[i] - prices_day_1[i]
        ax1.plot(time[i], prices_day_1[i], 'o', color='yellow')
        i += 1
    elif key == ord('h'):
        # Set battery behavior to neutral
        SoC[i+1] = SoC[i]
        profits[i+1] = profits[i]
        ax1.plot(time[i], prices_day_1[i], 'o', color='grey')
        i += 1
    elif key == ord('k'):
        i = len(time)+1

    # Update SoC plot
    ax2.clear()
    ax2.plot(time, SoC, 'o-', color='black')
    ax2.set_xlabel('')
    ax2.set_ylabel('Battery state of charge (SoC)', fontsize=7)
    ax2.set_title('')

    # Update profit plot
    ax3.clear()
    ax3.plot(time, profits, 'o-', color='black')
    ax3.set_xlabel('Time (hours)', fontsize=7)
    ax3.set_ylabel('Profits (Euros)', fontsize=7)
    ax3.set_title('')
    
    # Convert updated plots to image
    fig.canvas.draw()
    img = np.array(fig.canvas.renderer.buffer_rgba())
    
    # Display updated image
    cv2.imshow('a', img)
    cv2.waitKey(10)
    # Save the figure at the last time step
    fig.savefig('Trading_user_run.png')

cv2.destroyAllWindows()
