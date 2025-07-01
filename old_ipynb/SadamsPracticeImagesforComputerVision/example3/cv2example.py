import cv2
import matplotlib.pyplot as plt
#%matplotlib inline

# %% [markdown]
# ## Exercise 3: Combining Canny and Template Matching
# Task: Apply Canny edge detection to both the image and template, then perform template matching.

# %% [code] {"execution":{"iopub.status.busy":"2025-06-02T19:35:56.727859Z","iopub.execute_input":"2025-06-02T19:35:56.728208Z","iopub.status.idle":"2025-06-02T19:35:57.355272Z","shell.execute_reply.started":"2025-06-02T19:35:56.728167Z","shell.execute_reply":"2025-06-02T19:35:57.354309Z"}}
# Load image and template in grayscale
img = cv2.imread('car.png', cv2.IMREAD_GRAYSCALE)
template = cv2.imread('car mirror.png', cv2.IMREAD_GRAYSCALE)
if img is None or template is None:
    print("Error: Image or template not found.")
    exit()

# Apply Canny edge detection
img_edges = cv2.Canny(img, 100, 200)
template_edges = cv2.Canny(template, 100, 200)

# Perform template matching on edge images
result = cv2.matchTemplate(img_edges, template_edges, cv2.TM_CCOEFF_NORMED)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

# Draw rectangle on original color image
img_rgb = cv2.cvtColor(cv2.imread('car.png'), cv2.COLOR_BGR2RGB)
w, h = template.shape[::-1]
top_left = max_loc
bottom_right = (top_left[0] + w, top_left[1] + h)
cv2.rectangle(img_rgb, top_left, bottom_right, (255, 0, 0), 8)

# Display results
plt.figure(figsize=(20, 14))

plt.subplot(1, 3, 1)
plt.title('Image Edges')
plt.imshow(img_edges, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title('Template Edges')
plt.imshow(template_edges, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title('Matched Region')
plt.imshow(img_rgb)
plt.axis('off')

# %% [markdown]
# # By **Saddam Umer**
#
# BS AI, SMIU